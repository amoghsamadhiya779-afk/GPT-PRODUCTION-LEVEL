# tests/test_security.py
"""Regression tests for the security/robustness fixes in the serving layer.

Each test pins down a concrete failure that existed before:
engine-lock leaks, event-loop stalls, admin-only data, request limits,
untrusted-content handling, and adapter validation.
"""

import json
import os
import socket
import threading
import time
import urllib.request

import pytest
import torch
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.api import app, build_prompt_with_budget
from app.search import sanitize_link
from app.security import client_ip
from model.tokenizer import GPT2Tokenizer


@pytest.fixture
def client():
    with TestClient(app) as c:
        app.state.status = "active"  # serve with the placeholder engine
        yield c


@pytest.fixture
def live_server():
    """A real uvicorn server, needed to exercise genuine client disconnects."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert time.time() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    app.state.status = "active"
    yield server.servers[0].sockets[0].getsockname()[1]
    server.should_exit = True
    thread.join(timeout=10)


def _engine_idle() -> bool:
    stats = app.state.engine.batcher.stats()
    return stats["active"] == 0 and stats["waiting"] == 0


def _wait_idle(seconds: float = 10.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _engine_idle():
            return True
        time.sleep(0.1)
    return False


def _open_stream(port: int, payload: dict) -> socket.socket:
    body = json.dumps(payload).encode()
    sock = socket.create_connection(("127.0.0.1", port))
    sock.sendall(
        b"POST /generate/stream HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
        b"Content-Length: %d\r\n\r\n" % len(body) + body
    )
    return sock


SOURCE = [{"title": "Mars", "snippet": "Mars is the fourth planet from the Sun.", "link": "https://mars.test"}]


# ── Engine gate / streaming ──────────────────────────────────────────


def test_stream_disconnect_frees_the_batch_slot(live_server, monkeypatch):
    """Disconnecting right after the `sources` event used to leak the engine
    lock forever, so every later generation request returned 503. Now the
    request must be cancelled and its batch slot freed."""
    monkeypatch.setattr("app.search.web_search", lambda q, max_results=3: SOURCE)
    assert _engine_idle()

    sock = _open_stream(live_server, {
        "prompt": "Tell me about Mars", "max_new_tokens": 200, "min_new_tokens": 0, "web_search": True,
    })
    received = b""
    while b"sources" not in received:
        chunk = sock.recv(4096)
        assert chunk, "server closed the stream before sending sources"
        received += chunk
    sock.close()

    assert _wait_idle(), "generation still running 10s after the client disconnected"


def test_web_search_does_not_block_event_loop(live_server, monkeypatch):
    """The streaming endpoint used to call the (blocking, up to ~20s) web
    search directly on the event loop, freezing every other request."""
    def slow_search(query, max_results=3):
        time.sleep(2.0)
        return SOURCE

    monkeypatch.setattr("app.search.web_search", slow_search)
    sock = _open_stream(live_server, {
        "prompt": "Tell me about Mars", "max_new_tokens": 5, "min_new_tokens": 0, "web_search": True,
    })
    try:
        time.sleep(0.3)  # the stream request is now inside the slow search
        start = time.time()
        with urllib.request.urlopen(f"http://127.0.0.1:{live_server}/health", timeout=5) as resp:
            assert resp.status == 200
        assert time.time() - start < 1.0, "health check was blocked behind the web search"
    finally:
        sock.close()


def test_stream_error_does_not_leak_exception_text(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal path /srv/models/x.pt")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(app.state.engine, "stream_text", boom)
    resp = client.post("/generate/stream", json={"prompt": "hi", "max_new_tokens": 5, "min_new_tokens": 0})
    assert resp.status_code == 200
    assert "secret internal path" not in resp.text
    assert "Generation failed" in resp.text
    assert _wait_idle()


# ── Request handling ─────────────────────────────────────────────────


def _request(headers: dict, peer: str = "10.0.0.1") -> Request:
    return Request({
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (peer, 1234),
    })


def test_client_ip_ignores_spoofable_forwarded_entries(monkeypatch):
    spoofed = {"X-Forwarded-For": "1.1.1.1, 203.0.113.7"}  # client-sent, then proxy-appended
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    assert client_ip(_request(spoofed)) == "10.0.0.1"  # no proxy trusted: TCP peer
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    assert client_ip(_request(spoofed)) == "203.0.113.7"  # never the leftmost, spoofable one
    assert client_ip(_request({})) == "10.0.0.1"


def test_oversized_request_body_is_rejected(client, monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_BYTES", "1000")
    resp = client.post("/feedback", json={"prompt": "x" * 2000, "response": "y", "rating": "up"})
    assert resp.status_code == 413


def test_feedback_fields_are_bounded(client):
    resp = client.post("/feedback", json={"prompt": "x" * 5000, "response": "y", "rating": "up"})
    assert resp.status_code == 422


def test_security_headers_present(client):
    resp = client.get("/health")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"


def test_health_does_not_leak_error_details(client):
    app.state.status = "error"
    app.state.error_msg = "Traceback: /home/secret/path"
    resp = client.get("/health")
    assert resp.status_code == 503
    assert "secret" not in resp.text
    app.state.status = "active"


# ── Untrusted content ────────────────────────────────────────────────


@pytest.mark.parametrize("href, expected", [
    ("https://example.com/a?b=1", "https://example.com/a?b=1"),
    ("javascript:alert(document.cookie)", ""),
    ("JaVaScRiPt:alert(1)", ""),
    ("data:text/html,<script>alert(1)</script>", ""),
    ("//example.com/page", "https://example.com/page"),
    ("//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.example%2Fx&rut=abc", "https://real.example/x"),
    ("//duckduckgo.com/l/?uddg=javascript%3Aalert(1)", ""),
    ("/relative/path", ""),
])
def test_search_links_are_sanitized(href, expected):
    assert sanitize_link(href) == expected


def test_search_cache_is_bounded(monkeypatch):
    import app.search as search

    monkeypatch.setattr(search, "_search_cache", type(search._search_cache)())
    monkeypatch.setattr(search, "CACHE_MAX_ENTRIES", 5)
    for i in range(20):
        search._set_in_cache(f"query {i}", [{"title": "t", "snippet": "s", "link": ""}])
    assert len(search._search_cache) == 5
    assert search._get_from_cache("query 19") is not None
    assert search._get_from_cache("query 0") is None


def test_special_token_strings_in_prompts_stay_plain_text(client, monkeypatch):
    """A user typing <|endoftext|> must not inject the real control token."""
    tok = GPT2Tokenizer()
    assert tok.eos_id not in tok.encode_ordinary("hi <|endoftext|> there")

    seen = {}
    import app.inference as inference

    class SpyHandle(inference.GenerationHandle):
        def __init__(self, prompt_ids, *args, **kwargs):
            seen["ids"] = prompt_ids
            super().__init__(prompt_ids, *args, **kwargs)

    monkeypatch.setattr(inference, "GenerationHandle", SpyHandle)
    resp = client.post("/generate", json={"prompt": "a <|endoftext|> b", "max_new_tokens": 2, "min_new_tokens": 0})
    assert resp.status_code == 200
    assert tok.eos_id not in seen["ids"]


def test_long_prompt_keeps_instruction_template():
    """Over-long prompts used to be cropped from the left by generate(),
    silently dropping the instruction header the SFT adapters rely on."""
    enc = GPT2Tokenizer()
    prompt_text, _ = build_prompt_with_budget("word " * 3000 + "FINAL QUESTION", 100, None, 256)
    assert prompt_text.startswith("Below is an instruction that describes a task.")
    assert prompt_text.endswith("FINAL QUESTION\n\n### Response:\n")
    assert len(enc.encode_ordinary(prompt_text)) + 100 <= 256


# ── Adapters ─────────────────────────────────────────────────────────


def test_incompatible_adapter_is_rejected_not_half_loaded(client):
    path = os.path.join("checkpoints", "adapters", "test_bad_shape.pt")
    wrong_dim = 32  # the placeholder engine's emb_dim is 64
    torch.save({
        "model_state_dict": {
            "trf_blocks.0.att.W_query.lora_A": torch.zeros(4, wrong_dim),
            "trf_blocks.0.att.W_query.lora_B": torch.zeros(wrong_dim, 4),
        },
        "is_lora": True, "lora_r": 4, "lora_alpha": 8.0,
    }, path)
    try:
        resp = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 2, "min_new_tokens": 0, "adapter": "test_bad_shape"})
        assert resp.status_code == 400
        assert app.state.engine.active_adapter is None
    finally:
        os.remove(path)


def test_adapter_name_cannot_traverse_paths(client):
    resp = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 2, "min_new_tokens": 0, "adapter": "../../etc/passwd"})
    assert resp.status_code == 422
