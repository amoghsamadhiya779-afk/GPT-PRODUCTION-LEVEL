# tests/test_citations.py
"""Inline citations for grounded answers (app/citations.py) and the API/data plumbing."""

import json

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.citations import cite
from data.add_citations import parse_grounded_instruction

SOURCES = [
    {"title": "Mars", "snippet": "Mars is the fourth planet from the Sun. It is a dusty, cold, desert world.", "link": "https://mars.test"},
    {"title": "Moons", "snippet": "Mars has two small moons, Phobos and Deimos.", "link": "https://moons.test"},
]


def test_supported_sentences_get_the_right_source():
    out = cite("Tell me about Mars", "Mars is the fourth planet from the Sun. It has two moons, Phobos and Deimos.", SOURCES)
    assert out.text == "Mars is the fourth planet from the Sun [1]. It has two moons, Phobos and Deimos [2]."
    assert [c["source"] for c in out.citations] == [1, 2]
    assert out.grounded_fraction == 1.0 and out.safety_net_prefix == ""


def test_unsupported_claims_are_not_cited():
    out = cite("Tell me about Mars", "Mars is the fourth planet from the Sun. It is made entirely of cheese.", SOURCES)
    assert out.text == "Mars is the fourth planet from the Sun [1]. It is made entirely of cheese."
    assert out.grounded_fraction == 0.5


def test_model_written_markers_are_validated():
    out = cite("Mars moons?", "Mars has two moons, Phobos and Deimos [2]. They orbit closely [9].", SOURCES)
    assert "[2]" in out.text and "[9]" not in out.text  # a citation to a nonexistent source is dropped


def test_formatting_survives_annotation():
    out = cite("Mars?", "Facts:\n1. Mars is the fourth planet from the Sun.\n2. Mars has two small moons.\nYes.", SOURCES)
    assert out.text.count("\n") == 3
    assert out.text.endswith("Yes.")  # too short to attribute, left alone


def test_ungrounded_answers_lead_with_a_cited_quote():
    out = cite("Tell me about Mars", "I love eating chocolate cake and ice cream.", SOURCES)
    assert out.safety_net_prefix == "From the sources: Mars is the fourth planet from the Sun. [1]\n\n"
    assert out.full_text.endswith("ice cream.")


def test_no_sources_leaves_the_answer_untouched():
    assert cite("q", "Anything at all [1].", []).text == "Anything at all [1]."


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.search.web_search", lambda q, max_results=3: SOURCES)
    with TestClient(app) as c:
        app.state.status = "active"
        yield c


def _fake_answer(engine, monkeypatch, answer):
    def fake_generate(prompt, **kwargs):
        return {"completion_text": answer, "generated_text": prompt + answer, "tokens_generated": 5,
                "time_taken_seconds": 0.1, "tokens_per_second": 50.0}
    monkeypatch.setattr(engine, "generate", fake_generate)

    tokens = answer.split(" ")

    def fake_stream(handle):
        for i, word in enumerate(tokens):
            yield word + (" " if i < len(tokens) - 1 else ""), 0.1, i + 1
    monkeypatch.setattr(engine, "stream_text", fake_stream)


def test_generate_returns_cited_answer(client, monkeypatch):
    _fake_answer(app.state.engine, monkeypatch, "Mars has two moons, Phobos and Deimos.")
    body = client.post("/generate", json={"prompt": "What moons does Mars have?", "max_new_tokens": 5,
                                          "min_new_tokens": 0, "web_search": True}).json()
    assert body["generated_text"].endswith("Mars has two moons, Phobos and Deimos [2].")
    assert body["citations"] == [{"sentence": 0, "source": 2, "support": 1.0}]


def test_stream_done_event_carries_cited_final_text(client, monkeypatch):
    _fake_answer(app.state.engine, monkeypatch, "Mars has two moons, Phobos and Deimos.")
    resp = client.post("/generate/stream", json={"prompt": "What moons does Mars have?", "max_new_tokens": 5,
                                                 "min_new_tokens": 0, "web_search": True})
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    done = next(e for e in events if e.get("done"))
    assert done["final_text"] == "Mars has two moons, Phobos and Deimos [2]."
    assert done["citations"][0]["source"] == 2


def test_sft_grounded_instruction_parsing():
    parsed = parse_grounded_instruction("[1] Alpha fact.\n[2] Beta fact\nspanning lines.\n\nQuestion: What is beta?")
    assert parsed == ([{"snippet": "Alpha fact."}, {"snippet": "Beta fact\nspanning lines."}], "What is beta?")
    assert parse_grounded_instruction("What is beta?") is None
    assert parse_grounded_instruction("[2] Out of order.\n\nQuestion: q") is None
