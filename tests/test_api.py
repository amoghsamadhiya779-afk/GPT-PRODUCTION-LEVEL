import os
import pytest
from fastapi.testclient import TestClient

# Set TESTING env var before importing app to disable slowapi and other side effects
os.environ["TESTING"] = "1"

# Monkeypatch the background weight loader to prevent it from doing any work during tests
import training.load_pretrained
def mock_main():
    pass
training.load_pretrained.main = mock_main

# Now it is safe to import the app
from app.api import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_cors_headers(client):
    response = client.options(
        "/generate",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers.get("access-control-allow-credentials") == "true"

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checkpoint" in data
    assert "parameters" in data
    assert "device" in data
    assert "uptime_seconds" in data
    assert "total_requests" in data
    assert "avg_tokens_per_second" in data

def test_web_search_budget(client, monkeypatch):
    import json
    
    # Mock web_search to return large snippets
    def mock_web_search(query, max_results=3):
        return [
            {"title": f"Title {i}", "snippet": "A very long snippet that will consume lots of tokens " * 50, "link": f"http://test{i}.com"}
            for i in range(3)
        ]
    
    monkeypatch.setattr("app.search.web_search", mock_web_search)
    
    # Force DummyEngine context size to prevent CI/local cached model differences
    app.state.engine.context_size = 256
    
    # Generate request with web search
    payload = {
        "prompt": "Test query",
        "max_new_tokens": 100,
        "web_search": True
    }
    
    # Standard generate
    resp = client.post("/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sources" in data
    assert len(data["sources"]) > 0
    
    # Check that sources were trimmed or dropped so it doesn't exceed 256
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    snippet_texts = [s["snippet"] for s in data["sources"]]
    total_snippet_tokens = sum(len(enc.encode(t)) for t in snippet_texts)
    
    # Budget leaves room for max_new_tokens(100) + fixed_tokens(~50) -> ~106 left out of 256
    assert total_snippet_tokens < 120 
    
    # Assert actual invariant on the final reconstructed prompt
    template_start = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\nTest query\n\nWeb Search Context:\n"
    )
    template_end = "\n\n### Response:\n"
    context_str = "\n".join([f"- {s}" for s in snippet_texts])
    final_prompt = template_start + context_str + template_end
    
    assert len(enc.encode(final_prompt)) + 100 <= 256
    
    # Generate stream
    resp_stream = client.post("/generate/stream", json=payload)
    assert resp_stream.status_code == 200
    lines = resp_stream.text.strip().split("\n")
    
    # First SSE data should contain sources
    sources_found = False
    for line in lines:
        if line.startswith("data: ") and "[DONE]" not in line:
            chunk = json.loads(line[6:])
            if "sources" in chunk:
                sources_found = True
                assert len(chunk["sources"]) > 0
                break
    assert sources_found, "Sources were not emitted in stream"

def test_generate_prompt_bounds(client):
    long_prompt = "A" * 4001
    response = client.post(
        "/generate",
        json={
            "prompt": long_prompt,
            "max_new_tokens": 10
        }
    )
    assert response.status_code == 422
    data = response.json()
    assert any("String should have at most 4000 characters" in item["msg"] for item in data["detail"])

    # Test missing prompt
    response2 = client.post(
        "/generate",
        json={
            "max_new_tokens": 10
        }
    )
    assert response2.status_code == 422

def test_generate_stream_equivalence(client):
    import json
    import torch
    from model.gpt import GPTModel
    from app.inference import GPTInferenceEngine
    
    # 1. First test that DummyEngine returns 503
    # Our mocked fixture might already be using DummyEngine
    # Just in case, let's explicitly remove generate_stream to simulate DummyEngine
    original_engine = app.state.engine
    
    class MockDummyEngine:
        pass
        
    app.state.engine = MockDummyEngine()
    resp = client.post("/generate/stream", json={"prompt": "test", "max_new_tokens": 5})
    assert resp.status_code == 503
    
    # 2. Setup a temporary valid engine to test streaming equivalence
    dummy_cfg = {
        "vocab_size": 50257,
        "context_length": 64,
        "emb_dim": 32,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(dummy_cfg)
    checkpoint = {
        "model_config": dummy_cfg,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "test_engine.pt")
    
    # Swap engine
    app.state.engine = GPTInferenceEngine("test_engine.pt", device="cpu")
    app.state.status = "active"
    
    try:
        req_payload = {
            "prompt": "Hello",
            "max_new_tokens": 5,
            "temperature": 0.0, # greedy for deterministic output
        }
        
        # Non-streamed
        resp1 = client.post("/generate", json=req_payload)
        assert resp1.status_code == 200
        text_full = resp1.json()["generated_text"]
        
        ans1 = text_full.split("Hello\n\n")[-1]
            
        # Streamed
        resp2 = client.post("/generate/stream", json=req_payload)
        assert resp2.status_code == 200
        
        # Parse SSE
        streamed_tokens = []
        for line in resp2.iter_lines():
            if line:
                # iter_lines returns str in TestClient but could be bytes in httpx
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("data: "):
                    data_str = line_str[len("data: "):]
                    data = json.loads(data_str)
                    if "token" in data:
                        streamed_tokens.append(data["token"])
        
        ans2 = "".join(streamed_tokens)
        
        # Compare equivalence
        assert ans1.strip() == ans2.strip()
        
    finally:
        app.state.engine = original_engine
        if os.path.exists("test_engine.pt"):
            try:
                os.remove("test_engine.pt")
            except:
                pass

def test_finetune_and_adapters(client):
    import time
    
    # 1. Start a finetuning job with 2 steps
    payload = {
        "examples": [{"instruction": "Test instruction", "response": "Test response"}],
        "adapter_name": "test_adapter_1",
        "steps": 2,
        "lr": 1e-3
    }
    resp = client.post("/finetune", json=payload)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    
    # Check 409 conflict
    resp_conflict = client.post("/finetune", json=payload)
    assert resp_conflict.status_code == 409
    
    # Poll until done or failed
    max_retries = 30
    for _ in range(max_retries):
        status_resp = client.get(f"/finetune/{job_id}")
        assert status_resp.status_code == 200
        state = status_resp.json()
        if state["status"] in ["done", "failed"]:
            break
        time.sleep(0.5)
        
    assert state["status"] == "done", f"Job failed: {state}"
    
    # 2. Check adapter registry
    resp_adapters = client.get("/adapters")
    assert resp_adapters.status_code == 200
    assert "test_adapter_1" in resp_adapters.json()["adapters"]
    
    # 3. Activate adapter
    resp_activate = client.post("/adapters/test_adapter_1/activate")
    assert resp_activate.status_code == 200
    assert resp_activate.json()["status"] == "success"
    
    # 4. Deactivate adapter
    resp_deactivate = client.post("/adapters/deactivate")
    assert resp_deactivate.status_code == 200
    assert resp_deactivate.json()["status"] == "success"
    
def test_feedback_persistence(client):
    import os
    
    # Submit feedback
    payload = {
        "prompt": "What is AI?",
        "response": "AI is artificial intelligence.",
        "rating": "down",
        "correction": "AI is cool."
    }
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    
    # Get feedback
    resp_get = client.get("/feedback")
    assert resp_get.status_code == 200
    data = resp_get.json()["feedback"]
    assert len(data) > 0
    
    found = False
    for item in data:
        if item["prompt"] == "What is AI?" and item["correction"] == "AI is cool.":
            found = True
            break
    assert found

