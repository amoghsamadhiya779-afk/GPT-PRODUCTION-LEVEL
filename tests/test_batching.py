# tests/test_batching.py
"""Continuous batching (app/batching.py): correctness, isolation, backpressure."""

import threading
import time

import pytest
import torch

from app.adapters import Adapter, expected_lora_shapes
from app.batching import ContinuousBatcher, EngineBusy
from app.inference import GPTInferenceEngine
from model.gpt import generate

CFG = {"vocab_size": 50257, "context_length": 64, "emb_dim": 32, "n_heads": 2,
       "n_layers": 2, "drop_rate": 0.0, "qkv_bias": True, "model_size": "tiny"}
# The API's default penalties, greedy so outputs are deterministic.
SAMPLING = dict(temperature=0.0, top_k=50, top_p=0.9, repetition_penalty=1.15, frequency_penalty=0.1,
                presence_penalty=0.1, no_repeat_ngram_size=3, min_new_tokens=3, max_new_tokens=20)
PROMPTS = ["Hello there", "What is the capital of France?", "Tell me a long story about a robot who",
           "a", "Numbers: 1 2 3 4 5 6 7 8 9 10 11 12"]


def make_engine(cfg=CFG, **kwargs) -> GPTInferenceEngine:
    torch.manual_seed(0)
    engine = GPTInferenceEngine.from_config(cfg, **kwargs)
    for p in engine.model.parameters():  # larger weights -> varied, non-degenerate outputs
        p.data.normal_(0, 0.5)
    return engine


def reference(engine, prompt: str, **sampling) -> list[int]:
    """Single-request generation with model.gpt.generate (the unbatched path)."""
    ids = torch.tensor([engine.tokenizer.encode_ordinary(prompt)])
    s = {**SAMPLING, **sampling}
    out = generate(engine.model, ids, s["max_new_tokens"], engine.context_size, s["temperature"], s["top_k"],
                   s["top_p"], s["repetition_penalty"], s["frequency_penalty"], s["presence_penalty"],
                   s["no_repeat_ngram_size"], s["min_new_tokens"], engine.eos_id, s.get("use_cache", True))
    tokens = out[0, ids.shape[1]:].tolist()
    return tokens[: tokens.index(engine.eos_id)] if engine.eos_id in tokens else tokens


def run_concurrently(engine, requests):
    """Submit all requests while the scheduler is held, so they share batches."""
    with engine.model_lock:
        handles = [engine.submit(prompt, **{**SAMPLING, **extra}) for prompt, extra in requests]
    return [list(h.tokens()) for h in handles]


def test_batched_output_matches_single_request_generation():
    engine = make_engine()
    requests = [(p, {}) for p in PROMPTS]
    outputs = run_concurrently(engine, requests)
    assert engine.batcher.peak_batch_size >= 4, "requests were not actually batched together"
    for (prompt, _), got in zip(requests, outputs):
        assert got == reference(engine, prompt), prompt
    assert len({tuple(o) for o in outputs}) > 1  # the check isn't trivially passing


def test_batching_matches_across_the_context_window():
    """Sequences crossing context_length slide their window, like generate()."""
    small = {**CFG, "context_length": 16}
    engine = make_engine(small)
    requests = [("one two three four five six seven", {"max_new_tokens": 30, "min_new_tokens": 30}),
                ("short", {"max_new_tokens": 25, "min_new_tokens": 25})]
    outputs = run_concurrently(engine, requests)
    for (prompt, extra), got in zip(requests, outputs):
        assert len(got) == extra["max_new_tokens"]
        assert got == reference(engine, prompt, **extra)


def test_uncached_requests_share_the_batch_correctly():
    engine = make_engine()
    requests = [(PROMPTS[1], {"use_cache": False}), (PROMPTS[2], {}), (PROMPTS[0], {"use_cache": False})]
    outputs = run_concurrently(engine, requests)
    for (prompt, extra), got in zip(requests, outputs):
        assert got == reference(engine, prompt, **extra)


def _random_adapter(engine, name: str, seed: int) -> Adapter:
    gen = torch.Generator().manual_seed(seed)
    shapes = expected_lora_shapes(engine.model_config, 4)
    state = {k: torch.randn(*shape, generator=gen) * 0.5 for k, shape in shapes.items()}
    return Adapter(name=name, r=4, alpha=8.0, state_dict=state)


def test_mixed_adapters_are_isolated():
    """Requests for different adapters never run in the same batch: every
    output equals running that request alone with its own adapter."""
    engine = make_engine()
    a1, a2 = _random_adapter(engine, "a1", 1), _random_adapter(engine, "a2", 2)
    plan = [(PROMPTS[1], a1), (PROMPTS[1], None), (PROMPTS[1], a2), (PROMPTS[2], a1), (PROMPTS[1], a1)]
    with engine.model_lock:
        handles = [engine.submit(p, adapter=a, **SAMPLING) for p, a in plan]
    batched = [list(h.tokens()) for h in handles]

    for (prompt, adapter), got in zip(plan, batched):
        engine.set_adapter(adapter)
        assert got == reference(engine, prompt), (prompt, adapter and adapter.name)
    engine.set_adapter(None)
    # Same prompt, different adapter -> different text (adapters really applied).
    assert len({tuple(batched[0]), tuple(batched[1]), tuple(batched[2])}) == 3


def test_cancel_frees_the_slot_promptly():
    engine = make_engine()
    handle = engine.submit(PROMPTS[0], **{**SAMPLING, "max_new_tokens": 60, "min_new_tokens": 60})
    tokens = handle.tokens()
    next(tokens)
    handle.cancel()
    assert list(tokens) is not None  # drains to the end without hanging
    assert handle.finish_reason == "cancelled" and len(handle.generated) < 60
    deadline = time.time() + 5
    while engine.batcher.stats()["active"] and time.time() < deadline:
        time.sleep(0.01)
    assert engine.batcher.stats()["active"] == 0


def test_full_queue_rejects_fast_and_admission_times_out():
    engine = make_engine()
    engine._batcher = ContinuousBatcher(engine, max_batch_size=1, max_waiting=2)
    long = {**SAMPLING, "max_new_tokens": 40, "min_new_tokens": 40}
    with engine.model_lock:  # the scheduler can admit but not compute
        first = engine.submit(PROMPTS[0], **long)
        assert first.wait_admitted(2.0)                   # took the only slot
        second = engine.submit(PROMPTS[1], **long)        # waiting 1/2
        with pytest.raises(EngineBusy):                   # waiting 2/2, never admitted in time
            engine.generate(PROMPTS[3], admit_timeout=0.2, **long)
        third = engine.submit(PROMPTS[2], **long)         # the timed-out request freed its spot
        start = time.perf_counter()
        with pytest.raises(EngineBusy):                   # queue full -> immediate refusal
            engine.submit(PROMPTS[4], **long)
        assert time.perf_counter() - start < 0.1
    assert [len(list(h.tokens())) for h in (first, second, third)] == [40, 40, 40]
    assert engine.batcher.stats()["waiting"] == 0


def test_a_failing_step_fails_only_its_requests():
    engine = make_engine()
    real_forward = engine.model.forward
    calls = {"n": 0}

    def flaky_forward(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated CUDA error")
        return real_forward(*args, **kwargs)

    engine.model.forward = flaky_forward
    with pytest.raises(RuntimeError):
        engine.generate(PROMPTS[0], **SAMPLING)
    engine.model.forward = real_forward
    assert engine.generate(PROMPTS[0], **SAMPLING)["completion_text"] == engine.tokenizer.decode(reference(engine, PROMPTS[0]))


def test_concurrent_callers_from_many_threads():
    engine = make_engine()
    results = {}

    def worker(i):
        results[i] = engine.generate(PROMPTS[i % len(PROMPTS)], **SAMPLING)["completion_text"]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(results) == 10
    for i, text in results.items():
        assert text == engine.tokenizer.decode(reference(engine, PROMPTS[i % len(PROMPTS)]))


def test_half_precision_engine_runs_end_to_end():
    """The GPU path uses bf16/fp16 weights and KV cache; bf16 also runs on CPU."""
    engine = make_engine(dtype="bfloat16")
    assert next(engine.model.parameters()).dtype == torch.bfloat16
    out = engine.generate(PROMPTS[1], **SAMPLING)
    assert out["tokens_generated"] > 0
    assert engine.batcher.cache.k.dtype == torch.bfloat16
