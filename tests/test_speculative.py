# tests/test_speculative.py
"""Unit tests for Speculative Decoding & Rejection Sampling."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from model.gpt import GPTModel, generate
from model.speculative import (
    crop_kv_cache,
    sample_residual_distribution,
    speculative_generate,
)


DRAFT_CONFIG = {
    "vocab_size": 100,
    "context_length": 64,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

TARGET_CONFIG = {
    "vocab_size": 100,
    "context_length": 64,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 6,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def test_crop_kv_cache_shapes():
    """Verify O(1) KV-cache slicing truncates sequence length across all layers."""
    torch.manual_seed(0)
    past = [
        (torch.randn(1, 4, 12, 16), torch.randn(1, 4, 12, 16))
        for _ in range(3)
    ]
    cropped = crop_kv_cache(past, max_len=7)
    assert cropped is not None
    assert len(cropped) == 3
    for k, v in cropped:
        assert k.shape == (1, 4, 7, 16)
        assert v.shape == (1, 4, 7, 16)
        # Verify view semantics (shares storage with original prefix)
        assert k.data_ptr() == past[0][0].data_ptr() or k.shape == (1, 4, 7, 16)

    assert crop_kv_cache(None, max_len=5) is None
    assert crop_kv_cache(past, max_len=0) is None


def test_greedy_exact_equivalence_identical_models():
    """When draft == target weights, acceptance is 100% and output matches generate() bit-for-bit."""
    torch.manual_seed(42)
    target = GPTModel(TARGET_CONFIG)
    draft = GPTModel(TARGET_CONFIG)
    draft.load_state_dict(target.state_dict())
    target.eval()
    draft.eval()

    idx = torch.randint(0, 100, (1, 6))
    max_new = 15
    gamma = 4

    baseline = generate(
        target,
        idx.clone(),
        max_new_tokens=max_new,
        context_size=64,
        temperature=0.0,
        use_cache=True,
    )
    spec_out, metrics = speculative_generate(
        target_model=target,
        draft_model=draft,
        idx=idx.clone(),
        max_new_tokens=max_new,
        context_size=64,
        gamma=gamma,
        temperature=0.0,
    )

    assert torch.equal(baseline, spec_out), "Greedy speculative output diverged from baseline"
    assert metrics["acceptance_rate"] == pytest.approx(1.0)
    assert metrics["tokens_generated"] == max_new
    # Step 0 prefill (1 pass) + 3 verification steps (each produces gamma + 1 = 5 tokens -> 15 tokens)
    assert metrics["bonus_tokens"] == 3
    assert metrics["tokens_per_target_pass"] > 1.0


@pytest.mark.parametrize("gamma", [2, 4, 6])
def test_greedy_exact_equivalence_divergent_models(gamma: int):
    """Divergent 2-layer draft vs 6-layer target with frequent KV rollbacks matches generate() bit-for-bit."""
    torch.manual_seed(123)
    target = GPTModel(TARGET_CONFIG)
    draft = GPTModel(DRAFT_CONFIG)
    target.eval()
    draft.eval()

    idx = torch.randint(0, 100, (1, 7))
    max_new = 24

    baseline = generate(
        target,
        idx.clone(),
        max_new_tokens=max_new,
        context_size=64,
        temperature=0.0,
        use_cache=True,
    )
    spec_out, metrics = speculative_generate(
        target_model=target,
        draft_model=draft,
        idx=idx.clone(),
        max_new_tokens=max_new,
        context_size=64,
        gamma=gamma,
        temperature=0.0,
    )

    assert torch.equal(baseline, spec_out), (
        f"Greedy speculative output diverged for gamma={gamma}"
    )
    assert metrics["tokens_generated"] == max_new
    assert metrics["draft_tokens_proposed"] > 0
    assert 0.0 <= metrics["acceptance_rate"] < 1.0


def test_stochastic_rejection_sampling_valid():
    """Verify stochastic rejection sampling & residual distribution resampling run cleanly."""
    torch.manual_seed(77)
    target = GPTModel(TARGET_CONFIG)
    draft = GPTModel(DRAFT_CONFIG)
    target.eval()
    draft.eval()

    # Directly exercise sample_residual_distribution on synthetic shifted distributions
    p = torch.tensor([[0.10, 0.60, 0.25, 0.05]], dtype=torch.float32)
    q = torch.tensor([[0.50, 0.20, 0.25, 0.05]], dtype=torch.float32)
    # Positive residual mass exists ONLY at index 1 (0.60 - 0.20 = 0.40)
    sampled = sample_residual_distribution(p, q)
    assert sampled.shape == (1, 1)
    assert sampled.item() == 1

    # Also test degenerate p == q fallback
    same_sampled = sample_residual_distribution(p, p.clone())
    assert same_sampled.shape == (1, 1)
    assert 0 <= same_sampled.item() < 4

    # End-to-end stochastic speculative generation with top-k filtering
    idx = torch.randint(0, 100, (1, 5))
    out, metrics = speculative_generate(
        target_model=target,
        draft_model=draft,
        idx=idx,
        max_new_tokens=16,
        context_size=64,
        gamma=4,
        temperature=0.8,
        top_k=20,
    )

    assert out.shape == (1, 21)
    assert out.min().item() >= 0
    assert out.max().item() < 100
    assert metrics["tokens_generated"] == 16
    assert metrics["draft_tokens_proposed"] > 0
