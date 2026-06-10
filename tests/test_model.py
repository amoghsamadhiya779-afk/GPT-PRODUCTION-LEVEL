# tests/test_model.py
"""Unit tests for the GPT model architecture.

Usage:
    py -m pytest tests/test_model.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model.config import GPTConfig
from model.gpt import GPTModel, generate_text_simple, count_parameters
from model.attention import MultiHeadAttention
from model.layers import LayerNorm, GELU, FeedForward, TransformerBlock


# Use a tiny config for fast tests
TINY_CONFIG = {
    "vocab_size": 100,
    "context_length": 32,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


class TestGPTConfig:
    def test_default_config(self):
        cfg = GPTConfig()
        assert cfg.vocab_size == 50257
        assert cfg.emb_dim == 768
        assert cfg.n_heads == 12
        assert cfg.n_layers == 12

    def test_to_dict(self):
        cfg = GPTConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "emb_dim" in d
        assert "drop_rate" in d  # mapped from dropout


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(
            d_in=64, d_out=64, context_length=32,
            dropout=0.0, num_heads=4,
        )
        x = torch.randn(2, 10, 64)
        out = mha(x)
        assert out.shape == (2, 10, 64)

    def test_causal_mask_exists(self):
        mha = MultiHeadAttention(
            d_in=64, d_out=64, context_length=16,
            dropout=0.0, num_heads=4,
        )
        assert hasattr(mha, "mask")
        assert mha.mask.shape == (16, 16)


class TestLayers:
    def test_layer_norm(self):
        ln = LayerNorm(64)
        x = torch.randn(2, 10, 64)
        out = ln(x)
        assert out.shape == x.shape

    def test_gelu(self):
        gelu = GELU()
        x = torch.randn(2, 10, 64)
        out = gelu(x)
        assert out.shape == x.shape

    def test_feedforward(self):
        ff = FeedForward(TINY_CONFIG)
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_transformer_block(self):
        block = TransformerBlock(TINY_CONFIG)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == x.shape


class TestGPTModel:
    def test_forward_shape(self):
        model = GPTModel(TINY_CONFIG)
        idx = torch.randint(0, 100, (2, 16))
        logits = model(idx)
        assert logits.shape == (2, 16, 100)

    def test_generate_simple(self):
        model = GPTModel(TINY_CONFIG)
        model.eval()
        idx = torch.randint(0, 100, (1, 5))
        out = generate_text_simple(model, idx, max_new_tokens=10, context_size=32)
        assert out.shape == (1, 15)  # 5 original + 10 generated

    def test_count_parameters(self):
        model = GPTModel(TINY_CONFIG)
        n = count_parameters(model)
        assert n > 0
        assert isinstance(n, int)
