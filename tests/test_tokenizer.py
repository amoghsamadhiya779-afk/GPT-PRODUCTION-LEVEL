# tests/test_tokenizer.py
"""Unit tests for the GPT-2 tokenizer.

Usage:
    py -m pytest tests/test_tokenizer.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model.tokenizer import GPT2Tokenizer


class TestGPT2Tokenizer:
    def setup_method(self):
        self.tokenizer = GPT2Tokenizer()

    def test_encode_returns_list(self):
        ids = self.tokenizer.encode("Hello, world!")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_returns_string(self):
        ids = self.tokenizer.encode("Test")
        text = self.tokenizer.decode(ids)
        assert isinstance(text, str)

    def test_roundtrip(self):
        text = "Transformers are amazing."
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        assert decoded == text

    def test_text_to_token_ids_shape(self):
        tensor = self.tokenizer.text_to_token_ids("Hello")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 2
        assert tensor.shape[0] == 1  # batch dimension

    def test_token_ids_to_text(self):
        text = "The quick brown fox"
        tensor = self.tokenizer.text_to_token_ids(text)
        decoded = self.tokenizer.token_ids_to_text(tensor)
        assert decoded == text

    def test_vocab_size(self):
        assert self.tokenizer.vocab_size == 50257

    def test_special_token(self):
        text = "Hello<|endoftext|>World"
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        assert "<|endoftext|>" in decoded
