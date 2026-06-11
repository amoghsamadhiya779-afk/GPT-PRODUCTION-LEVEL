# tests/test_dataset.py
"""Unit tests for datasets and data loaders.

Usage:
    py -m pytest tests/test_dataset.py -v
"""

import os
import sys
import json
import torch
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import GPTDataset, InstructionDataset, create_dataloader, create_instruction_dataloader
from model.tokenizer import GPT2Tokenizer


class TestGPTDataset:
    def test_gpt_dataset(self):
        tokenizer = GPT2Tokenizer()
        txt = "Once upon a time there was a tiny model that trained on a simple text. " * 5
        dataset = GPTDataset(txt, tokenizer, max_length=16, stride=8)
        assert len(dataset) > 0
        x, y = dataset[0]
        assert x.shape == (16,)
        assert y.shape == (16,)
        # y should be shifted by 1 relative to x
        assert torch.equal(x[1:], y[:-1])


class TestInstructionDataset:
    @pytest.fixture
    def mock_json_file(self):
        data = [
            {
                "instruction": "Convert the value to uppercase.",
                "input": "hello world",
                "output": "HELLO WORLD"
            },
            {
                "instruction": "Greet the user.",
                "input": "",
                "output": "Hello user, hope you are doing well!"
            }
        ]
        # Save in local workspace
        file_path = "tests/mock_dataset.json"
        os.makedirs("tests", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        yield file_path
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

    def test_instruction_dataset_masking(self, mock_json_file):
        tokenizer = GPT2Tokenizer()
        max_length = 64
        dataset = InstructionDataset(mock_json_file, tokenizer, max_length=max_length)
        
        assert len(dataset) == 2
        x, y = dataset[0]
        
        # Verify shape
        assert x.shape == (max_length - 1,)
        assert y.shape == (max_length - 1,)
        
        # Verify that y has some -100 values (prompt tokens) and some non -100 values (response tokens)
        prompt_mask_indices = (y == -100)
        assert torch.any(prompt_mask_indices)
        
        # Check first token of target is masked
        assert y[0] == -100
        
        # The unmasked labels should match the shifted input tokens
        # For indices i where y[i] != -100, y[i] should be x[i+1] (which is the target next token)
        for i in range(len(y)):
            if y[i] != -100:
                assert y[i] == x[i + 1]

    def test_instruction_dataloader(self, mock_json_file):
        dataloader = create_instruction_dataloader(
            mock_json_file, batch_size=2, max_length=32
        )
        assert len(dataloader) == 1
        for x, y in dataloader:
            assert x.shape == (2, 31)
            assert y.shape == (2, 31)
