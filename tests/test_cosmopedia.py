# tests/test_cosmopedia.py
"""Unit tests for Cosmopedia dataset preparation, loading, and math pre-training pipelines."""

import os
import shutil
import pytest
import torch

from model.config import GPTConfig, TrainingConfig
from data.dataset import create_dataloader
from training.train import train


@pytest.fixture
def mock_text_data():
    """Fixture to generate a small math textbook text corpus for testing."""
    txt_path = "tests/mock_cosmopedia.txt"
    mock_content = (
        "Let x be a real number. We define the derivative of f(x) = x^2 as 2x.\n"
        "Therefore, f'(3) = 6.<|endoftext|>\n"
        "Khan Academy lesson: Solving systems of linear equations.\n"
        "Express x in terms of y, then substitute in the second equation.<|endoftext|>\n"
    ) * 10
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(mock_content)
    yield txt_path
    
    if os.path.exists(txt_path):
        os.remove(txt_path)


def test_cosmopedia_dataloader(mock_text_data):
    """Verify that sliding window data loading handles textbook sequences correctly."""
    with open(mock_text_data, "r", encoding="utf-8") as f:
        text = f.read()

    loader = create_dataloader(
        txt=text,
        batch_size=2,
        max_length=16,
        stride=16,
        shuffle=False,
    )
    
    assert len(loader) > 0
    for inputs, targets in loader:
        assert inputs.shape == (2, 16)
        assert targets.shape == (2, 16)
        break


def test_cosmopedia_train_pipeline(mock_text_data):
    """Verify a complete causal pretraining epoch dry-run on textbook data."""
    cfg = GPTConfig(
        vocab_size=50257,
        context_length=16,
        emb_dim=64,
        n_heads=2,
        n_layers=1,
    )
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=2,
        save_dir="tests/checkpoints_cosmopedia_test",
        log_dir="tests/logs_cosmopedia_test",
        seed=123,
        warmup_steps=1,
        eval_freq=1,
    )
    
    # Run training loop
    train(
        model_cfg=cfg,
        train_cfg=train_cfg,
        data_path=mock_text_data,
        data_type="pretrain",
    )
    
    best_path = os.path.join(train_cfg.save_dir, "best_model.pt")
    assert os.path.exists(best_path)
    
    # Assert it copied the checkpoint to the central models/ directory
    models_path = os.path.join("models", "best_model.pt")
    assert os.path.exists(models_path)
    
    # Clean up test outputs
    for path in [train_cfg.save_dir, train_cfg.log_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
            
    if os.path.exists(models_path):
        os.remove(models_path)


def test_streamed_dataloader(monkeypatch):
    """Verify that streamed dataloader yields batches and handles on-the-fly streaming correctly."""
    from model.tokenizer import GPT2Tokenizer
    from data.dataset import create_streamed_dataloader
    import pandas as pd

    # Mock datasets.load_dataset to return a simple generator
    class MockDataset:
        def __iter__(self):
            yield {"text": "Theorem: Let x be a real number. We prove x = x."}
            yield {"text": "In this math lesson, we solve systems of equations."}

    def mock_load_dataset(path, name=None, split=None, streaming=False):
        return MockDataset()

    # Mock pandas read_csv
    def mock_read_csv(filepath_or_buffer, *args, **kwargs):
        return pd.DataFrame([
            {"act": "Tutor", "prompt": "Explain math."},
            {"act": "Writer", "prompt": "Write story."}
        ])

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    tokenizer = GPT2Tokenizer()
    loader = create_streamed_dataloader(
        tokenizer=tokenizer,
        batch_size=2,
        max_length=8,
        split="train",
    )

    # Fetch one batch
    for inputs, targets in loader:
        assert inputs.shape == (2, 8)
        assert targets.shape == (2, 8)
        break
