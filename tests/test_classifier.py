# tests/test_classifier.py
"""Unit tests for sequence classification datasets, model architectures, and training pipelines."""

import os
import csv
import torch
import pytest
import shutil

from model.config import GPTConfig, TrainingConfig
from model.gpt import GPTModel
from model.tokenizer import GPT2Tokenizer
from model.classification import GPTClassificationModel
from data.classification_dataset import create_classification_dataloader
from training.train_classifier import train_classifier


@pytest.fixture
def mock_csv_data():
    """Fixture to generate a temporary CSV dataset matching Chatbot Arena schema."""
    csv_path = "tests/mock_train.csv"
    fieldnames = [
        "id", "model_a", "model_b", "prompt", "response_a", "response_b",
        "winner_model_a", "winner_model_b", "winner_tie"
    ]
    rows = [
        {
            "id": "1", "model_a": "m1", "model_b": "m2", "prompt": "Hello",
            "response_a": "Hi there", "response_b": "Hello human",
            "winner_model_a": "1", "winner_model_b": "0", "winner_tie": "0"
        },
        {
            "id": "2", "model_a": "m1", "model_b": "m2", "prompt": "Math",
            "response_a": "2+2=4", "response_b": "2+2=5",
            "winner_model_a": "1", "winner_model_b": "0", "winner_tie": "0"
        },
        {
            "id": "3", "model_a": "m1", "model_b": "m2", "prompt": "Weather",
            "response_a": "Rainy", "response_b": "Sunny",
            "winner_model_a": "0", "winner_model_b": "1", "winner_tie": "0"
        },
        {
            "id": "4", "model_a": "m1", "model_b": "m2", "prompt": "Tie",
            "response_a": "Same", "response_b": "Same",
            "winner_model_a": "0", "winner_model_b": "0", "winner_tie": "1"
        },
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    yield csv_path
    
    if os.path.exists(csv_path):
        os.remove(csv_path)


def test_classification_dataloader(mock_csv_data):
    """Verify that the classification dataloader handles parsing, mapping, and shape sizing correctly."""
    tokenizer = GPT2Tokenizer()
    loader = create_classification_dataloader(
        mock_csv_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=64,
        shuffle=False,
    )
    
    assert len(loader) == 2
    
    batches = list(loader)
    
    # Batch 1
    inputs, labels = batches[0]
    assert inputs.shape == (2, 64)
    assert labels.shape == (2,)
    assert torch.equal(labels, torch.tensor([0, 0]))  # both rows are winner_model_a

    # Batch 2
    inputs_2, labels_2 = batches[1]
    assert inputs_2.shape == (2, 64)
    assert labels_2.shape == (2,)
    assert torch.equal(labels_2, torch.tensor([1, 2]))  # row 3: model_b, row 4: tie


def test_classification_model():
    """Verify that GPTClassificationModel generates logit predictions of correct shape."""
    cfg = GPTConfig(
        vocab_size=1000,
        context_length=64,
        emb_dim=64,
        n_heads=2,
        n_layers=1,
        dropout=0.0
    )
    base = GPTModel(cfg)
    model = GPTClassificationModel(base, num_classes=3)
    
    x = torch.randint(0, 1000, (2, 64))
    logits = model(x)
    
    assert logits.shape == (2, 3)


def test_classifier_train_pipeline(mock_csv_data):
    """Verify a complete sequence classification training cycle smoke test."""
    cfg = GPTConfig(
        vocab_size=50257,
        context_length=64,
        emb_dim=64,
        n_heads=2,
        n_layers=1,
    )
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=2,
        save_dir="tests/checkpoints_classifier_test",
        log_dir="tests/logs_classifier_test",
        seed=123,
    )
    
    # Run the train pipeline
    train_classifier(
        model_cfg=cfg,
        train_cfg=train_cfg,
        data_path=mock_csv_data,
        use_lora=True,
        lora_r=2,
        lora_alpha=4.0,
    )
    
    checkpoint_file = os.path.join(train_cfg.save_dir, "classifier_best.pt")
    assert os.path.exists(checkpoint_file)
    
    # Verify checkpoint contents
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert checkpoint["is_lora"] is True
    assert checkpoint["is_hf"] is False
    
    # Clean up checkpoints and logs created during testing
    for path in [train_cfg.save_dir, train_cfg.log_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
