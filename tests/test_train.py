# tests/test_train.py
"""Integration tests for the training pipeline.

Usage:
    py -m pytest tests/test_train.py -v
"""

import os
import sys
import json
import shutil
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import GPTConfig, TrainingConfig
from training.train import train


@pytest.fixture
def dummy_data_paths():
    # Create temp directory in local tests workspace
    temp_dir = "tests/tmp_train_run"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create dummy raw text file for pretraining (make it large enough to fit multiple validation windows)
    pretrain_path = os.path.join(temp_dir, "dummy_pretrain.txt")
    with open(pretrain_path, "w", encoding="utf-8") as f:
        f.write("Once upon a time, there was a small model. It wanted to learn how to generate human-like text. " * 200)

    # Create dummy instructions JSON file
    instruction_path = os.path.join(temp_dir, "dummy_instruction.json")
    dummy_instructions = [
        {"instruction": "Capitalize word.", "input": "apple", "output": "APPLE"},
        {"instruction": "Capitalize word.", "input": "banana", "output": "BANANA"},
        {"instruction": "Capitalize word.", "input": "cherry", "output": "CHERRY"},
        {"instruction": "Capitalize word.", "input": "date", "output": "DATE"},
        {"instruction": "Capitalize word.", "input": "fig", "output": "FIG"},
        {"instruction": "Capitalize word.", "input": "grape", "output": "GRAPE"},
        {"instruction": "Capitalize word.", "input": "honeydew", "output": "HONEYDEW"},
        {"instruction": "Capitalize word.", "input": "kiwi", "output": "KIWI"},
        {"instruction": "Capitalize word.", "input": "lemon", "output": "LEMON"},
        {"instruction": "Capitalize word.", "input": "mango", "output": "MANGO"},
    ]
    with open(instruction_path, "w", encoding="utf-8") as f:
        json.dump(dummy_instructions, f)

    yield pretrain_path, instruction_path, temp_dir

    # Cleanup temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pretrain_pipeline_dry_run(dummy_data_paths):
    pretrain_path, _, temp_dir = dummy_data_paths
    
    # Configure tiny model and training config
    model_cfg = GPTConfig(
        vocab_size=50257,
        context_length=256,
        emb_dim=64,
        n_heads=2,
        n_layers=1,
    )
    train_cfg = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        eval_freq=2,
        eval_iter=1,
        save_dir=os.path.join(temp_dir, "checkpoints_pretrain"),
        log_dir=os.path.join(temp_dir, "logs_pretrain"),
        warmup_steps=1,
    )

    try:
        # Run pretraining dry-run
        train(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            data_path=pretrain_path,
            data_type="pretrain",
            accum_steps=1,
            use_lora=False,
        )
    finally:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.end_run()

    # Verify best_model and final_model checkpoints were created
    assert os.path.exists(os.path.join(train_cfg.save_dir, "best_model.pt"))
    assert os.path.exists(os.path.join(train_cfg.save_dir, "final_model.pt"))


def test_instruction_lora_pipeline_dry_run(dummy_data_paths):
    _, instruction_path, temp_dir = dummy_data_paths
    
    # Configure tiny model and training config
    model_cfg = GPTConfig(
        vocab_size=50257,
        context_length=256,
        emb_dim=64,
        n_heads=2,
        n_layers=1,
    )
    train_cfg = TrainingConfig(
        num_epochs=1,
        batch_size=1,
        eval_freq=2,
        eval_iter=1,
        save_dir=os.path.join(temp_dir, "checkpoints_lora"),
        log_dir=os.path.join(temp_dir, "logs_lora"),
        warmup_steps=1,
    )

    try:
        # Run instruction finetuning with LoRA dry-run
        train(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            data_path=instruction_path,
            data_type="instruction",
            accum_steps=2,  # test gradient accumulation
            use_lora=True,
            lora_r=2,
            lora_alpha=4.0,
        )
    finally:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.end_run()

    # Verify checkpoint creation
    best_path = os.path.join(train_cfg.save_dir, "best_model.pt")
    final_path = os.path.join(train_cfg.save_dir, "final_model.pt")
    assert os.path.exists(best_path)
    assert os.path.exists(final_path)

    # Load checkpoint and assert is_lora metadata is saved
    checkpoint = torch.load(final_path, map_location="cpu", weights_only=True)
    assert checkpoint.get("is_lora") is True
    assert checkpoint.get("lora_r") == 2
    assert checkpoint.get("lora_alpha") == 4.0
    
    # Check that state dict only contains LoRA weights
    for name in checkpoint["model_state_dict"].keys():
        assert "lora_" in name
