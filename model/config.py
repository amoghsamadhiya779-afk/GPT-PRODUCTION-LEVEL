# model/config.py
"""Configuration dataclasses for GPT model and training."""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class GPTConfig:
    """Configuration for the GPT model architecture.
    
    Default values correspond to GPT-2 Small (124M parameters).
    """
    vocab_size: int = 50257        # GPT-2 BPE vocabulary size
    context_length: int = 256      # Max sequence length (reduced for training demo)
    emb_dim: int = 768             # Embedding dimension
    n_heads: int = 12              # Number of attention heads
    n_layers: int = 12             # Number of transformer blocks
    dropout: float = 0.1           # Dropout rate
    qkv_bias: bool = False         # Bias in Q/K/V projections

    def to_dict(self) -> dict:
        """Convert config to dictionary for compatibility with dict-based APIs."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "drop_rate": self.dropout,
            "qkv_bias": self.qkv_bias,
        }


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    learning_rate: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 2
    weight_decay: float = 0.1
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    eval_freq: int = 5             # Evaluate every N steps
    eval_iter: int = 1             # Number of batches for evaluation
    save_dir: str = "checkpoints"  # Directory for saving model checkpoints
    log_dir: str = "logs"          # Directory for training logs
    seed: int = 123                # Random seed for reproducibility


def load_config_from_yaml(path: str) -> tuple[GPTConfig, TrainingConfig]:
    """Load model and training configs from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    model_cfg = GPTConfig(**raw.get("model", {}))
    train_cfg = TrainingConfig(**raw.get("training", {}))
    return model_cfg, train_cfg