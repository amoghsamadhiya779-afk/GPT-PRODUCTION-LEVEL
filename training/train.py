# training/train.py
"""GPT-2 pre-training pipeline with production-level features.

Features:
    - AdamW optimizer with configurable weight decay
    - Cosine learning rate scheduling with linear warmup
    - Gradient clipping (max_norm)
    - Model checkpointing (saves best model by validation loss)
    - Full training state saving for resume capability
    - Structured logging
    - Automatic device selection (CUDA -> CPU)

Usage:
    py training/train.py
    py training/train.py --config configs/gpt2_small.yaml
"""

import argparse
import logging
import math
import os
import sys
import time

import torch

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import GPTConfig, TrainingConfig, load_config_from_yaml
from model.gpt import GPTModel, count_parameters
from model.tokenizer import GPT2Tokenizer
from data.dataset import create_dataloader
from training.utils import (
    evaluate_model,
    generate_and_print_sample,
    plot_losses,
    calc_loss_batch,
)

# ─── Logging Setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 1e-6) -> float:
    """Cosine learning rate schedule with linear warmup.

    Args:
        step: Current training step.
        warmup_steps: Number of warmup steps.
        max_steps: Total number of training steps.
        max_lr: Peak learning rate after warmup.
        min_lr: Minimum learning rate.

    Returns:
        Learning rate for the current step.
    """
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(
    model_cfg: GPTConfig,
    train_cfg: TrainingConfig,
    text_data: str,
) -> None:
    """Run the full GPT-2 pre-training pipeline."""

    # ─── Device ──────────────────────────────────────────────────
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ─── Data ────────────────────────────────────────────────────
    cfg_dict = model_cfg.to_dict()
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader(
        text_data[:split_idx],
        batch_size=train_cfg.batch_size,
        max_length=model_cfg.context_length,
        stride=model_cfg.context_length,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        text_data[split_idx:],
        batch_size=train_cfg.batch_size,
        max_length=model_cfg.context_length,
        stride=model_cfg.context_length,
        shuffle=False,
        drop_last=False,
    )

    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ─── Model ───────────────────────────────────────────────────
    model = GPTModel(cfg_dict)
    model.to(device)
    n_params = count_parameters(model)
    logger.info("Model parameters: %s (%.1fM)", f"{n_params:,}", n_params / 1e6)

    # ─── Optimizer ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # ─── Training Loop ───────────────────────────────────────────
    tokenizer = GPT2Tokenizer()
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    best_val_loss = float("inf")

    total_steps = len(train_loader) * train_cfg.num_epochs
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  STARTING GPT-2 PRE-TRAINING")
    logger.info("=" * 60)
    logger.info("  Epochs       : %d", train_cfg.num_epochs)
    logger.info("  Batch size   : %d", train_cfg.batch_size)
    logger.info("  Learning rate: %.2e", train_cfg.learning_rate)
    logger.info("  Total steps  : %d", total_steps)
    logger.info("=" * 60)
    logger.info("")

    start_time = time.time()

    for epoch in range(train_cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for input_batch, target_batch in train_loader:
            global_step += 1

            # Update learning rate (cosine schedule with warmup)
            lr = get_lr(
                step=global_step,
                warmup_steps=train_cfg.warmup_steps,
                max_steps=total_steps,
                max_lr=train_cfg.learning_rate,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward + backward
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

            optimizer.step()

            tokens_seen += input_batch.numel()
            epoch_loss += loss.item()
            epoch_steps += 1

            # Periodic evaluation
            if global_step % train_cfg.eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, train_cfg.eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(
                    "Ep %d/%d (Step %06d) | Train Loss: %.4f | Val Loss: %.4f | LR: %.2e",
                    epoch + 1, train_cfg.num_epochs, global_step,
                    train_loss, val_loss, lr,
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": val_loss,
                        "model_config": model_cfg.to_dict(),
                    }
                    path = os.path.join(train_cfg.save_dir, "best_model.pt")
                    torch.save(checkpoint, path)
                    logger.info("  -> Saved best model (val_loss=%.4f)", val_loss)

        # End-of-epoch sample generation
        generate_and_print_sample(model, tokenizer, device, "Every effort moves you")

    # ─── Post-Training ───────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Total time     : %.1f seconds", elapsed)
    logger.info("  Best val loss  : %.4f", best_val_loss)
    logger.info("  Tokens seen    : %s", f"{tokens_seen:,}")
    logger.info("=" * 60)

    # Save final model
    final_path = os.path.join(train_cfg.save_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg.to_dict(),
    }, final_path)
    logger.info("Final model saved to %s", final_path)

    # Plot losses
    if train_losses:
        plot_path = os.path.join(train_cfg.log_dir, "loss.png")
        plot_losses(train_losses, val_losses, track_tokens_seen, train_cfg.num_epochs, plot_path)


def main() -> None:
    # Ensure stdout and stderr use UTF-8 encoding to avoid Windows encoding crashes
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="Pre-train a GPT-2 model from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--data",
        default=os.path.join("data", "the-verdict.txt"),
        help="Path to training text file.",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        model_cfg, train_cfg = load_config_from_yaml(args.config)
    else:
        model_cfg = GPTConfig()
        train_cfg = TrainingConfig()

    # Load data
    if not os.path.exists(args.data):
        logger.error("Training data not found: %s", args.data)
        logger.error("Run `py data/download.py` first to download training data.")
        sys.exit(1)

    with open(args.data, "r", encoding="utf-8") as f:
        text_data = f.read()
    logger.info("Loaded training data: %d characters", len(text_data))

    train(model_cfg, train_cfg, text_data)


if __name__ == "__main__":
    main()
