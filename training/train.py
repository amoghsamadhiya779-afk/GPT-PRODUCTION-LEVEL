# training/train.py
"""GPT-2 pre-training and instruction finetuning pipeline.

Features:
    - Support for standard pretraining and instruction-following datasets
    - Low-Rank Adaptation (LoRA) injection for parameter-efficient finetuning
    - Gradient accumulation steps for large batch simulation
    - Automatic Mixed Precision (AMP) FP16 training support
    - Cosine learning rate scheduling with linear warmup
    - AdamW optimizer with weight decay
    - Checkpoint loading and saving (saving only LoRA weights if enabled)
    - MLflow experiment tracking integration
"""

import argparse
import logging
import math
import os
import sys
import time
import json

import torch
import mlflow

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import GPTConfig, TrainingConfig, load_config_from_yaml
from model.gpt import GPTModel, count_parameters
from model.tokenizer import GPT2Tokenizer
from model.lora import inject_lora, get_lora_state_dict
from data.dataset import create_dataloader, create_instruction_dataloader, create_streamed_dataloader
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
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(
    model_cfg: GPTConfig,
    train_cfg: TrainingConfig,
    data_path: str,
    data_type: str = "pretrain",
    checkpoint_path: str | None = None,
    use_lora: bool = False,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    accum_steps: int = 1,
    use_amp: bool = False,
    steps_per_epoch: int | None = None,
) -> None:
    """Run the full GPT-2 pre-training or instruction finetuning pipeline."""

    # ─── Device ──────────────────────────────────────────────────
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ─── Data ────────────────────────────────────────────────────
    temp_train_path = "tests/temp_train.json"
    temp_val_path = "tests/temp_val.json"
    is_streaming = (data_path == "stream_hf" or data_type == "stream_hf")

    if is_streaming:
        logger.info("Initializing Hugging Face streamed datasets (on-the-fly Cosmopedia & prompts.chat)...")
        tokenizer = GPT2Tokenizer()
        train_loader = create_streamed_dataloader(
            tokenizer=tokenizer,
            batch_size=train_cfg.batch_size,
            max_length=model_cfg.context_length,
            split="train",
        )
        val_loader = create_streamed_dataloader(
            tokenizer=tokenizer,
            batch_size=train_cfg.batch_size,
            max_length=model_cfg.context_length,
            split="val",
        )
    elif data_type == "instruction":
        logger.info("Loading instruction dataset from: %s", data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        train_ratio = 0.90
        split_idx = int(train_ratio * len(data))

        os.makedirs("tests", exist_ok=True)
        with open(temp_train_path, "w", encoding="utf-8") as f:
            json.dump(data[:split_idx], f)
        with open(temp_val_path, "w", encoding="utf-8") as f:
            json.dump(data[split_idx:], f)

        train_loader = create_instruction_dataloader(
            temp_train_path,
            batch_size=train_cfg.batch_size,
            max_length=model_cfg.context_length,
            shuffle=True,
        )
        val_loader = create_instruction_dataloader(
            temp_val_path,
            batch_size=train_cfg.batch_size,
            max_length=model_cfg.context_length,
            shuffle=False,
        )
    else:
        logger.info("Loading raw text pretraining dataset from: %s", data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            text_data = f.read()
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

    try:
        train_len = len(train_loader)
        val_len = len(val_loader)
        logger.info("Train batches: %d | Val batches: %d", train_len, val_len)
    except TypeError:
        logger.info("Train batches: [STREAMING] | Val batches: [STREAMING]")

    # ─── Model ───────────────────────────────────────────────────
    cfg_dict = model_cfg.to_dict()
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info("Loading starting model weights from checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg_dict = checkpoint["model_config"]
        model_cfg = GPTConfig(**cfg_dict)
        model = GPTModel(cfg_dict)
        # Load weights with strict=False in case loading a LoRA-only checkpoint or mismatch
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        logger.info("Initializing model from scratch.")
        model = GPTModel(cfg_dict)

    # Inject LoRA weights if requested
    if use_lora:
        logger.info("Injecting LoRA adapters (rank=%d, alpha=%.1f) into attention projections...", lora_r, lora_alpha)
        inject_lora(model, r=lora_r, alpha=lora_alpha, target_modules=["W_query", "W_value"])

    model.to(device)
    n_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s (%.1fM) | Trainable: %s", 
                f"{n_params:,}", n_params / 1e6, f"{trainable_params:,}")

    # Set up MLflow tracking
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("GPT-2 Finetuning" if use_lora or data_type == "instruction" else "GPT-2 Pretraining")
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.start_run(run_name="gpt2-training")

    # Log config parameters
    mlflow.log_params({
        "vocab_size": model_cfg.vocab_size,
        "context_length": model_cfg.context_length,
        "emb_dim": model_cfg.emb_dim,
        "n_heads": model_cfg.n_heads,
        "n_layers": model_cfg.n_layers,
        "dropout": model_cfg.dropout,
        "qkv_bias": model_cfg.qkv_bias,
    })
    mlflow.log_params({
        "learning_rate": train_cfg.learning_rate,
        "num_epochs": train_cfg.num_epochs,
        "batch_size": train_cfg.batch_size,
        "weight_decay": train_cfg.weight_decay,
        "warmup_steps": train_cfg.warmup_steps,
        "max_grad_norm": train_cfg.max_grad_norm,
        "seed": train_cfg.seed,
        "data_type": data_type,
        "use_lora": use_lora,
        "lora_r": lora_r if use_lora else 0,
        "accum_steps": accum_steps,
        "use_amp": use_amp,
    })
    mlflow.log_param("parameter_count", n_params)
    mlflow.log_param("trainable_parameter_count", trainable_params)

    # ─── Optimizer ───────────────────────────────────────────────
    # Optimize only trainable parameters (critical when model is frozen for LoRA!)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # Initialize AMP GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    # ─── Training Loop ───────────────────────────────────────────
    tokenizer = GPT2Tokenizer()
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    best_val_loss = float("inf")

    if is_streaming:
        if steps_per_epoch is None:
            steps_per_epoch = 1000
        total_steps = (steps_per_epoch // accum_steps) * train_cfg.num_epochs
    else:
        total_steps = (len(train_loader) // accum_steps) * train_cfg.num_epochs
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  STARTING GPT-2 PIPELINE RUN")
    logger.info("=" * 60)
    logger.info("  Task Type    : %s", data_type.upper())
    logger.info("  Epochs       : %d", train_cfg.num_epochs)
    logger.info("  Batch size   : %d", train_cfg.batch_size)
    logger.info("  Learning rate: %.2e", train_cfg.learning_rate)
    logger.info("  Accumulation : %d steps", accum_steps)
    logger.info("  AMP Enabled  : %s", use_amp and device.type == "cuda")
    logger.info("  Total steps  : %d", total_steps)
    logger.info("=" * 60)
    logger.info("")

    start_time = time.time()

    for epoch in range(train_cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for step_in_epoch, (input_batch, target_batch) in enumerate(train_loader):
            if is_streaming and steps_per_epoch is not None and step_in_epoch >= steps_per_epoch * accum_steps:
                break
            # We scale step index relative to gradient accumulation
            if step_in_epoch % accum_steps == 0:
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

            # Autocast forward pass
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda" and use_amp)):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                if accum_steps > 1:
                    loss = loss / accum_steps

            # Backward pass with scaler
            scaler.scale(loss).backward()

            tokens_seen += input_batch.numel()
            epoch_loss += loss.item() * accum_steps if accum_steps > 1 else loss.item()
            epoch_steps += 1

            # Step optimizer after accumulating gradients
            if (step_in_epoch + 1) % accum_steps == 0 or (step_in_epoch + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

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

                    # Log metrics to MLflow
                    mlflow.log_metric("train_loss", train_loss, step=global_step)
                    mlflow.log_metric("val_loss", val_loss, step=global_step)
                    mlflow.log_metric("learning_rate", lr, step=global_step)

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint = {
                            "model_state_dict": get_lora_state_dict(model) if use_lora else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_loss": val_loss,
                            "model_config": model_cfg.to_dict(),
                            "is_lora": use_lora,
                            "lora_r": lora_r if use_lora else None,
                            "lora_alpha": lora_alpha if use_lora else None,
                        }
                        path = os.path.join(train_cfg.save_dir, "best_model.pt")
                        torch.save(checkpoint, path)
                        logger.info("  -> Saved best model (val_loss=%.4f)", val_loss)

                        # Copy to models directory
                        models_dir = "models"
                        os.makedirs(models_dir, exist_ok=True)
                        models_path = os.path.join(models_dir, "best_model.pt")
                        try:
                            import shutil
                            shutil.copy2(path, models_path)
                            logger.info("  -> Copied best model to %s", models_path)
                        except Exception as e:
                            logger.warning("Failed to copy best model to %s: %s", models_path, e)

        # End-of-epoch sample generation
        generate_and_print_sample(model, tokenizer, device, "Every effort moves you")

    # Clean up temporary split files if generated
    if data_type == "instruction":
        for path in [temp_train_path, temp_val_path]:
            if os.path.exists(path):
                os.remove(path)

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

    # Save final model state dict
    final_path = os.path.join(train_cfg.save_dir, "final_model.pt")
    torch.save({
        "model_state_dict": get_lora_state_dict(model) if use_lora else model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "is_lora": use_lora,
        "lora_r": lora_r if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
    }, final_path)
    logger.info("Final model saved to %s", final_path)

    # Copy final model to models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    models_final_path = os.path.join(models_dir, "final_model.pt")
    try:
        import shutil
        shutil.copy2(final_path, models_final_path)
        logger.info("Final model copied to %s", models_final_path)
    except Exception as e:
        logger.warning("Failed to copy final model to %s: %s", models_final_path, e)
    mlflow.log_artifact(final_path)

    # Log best checkpoint
    best_path = os.path.join(train_cfg.save_dir, "best_model.pt")
    if os.path.exists(best_path):
        mlflow.log_artifact(best_path)

    # Plot losses
    if train_losses:
        plot_path = os.path.join(train_cfg.log_dir, "loss.png")
        plot_losses(train_losses, val_losses, track_tokens_seen, train_cfg.num_epochs, plot_path)
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path)
    mlflow.end_run()


def main() -> None:
    # Ensure stdout/stderr UTF-8 compatibility on Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="Pre-train or finetune a GPT-2 model.",
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
        help="Path to training text or JSON file.",
    )
    parser.add_argument(
        "--data_type",
        choices=["pretrain", "instruction", "stream_hf"],
        default="pretrain",
        help="Type of training: pretrain (raw text), instruction (Alpaca JSON format), or stream_hf (Hugging Face streaming).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to starting checkpoint (.pt file) to resume or finetune.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Inject LoRA adapters for parameter-efficient finetuning.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=4,
        help="Rank for LoRA adapters.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=8.0,
        help="Scaling alpha for LoRA adapters.",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to simulate larger batch sizes.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable Automatic Mixed Precision (AMP) on CUDA device.",
    )
    parser.add_argument(
        "--epochs", "--num_epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Number of steps in a training epoch when streaming.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=None,
        help="Override evaluation frequency (every N steps).",
    )
    parser.add_argument(
        "--eval_iter",
        type=int,
        default=None,
        help="Override number of batches to evaluate.",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        model_cfg, train_cfg = load_config_from_yaml(args.config)
    else:
        model_cfg = GPTConfig()
        train_cfg = TrainingConfig()

    # Override config values if command line arguments are provided
    if args.epochs is not None:
        train_cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.eval_freq is not None:
        train_cfg.eval_freq = args.eval_freq
    if args.eval_iter is not None:
        train_cfg.eval_iter = args.eval_iter

    # Load data
    if args.data != "stream_hf" and args.data_type != "stream_hf" and not os.path.exists(args.data):
        logger.error("Training data not found: %s", args.data)
        sys.exit(1)

    train(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_path=args.data,
        data_type=args.data_type,
        checkpoint_path=args.checkpoint,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        accum_steps=args.accum_steps,
        use_amp=args.use_amp,
        steps_per_epoch=args.steps_per_epoch,
    )


if __name__ == "__main__":
    main()
