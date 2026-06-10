# training/utils.py
"""Training utility functions.

Includes loss computation, model evaluation, sample text generation,
and training loss visualization.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def calc_loss_batch(
    input_batch: Tensor,
    target_batch: Tensor,
    model,
    device: torch.device,
) -> Tensor:
    """Compute cross-entropy loss for a single batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Compute average cross-entropy loss across a DataLoader."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """Evaluate model on training and validation sets."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model,
    tokenizer,
    device: torch.device,
    start_context: str,
) -> None:
    """Generate and print a text sample during training."""
    from model.gpt import generate_text_simple

    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = tokenizer.text_to_token_ids(start_context).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
    decoded = tokenizer.token_ids_to_text(token_ids)
    logger.info("Sample: %s", decoded.replace("\n", " "))
    model.train()


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    tokens_seen: list[int],
    num_epochs: int,
    save_path: str = "loss.png",
) -> None:
    """Plot and save training/validation loss curves."""
    epochs_seen = torch.linspace(0, num_epochs, len(train_losses)).tolist()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Style
    fig.patch.set_facecolor("#1a1a2e")
    ax1.set_facecolor("#16213e")

    ax1.plot(epochs_seen, train_losses, color="#00d2ff", linewidth=2, label="Training Loss")
    ax1.plot(epochs_seen, val_losses, color="#ff6b6b", linewidth=2, linestyle="-.", label="Validation Loss")
    ax1.set_xlabel("Epochs", color="white", fontsize=12)
    ax1.set_ylabel("Loss", color="white", fontsize=12)
    ax1.tick_params(colors="white")
    ax1.legend(loc="upper right", facecolor="#16213e", edgecolor="white", labelcolor="white")
    ax1.set_title("GPT-2 Training Progress", color="white", fontsize=14, fontweight="bold")

    # Second x-axis for tokens
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Seen", color="white", fontsize=12)
    ax2.tick_params(colors="white")

    ax1.grid(True, alpha=0.2, color="white")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Loss plot saved to %s", save_path)
