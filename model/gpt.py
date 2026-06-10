# model/gpt.py
"""GPT-2 Language Model — built entirely from scratch.

Implements the full GPT-2 architecture with token + positional embeddings,
stacked transformer blocks, and a language model head for next-token prediction.

Also includes text generation utilities with greedy decoding, temperature
scaling, and top-k sampling.
"""

import torch
import torch.nn as nn
from torch import Tensor

from model.layers import LayerNorm, TransformerBlock
from model.config import GPTConfig


class GPTModel(nn.Module):
    """GPT-2 Language Model.

    Architecture:
        Token Embedding + Positional Embedding
        -> Embedding Dropout
        -> N x TransformerBlock (Pre-LayerNorm + Multi-Head Attention + FFN)
        -> Final LayerNorm
        -> Linear Output Head (projects to vocab_size)

    Args:
        cfg: GPTConfig dataclass or dict with model hyperparameters.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        # Support both GPTConfig dataclass and plain dict
        if isinstance(cfg, GPTConfig):
            cfg = cfg.to_dict()

        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: Tensor) -> Tensor:
        """Forward pass through the GPT model.

        Args:
            in_idx: Input token indices, shape (batch_size, seq_len).

        Returns:
            Logits over vocabulary, shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # (batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
) -> Tensor:
    """Generate text using greedy decoding (argmax).

    Args:
        model: Trained GPTModel instance.
        idx: Starting token indices, shape (batch, seq_len).
        max_new_tokens: Number of new tokens to generate.
        context_size: Maximum context window size.

    Returns:
        Extended token sequence, shape (batch, seq_len + max_new_tokens).
    """
    for _ in range(max_new_tokens):
        # Crop context to supported size
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on last time step: (batch, vocab_size)
        logits = logits[:, -1, :]

        # Greedy: pick the token with highest logits
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append to running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq_len + 1)

    return idx


def generate(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> Tensor:
    """Advanced text generation with temperature scaling and top-k sampling.

    Args:
        model: Trained GPTModel instance.
        idx: Starting token indices, shape (batch, seq_len).
        max_new_tokens: Maximum number of tokens to generate.
        context_size: Maximum context window size.
        temperature: Sampling temperature (0.0 = greedy, higher = more random).
        top_k: If set, only sample from the top-k most probable tokens.
        eos_id: End-of-sequence token ID (generation stops if encountered).

    Returns:
        Generated token sequence, shape (batch, seq_len + generated_tokens).
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        # Temperature scaling + sampling
        if temperature > 0.0:
            logits = logits / temperature
            # Numerical stability: subtract row-wise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Early stopping on EOS
        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
