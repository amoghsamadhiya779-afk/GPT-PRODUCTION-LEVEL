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

    def forward(
        self,
        in_idx: Tensor,
        use_cache: bool = False,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Forward pass through the GPT model.

        Args:
            in_idx: Input token indices, shape (batch_size, seq_len).
            use_cache: Whether to return the updated key/value cache.
            past_key_values: List of (past_key, past_value) tuples for each block.

        Returns:
            If use_cache is False: Logits tensor of shape (batch_size, seq_len, vocab_size).
            If use_cache is True: Tuple of (logits, list of updated past_key_values).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # Calculate dynamic absolute position indices for positional embeddings
        if past_key_values is not None:
            prev_tokens = past_key_values[0][0].shape[-2]
            pos_idx = torch.arange(prev_tokens, prev_tokens + seq_len, device=in_idx.device)
        else:
            pos_idx = torch.arange(seq_len, device=in_idx.device)

        pos_embeds = self.pos_emb(pos_idx)
        x = tok_embeds + pos_embeds  # (batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)

        new_past_key_values = []
        for i, block in enumerate(self.trf_blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, layer_past=past, use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(present)

        x = self.final_norm(x)
        logits = self.out_head(x)

        if use_cache:
            return logits, new_past_key_values
        return logits


def _sample_next_token(logits: Tensor, temperature: float, top_k: int | None) -> Tensor:
    """Helper function to sample the next token from logits."""
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
    return idx_next


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
    use_cache: bool = False,
) -> Tensor:
    """Advanced text generation with temperature scaling, top-k sampling, and optional KV-Caching.

    Args:
        model: Trained GPTModel instance.
        idx: Starting token indices, shape (batch, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        context_size: Maximum context window size.
        temperature: Sampling temperature (0.0 = greedy, higher = more random).
        top_k: If set, only sample from the top-k most probable tokens.
        eos_id: End-of-sequence token ID (generation stops if encountered).
        use_cache: Whether to use key-value caching to speed up generation.

    Returns:
        Generated token sequence, shape (batch, seq_len + generated_tokens).
    """
    if use_cache:
        # Pre-fill step: process the initial context prompt to populate the cache
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits, past_key_values = model(idx_cond, use_cache=True, past_key_values=None)
        logits = logits[:, -1, :]
        idx_next = _sample_next_token(logits, temperature, top_k)
        
        # Append first generated token
        if eos_id is not None and idx_next.item() == eos_id:
            return torch.cat((idx, idx_next), dim=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # Decode step: process only the newly generated token at each step
        for _ in range(max_new_tokens - 1):
            # Crop cache along the sequence dimension if it exceeds the maximum context length
            total_len = past_key_values[0][0].shape[-2] + 1
            if total_len > context_size:
                past_key_values = [
                    (k[:, :, -(context_size - 1):, :], v[:, :, -(context_size - 1):, :])
                    for k, v in past_key_values
                ]

            with torch.no_grad():
                logits, past_key_values = model(idx_next, use_cache=True, past_key_values=past_key_values)
            logits = logits[:, -1, :]
            idx_next = _sample_next_token(logits, temperature, top_k)

            if eos_id is not None and idx_next.item() == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
            
    else:
        # Standard generation (no cache) - recalculates everything at each step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            idx_next = _sample_next_token(logits, temperature, top_k)

            if eos_id is not None and idx_next.item() == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

    return idx


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
