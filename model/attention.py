# model/attention.py
"""Multi-Head Causal Self-Attention mechanism for GPT.

Implements scaled dot-product attention with causal masking,
built entirely from scratch using PyTorch primitives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.kv_cache import SlotBatch  # noqa: F401  (type annotations)
from model.lora import LoRARows, MultiLoRALinear


def _project(layer: nn.Module, x: Tensor, lora: LoRARows | None) -> Tensor:
    """Apply a projection, passing per-row adapter selection if it takes one."""
    if lora is not None and isinstance(layer, MultiLoRALinear):
        return layer(x, lora)
    return layer(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Causal Self-Attention.

    Splits the embedding dimension across multiple attention heads,
    computes scaled dot-product attention with a causal mask to prevent
    attending to future tokens, and projects the concatenated heads back
    to the original embedding dimension.

    Args:
        d_in:  Input embedding dimension.
        d_out: Output embedding dimension (must be divisible by num_heads).
        context_length: Maximum sequence length (used to pre-compute the causal mask).
        dropout: Dropout probability applied to attention weights.
        num_heads: Number of parallel attention heads.
        qkv_bias: Whether to include bias terms in the Q/K/V projections.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Separate linear projections for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Output projection to combine heads
        self.dropout = nn.Dropout(dropout)

        # Pre-compute upper-triangular causal mask (registered as a buffer, not a parameter)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(
        self,
        x: Tensor,
        layer_past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        slot: tuple["SlotBatch", int] | None = None,
        lora: LoRARows | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass through multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, d_in).
            layer_past: Optional tuple of (past_keys, past_values) from previous steps.
            use_cache: Whether to return key/value states for caching.
            slot: (SlotBatch, layer index) for batched serving with a shared
                slot cache (model/kv_cache.py); replaces layer_past.
            lora: per-row LoRA adapter selection for MultiLoRALinear projections.

        Returns:
            Tuple of (output tensor, updated layer_past cache).
        """
        b, num_tokens, d_in = x.shape

        # Project input to queries, keys, values — shape: (b, num_tokens, d_out)
        keys = _project(self.W_key, x, lora)
        queries = _project(self.W_query, x, lora)
        values = _project(self.W_value, x, lora)

        # Reshape to split into heads: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to: (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if slot is not None:
            # Batched serving: rows are different sequences at different
            # positions, each attending to its own slot of the shared cache.
            slot_batch, layer = slot
            keys, values = slot_batch.update(layer, keys, values)
            context_vec = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=slot_batch.attn_mask, dropout_p=0.0,
            )
            context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
            return self.out_proj(context_vec), None

        # Concatenate with past key/value states if present
        if layer_past is not None:
            past_keys, past_values = layer_past
            keys = torch.cat((past_keys, keys), dim=-2)
            values = torch.cat((past_values, values), dim=-2)

        # Current layer's key/value cache
        present = (keys, values)

        # Scaled dot-product attention with a causal mask:
        #     softmax(Q @ K^T / sqrt(head_dim) + causal_mask) @ V
        # computed by PyTorch's fused kernel, which never materializes the full
        # (num_tokens x total_tokens) score matrix and is several times faster
        # than the explicit matmul/softmax chain on both CPU and GPU.
        # tests/test_model.py checks it against the explicit formula.
        total_tokens = keys.shape[-2]
        if num_tokens == 1:
            # A single new query may attend to every cached position.
            attn_mask, is_causal = None, False
        elif total_tokens == num_tokens:
            attn_mask, is_causal = None, True
        else:
            # Multi-token query on top of a cache: query i sits at absolute
            # position prev_tokens + i and may attend to keys 0..prev_tokens + i.
            prev_tokens = total_tokens - num_tokens
            attn_mask = ~self.mask[prev_tokens:total_tokens, :total_tokens].bool()
            is_causal = False

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # Concatenate heads: (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, present
