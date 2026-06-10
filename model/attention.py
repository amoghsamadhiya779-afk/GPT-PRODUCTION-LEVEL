# model/attention.py
"""Multi-Head Causal Self-Attention mechanism for GPT.

Implements scaled dot-product attention with causal masking,
built entirely from scratch using PyTorch primitives.
"""

import torch
import torch.nn as nn
from torch import Tensor


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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            Output tensor of shape (batch_size, num_tokens, d_out).
        """
        b, num_tokens, d_in = x.shape

        # Project input to queries, keys, values — shape: (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to split into heads: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to: (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention: Q @ K^T / sqrt(head_dim)
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask — prevent attending to future tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Concatenate heads: (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
