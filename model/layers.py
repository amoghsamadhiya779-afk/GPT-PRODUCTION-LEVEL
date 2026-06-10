# model/layers.py
"""Core transformer building blocks for GPT.

Contains LayerNorm, GELU activation, FeedForward network,
and the TransformerBlock — all implemented from scratch.
"""

import torch
import torch.nn as nn
from torch import Tensor

from model.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """Layer Normalization with learnable scale and shift.

    Normalizes across the last dimension (embedding dim) with
    learnable affine parameters, matching the GPT-2 implementation.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation (tanh approximation).

    Hand-implemented using the exact formula from the GPT-2 paper:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Two linear transformations with a GELU activation in between.
    Expands the embedding dimension by 4x and projects back down,
    following the original transformer design.

    Architecture:
        Linear(emb_dim -> 4 * emb_dim) -> GELU -> Linear(4 * emb_dim -> emb_dim)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Single Transformer Block with Pre-LayerNorm architecture.

    Uses Pre-Norm (LayerNorm applied before attention/FFN) with
    residual connections and dropout on the shortcut path.
    This matches the GPT-2 architecture.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> + x (residual)
          -> LayerNorm -> FeedForward -> Dropout -> + x (residual)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(
        self,
        x: Tensor,
        layer_past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, present = self.att(x, layer_past=layer_past, use_cache=use_cache)  # (batch_size, num_tokens, emb_dim)
        x = self.drop_shortcut(x)
        x = x + shortcut           # Residual connection

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut           # Residual connection

        return x, present
