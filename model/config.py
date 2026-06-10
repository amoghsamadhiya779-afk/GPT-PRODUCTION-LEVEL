from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
    qkv_bias: bool = False


    