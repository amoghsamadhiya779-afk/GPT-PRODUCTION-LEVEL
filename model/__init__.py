# model/__init__.py
"""GPT-2 model package — built from scratch."""

from model.config import GPTConfig, TrainingConfig, load_config_from_yaml
from model.attention import MultiHeadAttention
from model.layers import LayerNorm, GELU, FeedForward, TransformerBlock
from model.gpt import GPTModel, generate, generate_text_simple, count_parameters
from model.tokenizer import GPT2Tokenizer
from model.lora import LoRALinear, inject_lora, get_lora_state_dict, mark_only_lora_as_trainable

__all__ = [
    "GPTConfig",
    "TrainingConfig",
    "load_config_from_yaml",
    "MultiHeadAttention",
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "GPTModel",
    "generate",
    "generate_text_simple",
    "count_parameters",
    "GPT2Tokenizer",
    "LoRALinear",
    "inject_lora",
    "get_lora_state_dict",
    "mark_only_lora_as_trainable",
]
