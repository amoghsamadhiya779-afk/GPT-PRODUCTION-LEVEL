# model/tokenizer.py
"""GPT-2 Tokenizer wrapper using tiktoken.

Provides a clean interface for encoding text to token IDs
and decoding token IDs back to text, using OpenAI's tiktoken
library with the GPT-2 BPE vocabulary (50,257 tokens).
"""

import tiktoken
import torch
from torch import Tensor


class GPT2Tokenizer:
    """GPT-2 BPE Tokenizer.

    Wraps tiktoken's GPT-2 encoding with convenience methods
    for encoding/decoding text and converting between text and tensors.

    Attributes:
        encoding: The tiktoken encoding instance.
        vocab_size: Size of the vocabulary (50,257).
    """

    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoding.n_vocab  # 50257

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Encode text to a list of token IDs."""
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self.encoding.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to text."""
        return self.encoding.decode(token_ids)

    def text_to_token_ids(self, text: str) -> Tensor:
        """Convert text to a batched tensor of token IDs.
        
        Returns:
            Tensor of shape (1, num_tokens).
        """
        encoded = self.encode(text)
        return torch.tensor(encoded).unsqueeze(0)

    def token_ids_to_text(self, token_ids: Tensor) -> str:
        """Convert a batched tensor of token IDs back to text.
        
        Args:
            token_ids: Tensor of shape (1, num_tokens) or (num_tokens,).
        """
        flat = token_ids.squeeze(0)
        return self.decode(flat.tolist())