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
        self.eos_id = self.encoding.eot_token  # <|endoftext|> = 50256

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Encode text to a list of token IDs.

        A literal "<|endoftext|>" in `text` becomes the real end-of-text
        control token, which is what training data formatting wants. Never use
        this for untrusted input -- use `encode_ordinary` instead.
        """
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self.encoding.encode(text, allowed_special=allowed_special)

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text with special-token strings treated as plain text.

        Use this for anything that isn't fully trusted (user prompts, web
        snippets): otherwise a user typing "<|endoftext|>" injects a genuine
        document-boundary token into the model's context.
        """
        return self.encoding.encode_ordinary(text)

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