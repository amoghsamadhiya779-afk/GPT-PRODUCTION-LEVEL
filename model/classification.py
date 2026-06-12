# model/classification.py
"""GPT sequence classification model wrapper.

Wraps the custom GPTModel with a linear classification head, using the last
sequence token's hidden state for logit generation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from model.gpt import GPTModel


class GPTClassificationModel(nn.Module):
    """Sequence classification model wrapping a scratch-built GPTModel."""

    def __init__(self, base_model: GPTModel, num_classes: int = 3) -> None:
        super().__init__()
        self.base_model = base_model
        
        # Pull emb_dim from base model config dict
        emb_dim = base_model.cfg["emb_dim"]
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, in_idx: Tensor) -> Tensor:
        """Forward pass for sequence classification.

        Args:
            in_idx: Input token IDs, shape (batch_size, seq_len).

        Returns:
            Class logits, shape (batch_size, num_classes).
        """
        batch_size, seq_len = in_idx.shape
        
        # Extract hidden states using base model components
        tok_embeds = self.base_model.tok_emb(in_idx)
        pos_embeds = self.base_model.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.base_model.drop_emb(x)
        
        # Loop through blocks manually since they return (x, present) for cache support
        for block in self.base_model.trf_blocks:
            x, _ = block(x, use_cache=False)
            
        x = self.base_model.final_norm(x)
        
        # Sequence classification: extract representation of the last token
        last_token_x = x[:, -1, :]  # Shape: (batch_size, emb_dim)
        
        # Map to class logits
        logits = self.classifier(last_token_x)  # Shape: (batch_size, num_classes)
        return logits
