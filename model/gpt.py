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
        # Deliberately NOT tied to tok_emb.weight (as official GPT-2 does).
        # Tying now would save real params/RAM, but every shipped checkpoint
        # (base + sft_v1_small/medium LoRA bases) was trained with these as
        # independent tensors. Loading an old checkpoint into a newly-tied
        # model would silently collapse tok_emb.weight and out_head.weight
        # to whichever key state_dict happens to apply last -- no error, no
        # warning, just a quietly different (untested) model. Only tie this
        # if retraining from scratch with tying baked in from the start.
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


def _sample_next_token(
    logits: Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    no_repeat_ngram_size: int = 0,
    idx: Tensor | None = None,
    prompt_len: int = 0,
) -> Tensor:
    """Helper function to sample the next token from logits."""
    # N-gram penalty applies to the whole sequence to prevent repeating any n-gram
    if idx is not None:
        for b in range(logits.shape[0]):
            if no_repeat_ngram_size > 0 and idx.shape[1] >= no_repeat_ngram_size - 1:
                n = no_repeat_ngram_size
                seq = idx[b]
                if n == 1:
                    logits[b, seq] = float('-inf')
                else:
                    ctx = seq[-(n-1):]
                    for i in range(len(seq) - n + 1):
                        if torch.equal(seq[i:i+n-1], ctx):
                            banned_token = seq[i+n-1]
                            logits[b, banned_token] = float('-inf')
                            
            # Other penalties apply ONLY to generated tokens
            if prompt_len > 0 and idx.shape[1] > prompt_len:
                gen_tokens = idx[b, prompt_len:]
                
                if repetition_penalty > 1.0:
                    unique_tokens = torch.unique(gen_tokens)
                    token_logits = logits[b, unique_tokens]
                    penalized = torch.where(
                        token_logits >= 0,
                        token_logits / repetition_penalty,
                        token_logits * repetition_penalty,
                    )
                    logits[b, unique_tokens] = penalized
                    
                if frequency_penalty != 0.0 or presence_penalty != 0.0:
                    unique_tokens, counts = torch.unique(gen_tokens, return_counts=True)
                    if presence_penalty != 0.0:
                        logits[b, unique_tokens] -= presence_penalty
                    if frequency_penalty != 0.0:
                        logits[b, unique_tokens] -= frequency_penalty * counts.float().to(logits.device)

    # Top-k filtering
    if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(
            logits < min_val,
            torch.tensor(float("-inf")).to(logits.device),
            logits,
        )

    # Top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Shift the indices to the right to keep the first token that exceeds the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter mask back to original logits shape
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

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


from typing import Iterator

def generate_stream(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    no_repeat_ngram_size: int = 0,
    min_new_tokens: int = 0,
    eos_id: int | None = None,
    use_cache: bool = False,
) -> Iterator[int]:
    """Yields each generated token id as an integer."""
    prompt_len = idx.shape[1]
    tokens_generated = 0
    
    if use_cache:
        # Pre-fill step: process the initial context prompt to populate the cache
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits, past_key_values = model(idx_cond, use_cache=True, past_key_values=None)
        logits = logits[:, -1, :]
        if eos_id is not None and tokens_generated < min_new_tokens:
            logits[:, eos_id] = float('-inf')
        idx_next = _sample_next_token(
            logits, temperature, top_k, top_p, repetition_penalty, 
            frequency_penalty, presence_penalty, no_repeat_ngram_size, 
            idx, prompt_len
        )
        
        token_id = idx_next.item()
        tokens_generated += 1
        yield token_id
        
        if eos_id is not None and token_id == eos_id:
            return
            
        idx = torch.cat((idx, idx_next), dim=1)

        # Decode step: process only the newly generated token at each step
        for _ in range(max_new_tokens - 1):
            # If the cache would grow past context_size, we can't just slice
            # the oldest entries off the front: their K/V vectors have
            # absolute positional embeddings already baked in from when they
            # were computed, so simply cropping pins every future token to
            # the same final position slot (prev_tokens stays constant at
            # context_size - 1) instead of the window sliding forward. That
            # silently desyncs cached generation from the uncached path once
            # a generation crosses the context window.
            #
            # The model only has context_size rows of learned absolute
            # position embeddings, so tokens can't be extrapolated past that
            # -- the only correct fix is to match what the uncached path does
            # for a sliding window: recompute the cache from scratch over the
            # trailing context_size - 1 raw tokens (fresh positions
            # 0..context_size-2), then decode the new token at position
            # context_size - 1 as usual. This costs an O(context_size)
            # forward pass on every step once past the boundary (same order
            # as the uncached path), which is the honest price of staying
            # correct with fixed absolute position embeddings.
            total_len = past_key_values[0][0].shape[-2] + 1
            if total_len > context_size:
                window = idx[:, -context_size:-1]
                with torch.no_grad():
                    _, past_key_values = model(window, use_cache=True, past_key_values=None)

            with torch.no_grad():
                logits, past_key_values = model(idx_next, use_cache=True, past_key_values=past_key_values)
            logits = logits[:, -1, :]
            if eos_id is not None and tokens_generated < min_new_tokens:
                logits[:, eos_id] = float('-inf')
            idx_next = _sample_next_token(
                logits, temperature, top_k, top_p, repetition_penalty, 
                frequency_penalty, presence_penalty, no_repeat_ngram_size, 
                idx, prompt_len
            )

            token_id = idx_next.item()
            tokens_generated += 1
            yield token_id

            if eos_id is not None and token_id == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
            
    else:
        # Standard generation (no cache) - recalculates everything at each step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]
            if eos_id is not None and tokens_generated < min_new_tokens:
                logits[:, eos_id] = float('-inf')

            idx_next = _sample_next_token(
                logits, temperature, top_k, top_p, repetition_penalty, 
                frequency_penalty, presence_penalty, no_repeat_ngram_size, 
                idx, prompt_len
            )

            token_id = idx_next.item()
            tokens_generated += 1
            yield token_id

            if eos_id is not None and token_id == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

def generate(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    no_repeat_ngram_size: int = 0,
    min_new_tokens: int = 0,
    eos_id: int | None = None,
    use_cache: bool = False,
) -> Tensor:
    """Advanced text generation with temperature scaling, top-k/top-p sampling, and optional KV-Caching.
    """
    for token_id in generate_stream(
        model, idx, max_new_tokens, context_size, temperature, top_k, top_p, 
        repetition_penalty, frequency_penalty, presence_penalty, no_repeat_ngram_size,
        min_new_tokens, eos_id, use_cache
    ):
        idx_next = torch.tensor([[token_id]], dtype=torch.long, device=idx.device)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
