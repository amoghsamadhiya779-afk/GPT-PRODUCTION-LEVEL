# model/kv_cache.py
"""Slot-based key/value cache for batched (continuous-batching) inference.

The per-request cache in model/gpt.py (`past_key_values`) grows by
concatenation and holds one sequence. For serving many requests at once we
instead preallocate storage for `max_slots` sequences: each request owns one
slot, and its token at absolute position p is stored at index p of that slot.
Because positions and cache indices coincide, sequences of different lengths
can share a single forward pass -- every row writes its new keys/values at its
own positions and attends to its own slot up to its own position.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SlotBatch:
    """Where a forward pass reads and writes the cache.

    slots:     (B,)   cache slot of each row
    positions: (B, T) absolute position of each input token
    attn_mask: (B, 1, T, L) True where a query may attend to a cached key
    """
    cache: "SlotKVCache"
    slots: Tensor
    positions: Tensor
    attn_mask: Tensor

    def update(self, layer: int, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        return self.cache.update(layer, self.slots, self.positions, keys, values, self.attn_mask.shape[-1])


class SlotKVCache:
    """Key/value storage for up to `max_slots` concurrent sequences.

    Storage grows by doubling up to `max_len` rather than reserving the full
    context window up front: most chats are far shorter than 1024 tokens, and
    a full-size cache for GPT-2 medium with 8 slots would take 1.6 GB.
    """

    def __init__(self, n_layers: int, max_slots: int, n_heads: int, head_dim: int, max_len: int,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                 initial_len: int = 128) -> None:
        self.max_slots = max_slots
        self.max_len = max_len
        self.capacity = min(initial_len, max_len)
        shape = (n_layers, max_slots, n_heads, self.capacity, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def ensure_capacity(self, length: int) -> None:
        if length <= self.capacity:
            return
        if length > self.max_len:
            raise ValueError(f"Cache length {length} exceeds the context length {self.max_len}.")
        new_capacity = min(max(length, 2 * self.capacity), self.max_len)
        pad = (0, 0, 0, new_capacity - self.capacity)  # grow the position dimension
        self.k = torch.nn.functional.pad(self.k, pad)
        self.v = torch.nn.functional.pad(self.v, pad)
        self.capacity = new_capacity

    def batch(self, slots: Tensor, positions: Tensor) -> SlotBatch:
        """Prepare a forward pass writing tokens at `positions` into `slots`."""
        length = int(positions.max()) + 1
        self.ensure_capacity(length)
        key_positions = torch.arange(length, device=positions.device)
        # Causal within each row's own history. Entries beyond a row's position
        # (a shorter sequence, or stale data from the slot's previous owner)
        # are masked out.
        attn_mask = key_positions[None, None, None, :] <= positions[:, None, :, None]
        return SlotBatch(self, slots, positions, attn_mask)

    def update(self, layer: int, slots: Tensor, positions: Tensor, keys: Tensor, values: Tensor,
               length: int) -> tuple[Tensor, Tensor]:
        """Store new (B, H, T, D) keys/values, return the (B, H, length, D) history.

        Writes happen before reads, so a position overwritten by padding in an
        earlier batched prefill is always rewritten by the real token first.
        """
        # Two advanced indices separated by a slice put the indexed dims first:
        # the target view is (B, T, H, D).
        self.k[layer][slots[:, None], :, positions] = keys.transpose(1, 2)
        self.v[layer][slots[:, None], :, positions] = values.transpose(1, 2)
        return self.k[layer][slots, :, :length], self.v[layer][slots, :, :length]
