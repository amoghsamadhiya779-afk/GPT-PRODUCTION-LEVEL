# model/lora.py
"""Low-Rank Adaptation (LoRA) for parameter-efficient finetuning.

Implements custom LoRA linear layer wrappers and model injection utilities,
built from scratch using PyTorch primitives.
"""

import math
import threading
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Attention projections every adapter in this repo adapts.
LORA_TARGET_MODULES = ("W_query", "W_value")


class LoRALinear(nn.Module):
    """Custom Low-Rank Adaptation (LoRA) Linear layer wrapper.

    Wraps an existing frozen nn.Linear layer and computes the parallel
    low-rank parameter update: h = W_0 * x + (alpha / r) * B * A * x
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Reference to frozen base linear layer
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        # Low-rank adapter weights: A (r x in) and B (out x r)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Optional dropout on inputs
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize adapter weights
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize LoRA weights."""
        # A is Kaiming uniform initialized, B is zero-initialized
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        # Standard frozen projection path
        base_out = self.linear(x)

        # Adapter path: (x @ A.T) @ B.T
        adapter_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()

        return base_out + adapter_out * self.scaling


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze all model parameters except LoRA parameters."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def lora_hparams_from_checkpoint(checkpoint: dict) -> tuple[int, float]:
    """Return (r, alpha) for a LoRA checkpoint.

    Older checkpoints may not record them: r is then inferred from the shape
    of any lora_A matrix (r x in_features) and alpha falls back to the 2*r
    convention every adapter in this repo was trained with.
    """
    r = checkpoint.get("lora_r")
    if r is None:
        a_ranks = [v.shape[0] for k, v in checkpoint["model_state_dict"].items() if k.endswith("lora_A")]
        r = a_ranks[0] if a_ranks else 4
    alpha = checkpoint.get("lora_alpha")
    if alpha is None:
        alpha = 2.0 * r
    return int(r), float(alpha)


def inject_lora(
    model: nn.Module,
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES,
) -> None:
    """Traverse the model and wrap target nn.Linear layers in LoRALinear adapters.

    Args:
        model: PyTorch model module.
        r: Rank of the low-rank projection.
        alpha: Scaling coefficient.
        dropout: Dropout rate applied before low-rank projection.
        target_modules: List of module attribute names to adapt (e.g. W_query, W_value).
    """
    # Snapshot the module list first so freshly inserted LoRALinear wrappers
    # (and the nn.Linear inside them) are never visited and wrapped again.
    for module in list(model.modules()):
        for attr_name, child in list(module.named_children()):
            if attr_name in target_modules and isinstance(child, nn.Linear):
                setattr(module, attr_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))

    # Freeze non-adapter parameters
    mark_only_lora_as_trainable(model)


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract a state dictionary containing only the trainable LoRA parameters."""
    return {name: param for name, param in model.named_parameters() if "lora_" in name}


def strip_lora_wrapper_keys(state_dict: dict) -> dict:
    """Convert a (possibly LoRA-wrapped) state dict into base-model key names.

    LoRALinear wraps a linear as `self.linear`, so a wrapped module's weights
    are saved under keys like `...W_query.linear.weight` instead of
    `...W_query.weight`, alongside `...W_query.lora_A` / `lora_B`. Loading
    such a state dict `strict=False` into a plain (unwrapped) GPTModel does
    NOT raise, but it also does not populate `W_query.weight` -- the base
    weights silently stay at random init. This produces a state dict with
    LoRA adapter params dropped and wrapper keys renamed back to the
    unwrapped form, so it loads correctly into a plain GPTModel regardless of
    whether the source model currently has adapters injected.
    """
    cleaned = {}
    for k, v in state_dict.items():
        if ".lora_A" in k or ".lora_B" in k:
            continue
        cleaned[k.replace(".linear.weight", ".weight").replace(".linear.bias", ".bias")] = v
    return cleaned


def remove_lora(model: nn.Module) -> int:
    """Unwrap any LoRALinear layers back to their original frozen nn.Linear.

    The wrapped base linear's weights are never modified by LoRA training
    (only lora_A/lora_B are), so unwrapping always restores true base-model
    behavior -- this is what makes it safe to call before re-injecting an
    adapter of a different rank, or to fully deactivate an adapter instead of
    just zeroing it (which would otherwise leave the extra matmuls wired into
    every forward pass permanently).

    Returns the number of LoRALinear layers that were unwrapped.
    """
    removed = 0
    for module in model.modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base_linear = child.linear
                base_linear.weight.requires_grad = True
                if base_linear.bias is not None:
                    base_linear.bias.requires_grad = True
                setattr(module, attr_name, base_linear)
                removed += 1
    return removed


# ── Multi-adapter serving ──────────────────────────────────────────────
#
# LoRALinear bakes ONE adapter into the model, so a batch can only use one.
# For serving, the adapters live in a LoRAPool instead and each targeted
# projection is a MultiLoRALinear that adds a different adapter's update to
# every batch row: out_b = W x_b + B_{i(b)} A_{i(b)} x_b, computed as one
# batched matmul over the pooled weights gathered by row (the S-LoRA/Punica
# approach, in plain PyTorch).


@dataclass(frozen=True)
class LoRARows:
    """Which pooled adapter each batch row uses (index 0 = no adapter)."""
    idx: Tensor        # (batch,) long
    any_adapter: bool  # False: skip the adapter matmuls entirely


class LoRAPool:
    """Up to `max_adapters` LoRA adapters resident at once, stacked per projection.

    For each adapted projection A is stored as (P, rank, in) and B as
    (P, out, rank), with B pre-scaled by alpha / r so rows needn't know their
    adapter's scaling. Entry 0 is all zeros and means "no adapter". Adapters
    of lower rank are zero-padded: the extra rank dimensions multiply by zero,
    so B @ A -- and the output -- is exactly unchanged.

    Entries are reference-counted by the requests using them and evicted
    least-recently-used when free. reserve() only does bookkeeping; the
    weight copy happens in load_pending(), which the caller runs while
    holding the model lock, so a forward pass never reads a half-written entry.
    """

    def __init__(self, n_layers: int, emb_dim: int, max_adapters: int = 8, rank: int = 16,
                 device="cpu", dtype=torch.float32, targets=LORA_TARGET_MODULES) -> None:
        self.keys = [f"trf_blocks.{i}.att.{t}" for i in range(n_layers) for t in targets]
        self.capacity = max_adapters
        self.rank = rank
        size = max_adapters + 1
        self.A = {k: torch.zeros(size, rank, emb_dim, device=device, dtype=dtype) for k in self.keys}
        self.B = {k: torch.zeros(size, emb_dim, rank, device=device, dtype=dtype) for k in self.keys}
        self.default_index = 0
        self._lock = threading.Lock()
        self._adapter_at = [None] * size
        self._refs = [0] * size
        self._last_used = [0.0] * size
        self._pending: dict[int, object] = {}

    def __getstate__(self) -> dict:
        # Locks can't be pickled; without this a model wrapped with
        # MultiLoRALinear couldn't be deep-copied or saved whole.
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def reserve(self, adapter) -> int | None:
        """Pin `adapter` (app.adapters.Adapter or None) and return its index,
        or None if every entry is in use by other adapters."""
        if adapter is None:
            return 0
        with self._lock:
            for i in range(1, self.capacity + 1):
                if self._adapter_at[i] is adapter:
                    self._refs[i] += 1
                    self._last_used[i] = time.monotonic()
                    return i
            free = [i for i in range(1, self.capacity + 1) if self._refs[i] == 0]
            if not free:
                return None
            # Prefer never-used entries, then the least recently used one.
            i = min(free, key=lambda j: (self._adapter_at[j] is not None, self._last_used[j]))
            self._adapter_at[i] = adapter
            self._refs[i] = 1
            self._last_used[i] = time.monotonic()
            self._pending[i] = adapter
            return i

    def release(self, index: int) -> None:
        if index:
            with self._lock:
                self._refs[index] -= 1

    def load_pending(self) -> None:
        """Copy newly reserved adapters' weights in. Caller holds the model lock."""
        with self._lock:
            pending, self._pending = self._pending, {}
        for index, adapter in pending.items():
            if adapter.r > self.rank:
                self._grow_rank(adapter.r)
            scale = adapter.alpha / adapter.r
            for key in self.keys:
                a = adapter.state_dict[f"{key}.lora_A"]
                b = adapter.state_dict[f"{key}.lora_B"]
                self.A[key][index].zero_()
                self.B[key][index].zero_()
                self.A[key][index, : adapter.r] = a
                self.B[key][index, :, : adapter.r] = b.float() * scale

    def _grow_rank(self, rank: int) -> None:
        extra = rank - self.rank
        for key in self.keys:
            self.A[key] = F.pad(self.A[key], (0, 0, 0, extra))
            self.B[key] = F.pad(self.B[key], (0, extra))
        self.rank = rank

    def set_default(self, adapter) -> None:
        """Adapter applied when a forward pass doesn't pass per-row `lora`.
        Caller holds the model lock."""
        index = self.reserve(adapter)
        if index is None:
            raise RuntimeError("All LoRA pool entries are in use; cannot load another adapter.")
        self.load_pending()
        previous, self.default_index = self.default_index, index
        self.release(previous)

    def rows(self, indices: list[int], device) -> LoRARows:
        return LoRARows(torch.tensor(indices, dtype=torch.long, device=device), any(i != 0 for i in indices))

    def default_rows(self, batch: int, device) -> LoRARows | None:
        if self.default_index == 0:
            return None
        return LoRARows(torch.full((batch,), self.default_index, dtype=torch.long, device=device), True)


class MultiLoRALinear(nn.Module):
    """A projection that applies a (possibly different) pooled adapter per batch row.

    Registers no parameters of its own -- the pool's tensors are shared by
    all layers and excluded from state_dict -- and keeps the wrapped layer as
    `self.linear`, so strip_lora_wrapper_keys recovers the base weights.
    """

    def __init__(self, linear: nn.Module, pool: LoRAPool, key: str) -> None:
        super().__init__()
        self.linear = linear
        self.pool = pool
        self.key = key

    def forward(self, x: Tensor, lora: LoRARows | None = None) -> Tensor:
        out = self.linear(x)
        rows = lora if lora is not None else self.pool.default_rows(x.shape[0], x.device)
        if rows is None or not rows.any_adapter:
            return out
        a = self.pool.A[self.key][rows.idx]  # (batch, rank, in)
        b = self.pool.B[self.key][rows.idx]  # (batch, out, rank)
        return out + (x @ a.transpose(1, 2)) @ b.transpose(1, 2)


def install_multi_lora(model: nn.Module, pool: LoRAPool, targets=LORA_TARGET_MODULES) -> None:
    """Wrap every targeted attention projection in a MultiLoRALinear."""
    for i, block in enumerate(model.trf_blocks):
        for target in targets:
            layer = getattr(block.att, target)
            if not isinstance(layer, MultiLoRALinear):
                setattr(block.att, target, MultiLoRALinear(layer, pool, f"trf_blocks.{i}.att.{target}"))
