# app/adapters.py
"""LoRA adapter registry: safe naming, validation, and caching.

Adapters live as `checkpoints/adapters/<name>.pt`. Everything that turns an
untrusted name into a file path, or an adapter file into weights applied to
the live model, goes through this module so the rules are enforced in one
place:

- names are a strict allow-list pattern (no path traversal, bounded length);
- files load with `weights_only=True` (no pickle code execution);
- an adapter must match the running model's architecture exactly -- every
  expected LoRA key present with the right shape -- so it is applied fully
  or not at all, never half-loaded under `strict=False`.
"""

import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

import torch

from model.lora import LORA_TARGET_MODULES, lora_hparams_from_checkpoint

ADAPTERS_DIR = os.path.join("checkpoints", "adapters")
ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Request-level selectors (see GenerationRequest.adapter), so never file names.
BASE_MODEL = "none"
RESERVED_NAMES = frozenset({BASE_MODEL, "default", "base"})
# Adapters shipped with the repo. Teach Mode may not create or overwrite them.
PROTECTED_PREFIXES = ("sft_",)


class AdapterError(ValueError):
    """The adapter name or file is invalid, or incompatible with the model."""


class AdapterNotFound(AdapterError):
    pass


@dataclass(frozen=True, eq=False)
class Adapter:
    name: str
    r: int
    alpha: float
    state_dict: dict


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not ADAPTER_NAME_RE.match(name):
        raise AdapterError("Adapter names must be 1-64 characters of letters, digits, '-' or '_'.")
    return name


def is_protected(name: str) -> bool:
    return name.lower() in RESERVED_NAMES or name.startswith(PROTECTED_PREFIXES)


def adapter_path(name: str) -> str:
    return os.path.join(ADAPTERS_DIR, f"{validate_name(name)}.pt")


def list_adapters() -> list[str]:
    if not os.path.isdir(ADAPTERS_DIR):
        return []
    names = (f[:-3] for f in os.listdir(ADAPTERS_DIR) if f.endswith(".pt"))
    return sorted(n for n in names if ADAPTER_NAME_RE.match(n) and n.lower() not in RESERVED_NAMES)


def expected_lora_shapes(model_config: dict, r: int) -> dict[str, tuple[int, int]]:
    emb_dim = model_config["emb_dim"]
    shapes = {}
    for i in range(model_config["n_layers"]):
        for target in LORA_TARGET_MODULES:
            prefix = f"trf_blocks.{i}.att.{target}"
            shapes[f"{prefix}.lora_A"] = (r, emb_dim)
            shapes[f"{prefix}.lora_B"] = (emb_dim, r)
    return shapes


def load_adapter(name: str, model_config: dict) -> Adapter:
    """Load and validate an adapter file against the running model's config."""
    path = adapter_path(name)
    if not os.path.isfile(path):
        raise AdapterNotFound(f"Adapter '{name}' not found.")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # corrupt file, git-lfs pointer, non-tensor payload...
        raise AdapterError(f"Adapter '{name}' could not be read.") from exc

    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise AdapterError(f"Adapter '{name}' contains no LoRA weights.")

    saved_cfg = checkpoint.get("model_config") or {}
    for key in ("n_layers", "emb_dim", "n_heads"):
        if key in saved_cfg and saved_cfg[key] != model_config.get(key):
            raise AdapterError(f"Adapter '{name}' was trained for a different model architecture.")

    r, alpha = lora_hparams_from_checkpoint(checkpoint)
    expected = expected_lora_shapes(model_config, r)
    if set(state_dict) != set(expected) or any(
        not isinstance(t, torch.Tensor) or tuple(t.shape) != expected[k] for k, t in state_dict.items()
    ):
        raise AdapterError(f"Adapter '{name}' does not match this model's architecture.")

    return Adapter(name=name, r=r, alpha=alpha, state_dict=state_dict)


class AdapterRegistry:
    """Thread-safe LRU cache of validated adapters for one model config.

    Entries are keyed by file mtime, so an adapter that is replaced on disk is
    re-validated instead of served stale.
    """

    def __init__(self, model_config: dict, max_cached: int = 8) -> None:
        self.model_config = model_config
        self.max_cached = max_cached
        self._cache: OrderedDict[str, tuple[float, Adapter]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str) -> Adapter:
        path = adapter_path(name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            raise AdapterNotFound(f"Adapter '{name}' not found.") from None

        with self._lock:
            cached = self._cache.get(name)
            if cached is not None and cached[0] == mtime:
                self._cache.move_to_end(name)
                return cached[1]

        adapter = load_adapter(name, self.model_config)
        with self._lock:
            self._cache[name] = (mtime, adapter)
            self._cache.move_to_end(name)
            while len(self._cache) > self.max_cached:
                self._cache.popitem(last=False)
        return adapter
