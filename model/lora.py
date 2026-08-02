# model/lora.py
"""Low-Rank Adaptation (LoRA) for parameter-efficient finetuning.

Implements custom LoRA linear layer wrappers and model injection utilities,
built from scratch using PyTorch primitives.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


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


def inject_lora(
    model: nn.Module,
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: list[str] = ["W_query", "W_value"],
) -> None:
    """Traverse the model and wrap target nn.Linear layers in LoRALinear adapters.

    Args:
        model: PyTorch model module.
        r: Rank of the low-rank projection.
        alpha: Scaling coefficient.
        dropout: Dropout rate applied before low-rank projection.
        target_modules: List of module attribute names to adapt (e.g. W_query, W_value).
    """
    for name, module in model.named_modules():
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, nn.Linear) and attr_name in target_modules:
                lora_layer = LoRALinear(attr, r=r, alpha=alpha, dropout=dropout)
                setattr(module, attr_name, lora_layer)

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
