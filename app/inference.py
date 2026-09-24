# app/inference.py
"""Unified object-oriented inference engine for GPT-2.

Loads checkpoint, auto-detects hardware (CUDA/CPU), handles tokenization,
and benchmarks text generation using standard or KV-Cached mode.
"""

import os
import sys
import time
from typing import Iterator

import torch

# Add root folder to path to enable clean imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import AdapterRegistry
from model.gpt import GPTModel, generate_stream, count_parameters
from model.lora import LORA_TARGET_MODULES, inject_lora, lora_hparams_from_checkpoint, remove_lora
from model.tokenizer import GPT2Tokenizer


def _resolve_device(device: str | torch.device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class GPTInferenceEngine:
    """Encapsulated text generation engine for the custom GPT-2 model.

    Not thread-safe: callers must serialize generate/generate_stream/
    set_adapter calls (the API does this with its engine gate).
    """

    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        """Initialize the inference engine and load model weights from a checkpoint.

        Args:
            checkpoint_path: Path to the PyTorch checkpoint (.pt file).
            device: Hardware selection ('auto', 'cpu', 'cuda').
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        # weights_only=True: a checkpoint is data, never code -- this refuses
        # the arbitrary-object unpickling that makes plain torch.load an RCE.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_config = checkpoint["model_config"]
        model = GPTModel(model_config)

        if checkpoint.get("is_lora", False):
            lora_r, lora_alpha = lora_hparams_from_checkpoint(checkpoint)
            inject_lora(model, r=lora_r, alpha=lora_alpha, target_modules=LORA_TARGET_MODULES)
            # strict=False since a LoRA checkpoint only contains adapter weights
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        self._setup(model, model_config, _resolve_device(device))

    @classmethod
    def from_config(cls, model_config: dict, device: str = "cpu") -> "GPTInferenceEngine":
        """Build an engine around a randomly initialized model.

        Used as a cold-start placeholder that keeps the server responsive while
        real weights download; the API never serves generations from it.
        """
        engine = cls.__new__(cls)
        engine._setup(GPTModel(model_config), model_config, _resolve_device(device))
        return engine

    def _setup(self, model: GPTModel, model_config: dict, device: torch.device) -> None:
        self.device = device
        self.model_config = model_config
        self.model_size = model_config.get("model_size", "small")
        self.model = model.to(device).eval()
        self.tokenizer = GPT2Tokenizer()
        self.eos_id = self.tokenizer.eos_id
        self.context_size = self.model.pos_emb.weight.shape[0]
        self.parameter_count = count_parameters(self.model)
        self.adapters = AdapterRegistry(model_config)
        self._active_adapter = None

    # ── Adapters ──────────────────────────────────────────────────────

    @property
    def active_adapter(self) -> str | None:
        """Name of the LoRA adapter currently applied, or None for the base model."""
        return self._active_adapter.name if self._active_adapter is not None else None

    def set_adapter(self, adapter) -> None:
        """Apply a validated adapter (app.adapters.Adapter), or None for the base model.

        A no-op when that exact adapter is already applied, so per-request
        adapter selection costs nothing in the common case.
        """
        if adapter is self._active_adapter:
            return
        # Always unwrap first: adapters can have different ranks (r=16 SFT vs
        # r=4 Teach Mode), and load_state_dict raises on a shape mismatch.
        remove_lora(self.model)
        self._active_adapter = None
        if adapter is not None:
            try:
                inject_lora(self.model, r=adapter.r, alpha=adapter.alpha, target_modules=LORA_TARGET_MODULES)
                self.model.to(self.device)  # freshly created LoRA params start on CPU
                self.model.load_state_dict(adapter.state_dict, strict=False)
            except Exception:
                remove_lora(self.model)  # all-or-nothing: never leave half-wired layers
                raise
            self._active_adapter = adapter
        self.model.eval()

    # ── Generation ────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode_ordinary(text))

    def _generate_ids(self, prompt: str, max_new_tokens: int, **sampling) -> Iterator[int]:
        """Yield generated token ids, stopping at (and never yielding) EOS."""
        # encode_ordinary: the prompt carries user text and web snippets, so a
        # literal "<|endoftext|>" in it must stay plain text, not a control token.
        input_ids = torch.tensor([self.tokenizer.encode_ordinary(prompt)], device=self.device)
        for token_id in generate_stream(
            model=self.model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=self.context_size,
            eos_id=self.eos_id,
            **sampling,
        ):
            if token_id == self.eos_id:
                return
            yield token_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
        use_cache: bool = True,
    ) -> dict:
        """Generate text from a prompt and return the output with latency statistics.

        Returns:
            Dictionary with the GenerationResponse fields plus `completion_text`
            (only the newly generated text, decoded from the new token ids --
            so callers never have to split it back out of the prompt).
        """
        start_time = time.perf_counter()
        token_ids = list(self._generate_ids(
            prompt,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            use_cache=use_cache,
        ))
        latency = time.perf_counter() - start_time

        completion = self.tokenizer.decode(token_ids)
        return {
            "prompt": prompt,
            "generated_text": prompt + completion,
            "completion_text": completion,
            "tokens_generated": len(token_ids),
            "time_taken_seconds": latency,
            "tokens_per_second": len(token_ids) / latency if latency > 0 else 0.0,
        }

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
        use_cache: bool = True,
    ) -> Iterator[tuple[str, float, int]]:
        """Yield (text_chunk, latency_seconds, tokens_generated) as tokens are produced.

        Multi-byte UTF-8 characters can span several BPE tokens; bytes are
        buffered until they form valid text, so a chunk may be "".
        """
        start_time = time.perf_counter()
        tokens_generated = 0
        byte_buffer = b""

        for token_id in self._generate_ids(
            prompt,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            use_cache=use_cache,
        ):
            tokens_generated += 1
            byte_buffer += self.tokenizer.encoding.decode_single_token_bytes(token_id)
            try:
                text_chunk = byte_buffer.decode("utf-8")
                byte_buffer = b""
            except UnicodeDecodeError:
                text_chunk = ""
            yield text_chunk, time.perf_counter() - start_time, tokens_generated

        if byte_buffer:
            yield byte_buffer.decode("utf-8", errors="replace"), time.perf_counter() - start_time, tokens_generated
