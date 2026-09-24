# app/inference.py
"""Unified object-oriented inference engine for GPT-2.

Loads checkpoint, auto-detects hardware (CUDA/CPU) and precision, handles
tokenization, and serves generation through a continuous-batching scheduler
(app/batching.py) so concurrent requests share forward passes.
"""

import os
import sys
import threading
from contextlib import contextmanager
from typing import Iterator

import torch

# Add root folder to path to enable clean imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import AdapterRegistry
from app.batching import ContinuousBatcher, EngineBusy, GenerationHandle, SamplingParams
from model.gpt import GPTModel, count_parameters
from model.lora import LORA_TARGET_MODULES, LoRAPool, inject_lora, install_multi_lora, lora_hparams_from_checkpoint
from model.tokenizer import GPT2Tokenizer

# Default for the `adapter` argument: generate with whatever adapter is
# currently applied (set_adapter), rather than switching.
CURRENT_ADAPTER = object()

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _resolve_device(device: str | torch.device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    """'auto': bf16 on GPUs that support it, else fp16 on CUDA, else fp32.

    Half precision halves weight and KV-cache memory and roughly doubles GPU
    throughput; on CPU it is usually slower, so CPU stays in fp32.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return _DTYPES[dtype]


def _default_batch_size(device: torch.device) -> int:
    # Measured on 4 CPU cores, GPT-2 small, 8 concurrent requests: 1 slot
    # 25 tok/s, 4 slots 48 tok/s, 8 slots 68 tok/s. GPUs keep scaling further.
    return int(os.environ.get("ENGINE_MAX_BATCH", 16 if device.type == "cuda" else 8))


class GPTInferenceEngine:
    """Encapsulated text generation engine for the custom GPT-2 model.

    Thread-safe: generate/generate_stream/submit may be called concurrently;
    a single scheduler thread batches them onto the model.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto", dtype: str | torch.dtype | None = None) -> None:
        """Initialize the inference engine and load model weights from a checkpoint.

        Args:
            checkpoint_path: Path to the PyTorch checkpoint (.pt file).
            device: Hardware selection ('auto', 'cpu', 'cuda').
            dtype: 'auto', 'float32', 'bfloat16' or 'float16'
                (default: the MODEL_DTYPE env var, else 'auto').
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

        self._setup(model, model_config, _resolve_device(device), dtype)

    @classmethod
    def from_config(cls, model_config: dict, device: str = "cpu", dtype=None) -> "GPTInferenceEngine":
        """Build an engine around a randomly initialized model.

        Used as a cold-start placeholder that keeps the server responsive while
        real weights download; the API never serves generations from it.
        """
        engine = cls.__new__(cls)
        engine._setup(GPTModel(model_config), model_config, _resolve_device(device), dtype)
        return engine

    def _setup(self, model: GPTModel, model_config: dict, device: torch.device, dtype) -> None:
        self.device = device
        self.dtype = _resolve_dtype(dtype or os.environ.get("MODEL_DTYPE", "auto"), device)
        self.model_config = model_config
        self.model_size = model_config.get("model_size", "small")
        self.model = model.to(device=device, dtype=self.dtype).eval()
        # Adapters are served from a shared pool rather than by rewiring the
        # model, so one batch can mix requests for different adapters.
        self.lora_pool = LoRAPool(
            n_layers=model_config["n_layers"], emb_dim=model_config["emb_dim"],
            # At least 2: the default adapter pins one entry, and a pool with no
            # other entry could never admit a request for a different adapter.
            max_adapters=max(2, int(os.environ.get("ENGINE_MAX_ADAPTERS", 8))), device=device, dtype=self.dtype,
        )
        install_multi_lora(self.model, self.lora_pool)
        self.tokenizer = GPT2Tokenizer()
        self.eos_id = self.tokenizer.eos_id
        self.context_size = self.model.pos_emb.weight.shape[0]
        self.parameter_count = count_parameters(self.model)
        self.adapters = AdapterRegistry(model_config)
        self._active_adapter = None
        self._batcher: ContinuousBatcher | None = None
        self._batcher_lock = threading.Lock()

    # ── Batching ──────────────────────────────────────────────────────

    @property
    def batcher(self) -> ContinuousBatcher:
        with self._batcher_lock:
            if self._batcher is None:
                self._batcher = ContinuousBatcher(
                    self,
                    max_batch_size=_default_batch_size(self.device),
                    max_waiting=int(os.environ.get("ENGINE_MAX_QUEUE", 8)),
                )
            return self._batcher

    @property
    def model_lock(self) -> threading.Lock:
        """Held by the scheduler for every step. Take it to read or rewire the
        model from outside (e.g. snapshotting weights for fine-tuning)."""
        return self.batcher.model_lock

    @contextmanager
    def exclusive(self, timeout: float = 30.0):
        if not self.model_lock.acquire(timeout=timeout):
            raise EngineBusy("Timed out waiting for exclusive access to the model.")
        try:
            yield
        finally:
            self.model_lock.release()

    def close(self) -> None:
        """Stop the scheduler thread (if it was ever started)."""
        if self._batcher is not None:
            self._batcher.close()

    # ── Adapters ──────────────────────────────────────────────────────

    @property
    def active_adapter(self) -> str | None:
        """Name of the LoRA adapter currently applied, or None for the base model."""
        return self._active_adapter.name if self._active_adapter is not None else None

    def set_adapter(self, adapter) -> None:
        """Make `adapter` (app.adapters.Adapter, or None for the base model) the default.

        The default applies to requests submitted without an explicit adapter
        and to direct forward passes (e.g. the eval harness). To use an
        adapter for one request only, pass it to generate()/submit().
        """
        with self.model_lock:
            self.lora_pool.set_default(adapter)
            self._active_adapter = adapter

    # ── Generation ────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode_ordinary(text))

    def submit(self, prompt: str, adapter=CURRENT_ADAPTER, **sampling) -> GenerationHandle:
        """Queue a generation; returns immediately. Raises EngineBusy if the
        wait queue is full.

        `sampling` takes the SamplingParams fields (max_new_tokens,
        temperature, top_k, top_p, penalties, min_new_tokens, use_cache).
        """
        # encode_ordinary: the prompt carries user text and web snippets, so a
        # literal "<|endoftext|>" in it must stay plain text, not a control token.
        handle = GenerationHandle(
            self.tokenizer.encode_ordinary(prompt),
            SamplingParams(**sampling),
            self._active_adapter if adapter is CURRENT_ADAPTER else adapter,
        )
        return self.batcher.submit(handle)

    def _admitted(self, handle: GenerationHandle, admit_timeout: float | None) -> None:
        if not handle.wait_admitted(admit_timeout):
            handle.cancel()
            raise EngineBusy("The server is busy; the request was not scheduled in time.")

    def generate(self, prompt: str, adapter=CURRENT_ADAPTER, admit_timeout: float | None = None, **sampling) -> dict:
        """Generate a completion and return it with latency statistics.

        Returns:
            Dictionary with the GenerationResponse fields plus `completion_text`
            (only the newly generated text, decoded from the new token ids --
            so callers never have to split it back out of the prompt).
        """
        handle = self.submit(prompt, adapter=adapter, **sampling)
        self._admitted(handle, admit_timeout)
        token_ids = list(handle.tokens())
        latency = handle.elapsed()
        completion = self.tokenizer.decode(token_ids)
        return {
            "prompt": prompt,
            "generated_text": prompt + completion,
            "completion_text": completion,
            "tokens_generated": len(token_ids),
            "time_taken_seconds": latency,
            "tokens_per_second": len(token_ids) / latency if latency > 0 else 0.0,
            "finish_reason": handle.finish_reason,
        }

    def stream_text(self, handle: GenerationHandle) -> Iterator[tuple[str, float, int]]:
        """Yield (text_chunk, latency_seconds, tokens_generated) for a submitted handle.

        Multi-byte UTF-8 characters can span several BPE tokens; bytes are
        buffered until they form valid text, so a chunk may be "".
        """
        tokens_generated = 0
        byte_buffer = b""
        for token_id in handle.tokens():
            tokens_generated += 1
            byte_buffer += self.tokenizer.encoding.decode_single_token_bytes(token_id)
            try:
                text_chunk = byte_buffer.decode("utf-8")
                byte_buffer = b""
            except UnicodeDecodeError:
                text_chunk = ""
            yield text_chunk, handle.elapsed(), tokens_generated

        if byte_buffer:
            yield byte_buffer.decode("utf-8", errors="replace"), handle.elapsed(), tokens_generated

    def generate_stream(self, prompt: str, adapter=CURRENT_ADAPTER, admit_timeout: float | None = None,
                        **sampling) -> Iterator[tuple[str, float, int]]:
        """Streaming variant of generate(); closing the iterator cancels generation."""
        handle = self.submit(prompt, adapter=adapter, **sampling)
        try:
            self._admitted(handle, admit_timeout)
            yield from self.stream_text(handle)
        finally:
            handle.cancel()
