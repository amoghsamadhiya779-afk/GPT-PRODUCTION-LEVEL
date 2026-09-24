# app/batching.py
"""Continuous batching: many concurrent requests share every forward pass.

Serving one request at a time leaves most of the hardware idle -- a decode
step for one sequence is a (1 x d) @ (d x d) matmul, and a batch of 8 costs
barely more. The scheduler below keeps up to `max_batch_size` sequences in
flight and, at every iteration:

1. admits waiting requests into free slots (between steps, never mid-step),
2. prefills all newly admitted prompts in one padded forward pass,
3. runs ONE batched decode step for every other active sequence,
4. samples each row with that request's own settings and streams the token,
5. retires finished, cancelled, or failed sequences, freeing their slots.

Keys/values live in a shared model.kv_cache.SlotKVCache. Semantics match the
single-request path in model.gpt.generate_stream exactly -- same penalties,
same min_new_tokens handling, same sliding window past the context length --
and tests/test_batching.py checks batched output token-for-token against it.

LoRA adapters are applied to the whole model, so one batch runs one adapter.
Admission is FIFO: when the next waiting request needs a different adapter,
no one else is admitted until the batch drains, so it can't be starved.
"""

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from model.gpt import _sample_next_token
from model.kv_cache import SlotKVCache

logger = logging.getLogger(__name__)


class EngineBusy(RuntimeError):
    """The wait queue is full, or the request wasn't admitted in time."""


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    min_new_tokens: int = 0
    use_cache: bool = True


class GenerationHandle:
    """One request's view of its generation. Thread-safe for its consumer."""

    def __init__(self, prompt_ids: list[int], params: SamplingParams, adapter) -> None:
        if not prompt_ids:
            raise ValueError("Prompt encodes to zero tokens.")
        self.prompt_ids = prompt_ids
        self.params = params
        self.adapter = adapter
        self.submitted_at = time.perf_counter()
        self.admitted_at: float | None = None
        self.finished_at: float | None = None
        self.finish_reason: str | None = None  # "stop" | "length" | "cancelled" | "error"
        self.generated: list[int] = []

        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._admitted = threading.Event()
        self._cancelled = threading.Event()

        # Scheduler-owned state.
        self._ids = list(prompt_ids)  # prompt + generated; the last one isn't in the cache yet
        self._slot: int | None = None
        self._position = 0            # number of tokens stored in the slot

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        """Stop generating; the slot is freed at the next scheduler step."""
        self._cancelled.set()

    def wait_admitted(self, timeout: float | None = None) -> bool:
        return self._admitted.wait(timeout)

    def tokens(self) -> Iterator[int]:
        """Yield generated token ids as they arrive (blocking)."""
        while True:
            kind, value = self._events.get()
            if kind == "token":
                yield value
            elif kind == "error":
                raise RuntimeError("Generation failed") from value
            else:
                return

    def elapsed(self) -> float:
        """Seconds spent generating (excluding time spent waiting to be admitted)."""
        start = self.admitted_at or self.submitted_at
        return (self.finished_at or time.perf_counter()) - start


class ContinuousBatcher:
    """Scheduler thread driving batched generation for one engine."""

    def __init__(self, engine, max_batch_size: int = 8, max_waiting: int = 8) -> None:
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_waiting = max_waiting
        # Held for every scheduler step. Anything else that reads or rewires
        # the model (applying an adapter, snapshotting weights) takes it too.
        self.model_lock = threading.Lock()

        self._cv = threading.Condition()
        self._waiting: deque[GenerationHandle] = deque()
        self._active: list[GenerationHandle] = []
        self._free_slots = list(range(max_batch_size - 1, -1, -1))
        self._cache: SlotKVCache | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        self.peak_batch_size = 0  # most sequences ever decoded in one step

    # ── Public API ────────────────────────────────────────────────────

    def submit(self, handle: GenerationHandle) -> GenerationHandle:
        with self._cv:
            self._drop_cancelled_waiting()
            # Requests that will take a free slot at the next step aren't
            # really waiting; only the overflow counts against the limit.
            if len(self._waiting) - len(self._free_slots) >= self.max_waiting:
                raise EngineBusy("Too many requests are already waiting.")
            self._waiting.append(handle)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="continuous-batcher", daemon=True)
                self._thread.start()
            self._cv.notify()
        return handle

    def stats(self) -> dict:
        with self._cv:
            return {"active": len(self._active), "waiting": len(self._waiting),
                    "max_batch_size": self.max_batch_size, "max_waiting": self.max_waiting,
                    "peak_batch_size": self.peak_batch_size}

    def close(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify()

    # ── Scheduler loop ────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stopping and not self._waiting and not self._active:
                    self._cv.wait()
                if self._stopping:
                    return
                admitted = self._admit()

            with self.model_lock, torch.inference_mode():
                try:
                    self._step(admitted)
                except Exception as exc:
                    logger.exception("Batched generation step failed; failing %d request(s)",
                                     len(self._active) + len(admitted))
                    for handle in set(self._active) | set(admitted):
                        self._finish(handle, "error", exc)

    def _drop_cancelled_waiting(self) -> None:
        for handle in [h for h in self._waiting if h.cancelled]:
            self._waiting.remove(handle)
            handle.finish_reason = "cancelled"
            handle.finished_at = time.perf_counter()
            handle._admitted.set()
            handle._events.put(("done", "cancelled"))

    def _admit(self) -> list[GenerationHandle]:
        """Move waiting requests into free slots (called with the lock held)."""
        self._drop_cancelled_waiting()
        admitted = []
        while self._waiting and self._free_slots:
            head = self._waiting[0]
            batch_running = bool(self._active or admitted)
            batch_adapter = self._active[0].adapter if self._active else (admitted[0].adapter if admitted else None)
            if batch_running and head.adapter is not batch_adapter:
                break  # FIFO: wait for the batch to drain, then switch adapters
            self._waiting.popleft()
            head._slot = self._free_slots.pop()
            head.admitted_at = time.perf_counter()
            head._admitted.set()  # it has a slot; generation starts at the next step
            admitted.append(head)
        return admitted

    def _step(self, admitted: list[GenerationHandle]) -> None:
        for handle in [h for h in self._active if h.cancelled]:
            self._finish(handle, "cancelled")

        if admitted:
            if not self._active:
                self.engine._apply_adapter(admitted[0].adapter)
            self._active.extend(admitted)
            self.peak_batch_size = max(self.peak_batch_size, len(self._active))
            self._prefill([h for h in admitted if not h.cancelled])

        decode = [h for h in self._active if h not in admitted and h.finish_reason is None]
        if decode:
            self._decode(decode)

    # ── Model work ────────────────────────────────────────────────────

    @property
    def cache(self) -> SlotKVCache:
        if self._cache is None:
            cfg = self.engine.model_config
            param = next(self.engine.model.parameters())
            self._cache = SlotKVCache(
                n_layers=cfg["n_layers"], max_slots=self.max_batch_size, n_heads=cfg["n_heads"],
                head_dim=cfg["emb_dim"] // cfg["n_heads"], max_len=self.engine.context_size,
                device=param.device, dtype=param.dtype,
            )
        return self._cache

    def _prefill(self, handles: list[GenerationHandle]) -> None:
        """First forward pass for new requests; samples their first token."""
        ctx = self.engine.context_size
        cached = [h for h in handles if h.params.use_cache]
        if cached:
            device = self.cache.k.device
            prompts = [h._ids[-ctx:] for h in cached]  # like generate_stream: keep the last ctx tokens
            lengths = torch.tensor([len(p) for p in prompts], device=device)
            width = int(lengths.max())
            inputs = torch.full((len(cached), width), self.engine.eos_id, dtype=torch.long, device=device)
            for row, prompt in enumerate(prompts):
                inputs[row, : len(prompt)] = torch.tensor(prompt, device=device)
            positions = torch.arange(width, device=device).expand(len(cached), width)
            slots = torch.tensor([h._slot for h in cached], device=device)
            logits = self.engine.model(inputs, slot_batch=self.cache.batch(slots, positions))
            last = logits[torch.arange(len(cached), device=device), lengths - 1]
            for row, handle in enumerate(cached):
                handle._position = len(prompts[row])
                self._sample_and_emit(handle, last[row])
        for handle in handles:
            if not handle.params.use_cache:
                self._uncached_step(handle)

    def _decode(self, handles: list[GenerationHandle]) -> None:
        ctx = self.engine.context_size
        cached = [h for h in handles if h.params.use_cache]
        if cached:
            device = self.cache.k.device
            for handle in cached:
                if handle._position >= ctx:
                    self._slide_window(handle)
            tokens = torch.tensor([[h._ids[-1]] for h in cached], device=device)
            positions = torch.tensor([[h._position] for h in cached], device=device)
            slots = torch.tensor([h._slot for h in cached], device=device)
            logits = self.engine.model(tokens, slot_batch=self.cache.batch(slots, positions))[:, 0]
            for row, handle in enumerate(cached):
                handle._position += 1
                self._sample_and_emit(handle, logits[row])
        for handle in handles:
            if not handle.params.use_cache:
                self._uncached_step(handle)

    def _slide_window(self, handle: GenerationHandle) -> None:
        """The slot is full: re-encode the trailing ctx - 1 tokens at fresh
        positions 0..ctx-2 (absolute position embeddings can't be shifted),
        exactly as model.gpt.generate_stream does past the context length."""
        ctx = self.engine.context_size
        device = self.cache.k.device
        window = torch.tensor([handle._ids[-ctx:-1]], device=device)
        positions = torch.arange(window.shape[1], device=device)[None]
        self.engine.model(window, slot_batch=self.cache.batch(torch.tensor([handle._slot], device=device), positions))
        handle._position = window.shape[1]

    def _uncached_step(self, handle: GenerationHandle) -> None:
        """use_cache=False: recompute the whole window every token (kept so
        the UI's KV-cache toggle still demonstrates the difference)."""
        device = next(self.engine.model.parameters()).device
        window = torch.tensor([handle._ids[-self.engine.context_size:]], device=device)
        self._sample_and_emit(handle, self.engine.model(window)[0, -1])

    def _sample_and_emit(self, handle: GenerationHandle, logits_row: torch.Tensor) -> None:
        if handle.cancelled:
            self._finish(handle, "cancelled")
            return
        p = handle.params
        logits = logits_row.float().unsqueeze(0).clone()
        eos = self.engine.eos_id
        if len(handle.generated) < p.min_new_tokens:
            logits[:, eos] = float("-inf")
        history = torch.tensor([handle._ids], device=logits.device)
        token = _sample_next_token(
            logits, p.temperature, p.top_k, p.top_p, p.repetition_penalty, p.frequency_penalty,
            p.presence_penalty, p.no_repeat_ngram_size, history, len(handle.prompt_ids),
        ).item()

        if token == eos:
            self._finish(handle, "stop")
            return
        handle._ids.append(token)
        handle.generated.append(token)
        handle._events.put(("token", token))
        if len(handle.generated) >= p.max_new_tokens:
            self._finish(handle, "length")

    def _finish(self, handle: GenerationHandle, reason: str, error: Exception | None = None) -> None:
        if handle.finish_reason is not None:
            return
        handle.finish_reason = reason
        handle.finished_at = time.perf_counter()
        handle._admitted.set()  # never leave a waiter hanging
        if handle in self._active:
            self._active.remove(handle)
        if handle._slot is not None:
            with self._cv:
                self._free_slots.append(handle._slot)
            handle._slot = None
        handle._events.put(("error", error) if error is not None else ("done", reason))
