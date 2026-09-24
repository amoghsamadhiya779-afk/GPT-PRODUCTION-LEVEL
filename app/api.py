# app/api.py
"""FastAPI REST API service for GPT-2 model serving.

Exposes endpoints for text generation, system health checks, and
training loss plot visualization. Supports lifespan checkpoint preloading.

Concurrency model: there is one model instance, so every operation that runs
or mutates it (generation, applying an adapter, snapshotting weights for
fine-tuning) holds the `EngineGate`. The gate has a bounded wait queue so a
burst of requests is turned away with a fast 503 instead of tying up every
server thread waiting for the model.
"""

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Sequence
from contextlib import asynccontextmanager

import tiktoken
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Add root folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import adapters as adapter_store
from app.adapters import BASE_MODEL, AdapterError, AdapterNotFound
from app.finetune import run_lora_finetune_job
from app.inference import GPTInferenceEngine
from app.schemas import GenerationRequest, GenerationResponse, FinetuneRequest, FinetuneStatus, FeedbackRequest
from app.security import SecurityMiddleware, client_ip, require_admin
# The instruction template shared with SFT training, so serving prompts are
# byte-for-byte what the adapters were trained on.
from data.sft import format_prompt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# System stdout UTF-8 compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


ENGINE_MAX_QUEUE = _env_int("ENGINE_MAX_QUEUE", 8)
ENGINE_QUEUE_TIMEOUT = float(_env_int("ENGINE_QUEUE_TIMEOUT", 30))
MAX_TEACH_ADAPTERS = _env_int("MAX_TEACH_ADAPTERS", 50)
MAX_FINETUNE_JOBS_KEPT = 50
FEEDBACK_FILE = os.path.join("data", "feedback.jsonl")
FEEDBACK_MAX_BYTES = _env_int("FEEDBACK_MAX_BYTES", 20 * 1024 * 1024)

# Randomly initialized stand-in that keeps the port open while real weights
# download in the background. It is never used to serve generations.
PLACEHOLDER_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 64,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "model_size": "tiny",
}


class EngineGate:
    """Exclusive access to the model with a bounded wait queue.

    Acquire and release may happen on different threads: a streaming request
    acquires in the request handler, then hands ownership to the generation
    thread, which releases only when generation has really stopped.
    """

    def __init__(self, max_waiting: int) -> None:
        self._sem = threading.BoundedSemaphore(1)
        self._mutex = threading.Lock()
        self._waiting = 0
        self.max_waiting = max_waiting

    def acquire(self, timeout: float) -> bool:
        if self._sem.acquire(blocking=False):
            return True
        with self._mutex:
            if self._waiting >= self.max_waiting:
                return False
            self._waiting += 1
        try:
            return self._sem.acquire(timeout=timeout)
        finally:
            with self._mutex:
                self._waiting -= 1

    def release(self) -> None:
        self._sem.release()


def _load_default_adapter(engine):
    """Resolve the DEFAULT_ADAPTER env var against a freshly loaded engine."""
    name = os.environ.get("DEFAULT_ADAPTER", "").strip()
    if not name or name.lower() == BASE_MODEL:
        return None
    try:
        return engine.adapters.get(name)
    except AdapterError as e:
        logger.warning("DEFAULT_ADAPTER %r not activated: %s", name, e)
        return None


def _install_engine(application: FastAPI, engine, checkpoint_path: str) -> None:
    """Make a fully loaded engine live. Shared by both startup paths so the
    DEFAULT_ADAPTER hook can't be skipped depending on which one runs."""
    default = _load_default_adapter(engine)
    engine.set_adapter(default)  # pre-apply so the first request doesn't pay for it
    application.state.default_adapter = default.name if default else None
    application.state.engine = engine
    application.state.checkpoint_path = checkpoint_path
    application.state.parameter_count = engine.parameter_count
    application.state.device = str(engine.device)
    application.state.status = "active"  # last: only now may requests use it
    logger.info("Model is live (%d parameters, default adapter: %s)",
                engine.parameter_count, application.state.default_adapter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for FastAPI application startup and shutdown lifecycle events."""
    # 1. Determine checkpoint path
    checkpoint_path = os.environ.get("MODEL_CHECKPOINT")

    if not checkpoint_path:
        tiny_path = os.path.join("checkpoints_tiny", "best_model.pt")
        small_path = os.path.join("checkpoints", "best_model.pt")
        if os.path.exists(small_path):
            checkpoint_path = small_path
        elif os.path.exists(tiny_path):
            checkpoint_path = tiny_path

    # Initialize state variables
    app.state.status = "initializing"
    app.state.checkpoint_path = "None"
    app.state.parameter_count = 0
    app.state.device = "cpu"
    app.state.start_time = time.time()
    app.state.metrics_lock = threading.Lock()
    app.state.total_requests = 0
    app.state.total_tokens_generated = 0
    app.state.total_time_taken = 0.0
    app.state.engine_gate = EngineGate(max_waiting=ENGINE_MAX_QUEUE)
    app.state.default_adapter = None
    app.state.finetune_jobs = {}
    app.state.finetune_lock = threading.Lock()
    app.state.feedback_lock = threading.Lock()

    def load_weights_background(application):
        try:
            logger.info("Starting background download/mapping of pretrained weights...")
            from training.load_pretrained import main as load_weights
            load_weights()

            small_path = os.path.join("checkpoints", "best_model.pt")
            if os.path.exists(small_path):
                logger.info("Loading background-downloaded model checkpoint from: %s", small_path)
                _install_engine(application, GPTInferenceEngine(small_path), small_path)
        except Exception:
            logger.exception("Background weight load failed")
            if application.state.status != "active":
                application.state.status = "error"

    # 2. Try loading the model engine
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            logger.info("Loading model checkpoint from: %s", checkpoint_path)
            _install_engine(app, GPTInferenceEngine(checkpoint_path), checkpoint_path)
        except Exception:
            logger.exception("Failed to load checkpoint")
            app.state.status = "error"
    else:
        # Checkpoint is missing! Boot a placeholder model first to keep the port open.
        logger.warning("No checkpoint found. Initializing placeholder model and starting background download.")
        try:
            app.state.engine = GPTInferenceEngine.from_config(PLACEHOLDER_CONFIG, device="cpu")
            app.state.status = "loading"  # real weights are loading in the background
            app.state.checkpoint_path = "None (Downloading Real Weights...)"
            app.state.parameter_count = app.state.engine.parameter_count
            app.state.device = "cpu"

            t = threading.Thread(target=load_weights_background, args=(app,), daemon=True)
            t.start()
        except Exception:
            logger.critical("Failed to build fallback model", exc_info=True)
            app.state.status = "failed"

    yield
    logger.info("Shutting down model serving API.")


# Instantiate FastAPI
app = FastAPI(
    title="GPT-2 Serving API",
    description="REST API for serving a custom GPT-2 model trained from scratch.",
    version="1.1.0",
    lifespan=lifespan,
)

# Rate limiting, keyed on the real client (see app.security.client_ip).
limiter = Limiter(key_func=client_ip, enabled=(os.environ.get("TESTING") != "1"))
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Body-size cap + security headers. Added before CORS so CORS stays the
# outermost layer and even a 413 carries CORS headers the browser can read.
app.add_middleware(SecurityMiddleware)

allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,https://gpt-production-level.vercel.app")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

# No cookies or HTTP auth are used cross-origin (admin calls use an explicit
# bearer token from server-side tooling), so credentials stay disabled -- with
# them on, any misconfigured origin could make authenticated requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

# Auth model: end-user features (chat, Teach Mode, choosing an adapter for your
# own requests, submitting feedback) are public and rate-limited. Anything
# that changes behavior for *other* users or exposes their data -- setting
# the server-wide default adapter, reading collected feedback -- requires
# ADMIN_API_KEY (see app.security.require_admin).


def _record_metrics(tokens: int, latency: float) -> None:
    with app.state.metrics_lock:
        app.state.total_tokens_generated += tokens
        app.state.total_time_taken += latency


def _count_request() -> None:
    with app.state.metrics_lock:
        app.state.total_requests += 1


@app.get("/health")
def health_check():
    """Retrieve service health status, hardware device, and active checkpoint metadata."""
    current_status = getattr(app.state, "status", "offline")
    status_code = status.HTTP_200_OK
    if current_status in ["error", "failed"]:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    uptime = 0.0
    if hasattr(app.state, "start_time"):
        uptime = time.time() - app.state.start_time

    total_reqs = getattr(app.state, "total_requests", 0)
    total_tokens = getattr(app.state, "total_tokens_generated", 0)
    total_time = getattr(app.state, "total_time_taken", 0.0)

    avg_tokens_per_second = 0.0
    if total_time > 0:
        avg_tokens_per_second = total_tokens / total_time

    engine = getattr(app.state, "engine", None)
    cfg = getattr(engine, "model_config", None) or {}

    return JSONResponse(
        status_code=status_code,
        content={
            "status": current_status,
            "checkpoint": getattr(app.state, "checkpoint_path", "unknown"),
            "parameters": getattr(app.state, "parameter_count", 0),
            "device": getattr(app.state, "device", "unknown"),
            # Raw exception text can contain file paths and internals; the
            # details are in the server log, never in a public response.
            "error_details": "Model failed to load; see server logs." if status_code != 200 else None,
            "uptime_seconds": uptime,
            "total_requests": total_reqs,
            "avg_tokens_per_second": avg_tokens_per_second,
            "model_size": getattr(engine, "model_size", "small") if engine is not None else "unknown",
            "layers": cfg.get("n_layers") if cfg else None,
            "heads": cfg.get("n_heads") if cfg else None,
            "emb_dim": cfg.get("emb_dim") if cfg else None,
            "context": cfg.get("context_length") if cfg else None,
            "default_adapter": getattr(app.state, "default_adapter", None),
        },
    )


STOP_WORDS = {"a", "an", "the", "and", "but", "if", "or", "because", "as", "what", "which", "this", "that", "these", "those", "then", "just", "so", "than", "such", "both", "through", "about", "for", "is", "of", "while", "during", "to", "in", "it", "on", "with"}

def check_grounding_safety(prompt: str, answer: str, sources: list) -> str:
    if not sources:
        return ""

    def get_words(text):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return set(w for w in words if w not in STOP_WORDS)

    gen_words = get_words(answer)
    source_words = set()
    for s in sources:
        source_words.update(get_words(s['snippet']))

    overlap = len(gen_words.intersection(source_words))
    overlap_ratio = overlap / len(gen_words) if gen_words else 0.0
    # Trigger when the answer barely echoes the sources -- either in absolute
    # terms (very few shared substantive words, catches short answers that
    # are entirely off-topic) or proportionally (mostly made up of words that
    # appear nowhere in the sources, catches longer answers that drift).
    if overlap < 3 or (len(gen_words) >= 6 and overlap_ratio < 0.2):
        best_sentence = ""
        max_overlap = -1
        prompt_words = get_words(prompt)

        for s in sources:
            sentences = re.split(r'(?<=[.!?]) +', s['snippet'])
            for sent in sentences:
                sent_words = get_words(sent)
                o = len(prompt_words.intersection(sent_words))
                if o > max_overlap:
                    max_overlap = o
                    best_sentence = sent.strip()

        if best_sentence:
            return f"From the sources: {best_sentence}.\n\n"
    return ""


def history_pairs(turns) -> list[tuple[str, str]]:
    """Pair each user turn with the assistant reply that directly follows it.

    Unanswered user turns (e.g. a failed generation) and assistant turns with
    no preceding question (e.g. a persona's welcome message) are dropped, so
    the prompt always alternates Instruction/Response.
    """
    pairs, question = [], None
    for turn in turns:
        text = turn.content.strip()
        if turn.role == "user":
            question = text or None
        elif question is not None and text:
            pairs.append((question, text))
            question = None
    return pairs


def build_prompt_with_budget(
    original_prompt: str,
    max_new_tokens: int,
    sources: list | None,
    context_size: int,
    history: Sequence[tuple[str, str]] = (),
) -> tuple[str, list | None]:
    """Build the model prompt so prompt + completion fit the context window.

    Priority when space runs out: the current prompt, then its web sources,
    then earlier conversation turns (most recent first; the oldest are
    dropped). User text and web snippets are counted with encode_ordinary,
    exactly as the engine will encode them (special-token strings stay text).
    """
    enc = tiktoken.get_encoding("gpt2")

    def n_tokens(text: str) -> int:
        return len(enc.encode_ordinary(text))

    # Reserve room for the completion, but always keep at least half the
    # window for the prompt (generation past the window slides it forward).
    budget = max(context_size - max_new_tokens, context_size // 2)

    # A prompt longer than the budget used to be cropped from the *left* by
    # generate(), silently cutting off the instruction header so the model no
    # longer saw the template it was tuned on. Keep the template intact and
    # drop the oldest part of the user's text instead.
    user_budget = max(budget - n_tokens(format_prompt("")), 1)
    prompt_ids = enc.encode_ordinary(original_prompt)
    while len(prompt_ids) > user_budget:
        original_prompt = enc.decode(prompt_ids[-user_budget:]).lstrip("\ufffd")
        # BPE merges across the template boundary can shift the count by a
        # token or two, so re-check the assembled prompt.
        excess = n_tokens(format_prompt(original_prompt)) - budget
        if excess <= 0 or user_budget == 1:
            break
        user_budget = max(user_budget - excess, 1)

    instruction = original_prompt
    if sources:
        valid_sources = sources.copy()
        while valid_sources:
            context_str = ""
            for i, res in enumerate(valid_sources, 1):
                context_str += f"[{i}] {res['snippet']}\n"
            grounded = f"{context_str}\nQuestion: {original_prompt}"

            prompt_tokens = n_tokens(format_prompt(grounded))
            if prompt_tokens <= budget:
                instruction = grounded
                break

            excess = prompt_tokens - budget

            last_src = valid_sources[-1]
            raw_tokens = enc.encode_ordinary(last_src['snippet'])

            trim_len = len(raw_tokens) - excess - 2
            if trim_len <= 0:
                valid_sources.pop()
            else:
                new_src = last_src.copy()
                new_src['snippet'] = enc.decode(raw_tokens[:trim_len]).strip() + "..."
                valid_sources[-1] = new_src
        sources = valid_sources

    # Earlier turns get whatever room is left, newest first.
    kept: list[tuple[str, str]] = []
    for turn in reversed(history):
        if n_tokens(format_prompt(instruction, [turn] + kept)) > budget:
            break
        kept.insert(0, turn)

    return format_prompt(instruction, kept), sources


def _require_engine():
    current_status = getattr(app.state, "status", None)
    if current_status != "active" or not hasattr(app.state, "engine"):
        if current_status in ("loading", "initializing"):
            detail = "Model is still warming up (downloading/loading weights). Please retry in a few seconds."
        else:
            detail = "Model serving engine is offline or failed to initialize."
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return app.state.engine


def _resolve_adapter(engine, requested: str | None):
    """Map GenerationRequest.adapter to a validated Adapter (None = base model)."""
    name = requested if requested is not None else app.state.default_adapter
    if name is None or name.lower() == BASE_MODEL:
        return None
    try:
        return engine.adapters.get(name)
    except AdapterError as e:
        if requested is None:
            # The server default vanished or broke after startup: degrade to
            # the base model rather than failing every request.
            logger.warning("Default adapter %r unavailable, using base model: %s", name, e)
            return None
        code = status.HTTP_404_NOT_FOUND if isinstance(e, AdapterNotFound) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=str(e))


def _web_sources(body: GenerationRequest) -> list | None:
    if not body.web_search:
        return None
    from app.search import web_search
    return web_search(body.prompt, max_results=3)


def _sampling_kwargs(body: GenerationRequest) -> dict:
    return {
        "max_new_tokens": body.max_new_tokens,
        "temperature": body.temperature,
        "top_k": body.top_k,
        "top_p": body.top_p,
        "repetition_penalty": body.repetition_penalty,
        "frequency_penalty": body.frequency_penalty,
        "presence_penalty": body.presence_penalty,
        "no_repeat_ngram_size": body.no_repeat_ngram_size,
        "min_new_tokens": body.min_new_tokens,
        "use_cache": body.use_cache,
    }


def _busy_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Server is currently busy generating other requests. Please try again shortly."},
        headers={"Retry-After": "5"},
    )


@app.post("/generate", response_model=GenerationResponse)
@limiter.limit("10/minute")
def generate_text(request: Request, body: GenerationRequest):
    """Generate text from a prompt using the loaded GPT model."""
    engine = _require_engine()
    req_id = str(uuid.uuid4())
    adapter = _resolve_adapter(engine, body.adapter)

    sources = _web_sources(body)
    prompt_text, sources = build_prompt_with_budget(
        body.prompt, body.max_new_tokens, sources, engine.context_size, history_pairs(body.history)
    )

    gate = app.state.engine_gate
    if not gate.acquire(timeout=ENGINE_QUEUE_TIMEOUT):
        return _busy_response()
    try:
        _count_request()
        engine.set_adapter(adapter)
        result = engine.generate(prompt=prompt_text, **_sampling_kwargs(body))
    except Exception:
        logger.exception("Generation error [req_id=%s]", req_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed (req_id={req_id}). Check server logs for details.",
        )
    finally:
        gate.release()

    answer = result["completion_text"].strip()
    if sources:
        answer = check_grounding_safety(body.prompt, answer, sources) + answer

    tokens_gen = result["tokens_generated"]
    latency = result["time_taken_seconds"]
    _record_metrics(tokens_gen, latency)
    logger.info(
        "Generate | req_id=%s | prompt_len=%d | tokens=%d | latency_ms=%.1f | speed=%.1f t/s | adapter=%s",
        req_id, len(body.prompt), tokens_gen, latency * 1000, result["tokens_per_second"],
        adapter.name if adapter else None,
    )

    return {
        "prompt": body.prompt,
        "generated_text": body.prompt + "\n\n" + answer,
        "tokens_generated": tokens_gen,
        "time_taken_seconds": latency,
        "tokens_per_second": result["tokens_per_second"],
        "sources": sources,
        "adapter": adapter.name if adapter else None,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/generate/stream")
@limiter.limit("10/minute")
async def generate_text_stream(request: Request, body: GenerationRequest):
    """Stream text generation from a prompt using the loaded GPT model."""
    engine = _require_engine()

    # Defense in depth: refuse to start an SSE stream against an engine object
    # that doesn't implement streaming, with a clean 503 before any response.
    if not hasattr(engine, "generate_stream"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming is not available on the current engine.",
        )

    req_id = str(uuid.uuid4())
    # Adapter loading (disk I/O) and web search (network, up to ~20s) are
    # blocking calls: run them in worker threads. Calling them directly here
    # would freeze the event loop -- and with it every other request and the
    # health check -- for their full duration.
    adapter = await asyncio.to_thread(_resolve_adapter, engine, body.adapter)
    sources = await asyncio.to_thread(_web_sources, body)
    prompt_text, sources = build_prompt_with_budget(
        body.prompt, body.max_new_tokens, sources, engine.context_size, history_pairs(body.history)
    )

    gate = app.state.engine_gate
    if not await asyncio.to_thread(gate.acquire, ENGINE_QUEUE_TIMEOUT):
        return _busy_response()

    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def emit(item: tuple) -> None:
        try:
            loop.call_soon_threadsafe(events.put_nowait, item)
        except RuntimeError:  # event loop closed (server shutting down)
            stop_event.set()

    def run_generation() -> None:
        # This thread owns the engine gate from here on and releases it only
        # once generation has actually stopped. Previously the response
        # generator released it, which (a) leaked it forever if the client
        # disconnected before the generator's try/finally was entered -- one
        # such request made every later request 503 -- and (b) could release
        # it while this thread was still running the model.
        try:
            engine.set_adapter(adapter)
            latency, tokens_gen = 0.0, 0
            for text_chunk, latency, tokens_gen in engine.generate_stream(prompt=prompt_text, **_sampling_kwargs(body)):
                if stop_event.is_set():
                    return
                if text_chunk:
                    emit(("chunk", text_chunk, latency, tokens_gen))
            emit(("done", None, latency, tokens_gen))
        except Exception:
            logger.exception("Streaming generation error [req_id=%s]", req_id)
            emit(("error", None, 0.0, 0))
        finally:
            gate.release()

    try:
        _count_request()
        threading.Thread(target=run_generation, name=f"generate-{req_id[:8]}", daemon=True).start()
    except BaseException:
        gate.release()
        raise

    async def event_generator():
        # Every yield sits inside this try, so however the response ends
        # (normal completion, client disconnect, cancellation, GC of an
        # unfinished generator) the worker is told to stop.
        accumulated = []
        try:
            if sources:
                yield _sse({"sources": sources})

            while True:
                try:
                    kind, text_chunk, latency, tokens_gen = await asyncio.wait_for(events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        logger.info("Client disconnected during stream [req_id=%s]", req_id)
                        return
                    continue

                if kind == "chunk":
                    accumulated.append(text_chunk)
                    yield _sse({"token": text_chunk, "index": tokens_gen})
                elif kind == "done":
                    speed = tokens_gen / latency if latency > 0 else 0.0
                    _record_metrics(tokens_gen, latency)
                    logger.info(
                        "Stream Generate | req_id=%s | prompt_len=%d | tokens=%d | latency_ms=%.1f | speed=%.1f t/s | adapter=%s",
                        req_id, len(body.prompt), tokens_gen, latency * 1000, speed,
                        adapter.name if adapter else None,
                    )
                    final_data = {
                        "done": True,
                        "tokens_generated": tokens_gen,
                        "time_taken_seconds": latency,
                        "tokens_per_second": speed,
                        "adapter": adapter.name if adapter else None,
                    }
                    if sources is not None:
                        final_data["sources"] = sources
                        safety_prefix = check_grounding_safety(body.prompt, "".join(accumulated).strip(), sources)
                        if safety_prefix:
                            final_data["safety_net_prefix"] = safety_prefix
                    yield _sse(final_data)
                    return
                else:
                    yield _sse({"error": f"Generation failed (req_id={req_id}). Please try again."})
                    return
        finally:
            stop_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/training/plot")
def get_training_plot():
    """Fetch the latest training loss curve plot image."""
    paths_to_check = [
        os.path.join("logs_tiny", "loss.png"),
        os.path.join("logs", "loss.png"),
        "loss.png",
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            return FileResponse(path, media_type="image/png")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Training loss plot image not found.",
    )


@app.post("/finetune")
@limiter.limit("3/minute")
def start_finetuning(request: Request, body: FinetuneRequest):
    """Start a background LoRA finetuning job (Teach Mode)."""
    engine = _require_engine()
    name = body.adapter_name

    # Teach Mode creates new adapters only. Previously any visitor could pick
    # the name of an existing adapter -- including the shipped sft_v1_*
    # instruction-tuning adapters that DEFAULT_ADAPTER loads at startup -- and
    # silently overwrite it with whatever they trained.
    if adapter_store.is_protected(name):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="That adapter name is reserved.")
    if os.path.exists(adapter_store.adapter_path(name)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An adapter with that name already exists.")
    if len(adapter_store.list_adapters()) >= MAX_TEACH_ADAPTERS:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Adapter storage limit reached.")

    with app.state.finetune_lock:
        jobs = app.state.finetune_jobs
        active_jobs = [k for k, v in jobs.items() if v.get("status") in ("running", "queued")]
        if active_jobs:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A finetuning job is already running: {active_jobs[0]}"
            )

        # Keep job bookkeeping bounded; oldest finished jobs are dropped first.
        for old_id in list(jobs)[: max(0, len(jobs) - MAX_FINETUNE_JOBS_KEPT + 1)]:
            del jobs[old_id]

        job_id = f"job-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        job_state = {
            "id": job_id,
            "status": "queued",
            "step": 0,
            "total_steps": body.steps,
            "current_loss": None,
            "eta_seconds": None,
            "error": None,
        }
        jobs[job_id] = job_state

    thread = threading.Thread(
        target=run_lora_finetune_job,
        args=(job_state, body, engine, app.state.finetune_lock, app.state.engine_gate),
        daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/finetune/{job_id}", response_model=FinetuneStatus)
def get_finetuning_status(job_id: str):
    """Get the status of a finetuning job."""
    with app.state.finetune_lock:
        if job_id not in app.state.finetune_jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        state = app.state.finetune_jobs[job_id]
        return FinetuneStatus(**state)


@app.get("/adapters")
def list_adapters():
    """List available LoRA adapters and the server's default."""
    return {
        "adapters": adapter_store.list_adapters(),
        "default": getattr(app.state, "default_adapter", None),
    }


@app.post("/adapters/{name}/activate", dependencies=[Depends(require_admin)])
@limiter.limit("15/minute")
def activate_adapter(request: Request, name: str):
    """Admin: set the server-wide default adapter (used when a request doesn't pick one).

    Clients choose an adapter for their own requests with
    GenerationRequest.adapter; this endpoint changes it for everyone.
    """
    engine = _require_engine()
    try:
        adapter_store.validate_name(name)
        adapter = engine.adapters.get(name)
    except AdapterNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AdapterError as e:
        raise HTTPException(status_code=400, detail=str(e))
    app.state.default_adapter = adapter.name
    logger.info("Server default adapter set to %s", adapter.name)
    return {"status": "success", "adapter": adapter.name}


@app.post("/adapters/deactivate", dependencies=[Depends(require_admin)])
@limiter.limit("15/minute")
def deactivate_adapter(request: Request):
    """Admin: make the plain base model the server-wide default."""
    app.state.default_adapter = None
    logger.info("Server default adapter cleared (base model)")
    return {"status": "success", "adapter": None}


@app.post("/feedback")
@limiter.limit("20/minute")
def submit_feedback(request: Request, body: FeedbackRequest):
    """Save feedback to data/feedback.jsonl."""
    entry = {
        "timestamp": time.time(),
        "prompt": body.prompt,
        "response": body.response,
        "rating": body.rating,
        "correction": body.correction or None,
    }

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with app.state.feedback_lock:
        if os.path.exists(FEEDBACK_FILE) and os.path.getsize(FEEDBACK_FILE) >= FEEDBACK_MAX_BYTES:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feedback storage is full; please try again later.",
            )
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    return {"status": "success"}


@app.get("/feedback", dependencies=[Depends(require_admin)])
def get_feedback(limit: int = 1000):
    """Admin: most recent stored feedback entries.

    Feedback contains other users' prompts and answers, so it is not public.
    (Teach Mode loads a user's own corrections from their browser instead.)
    """
    limit = max(1, min(limit, 10000))
    if not os.path.exists(FEEDBACK_FILE):
        return {"feedback": []}

    recent = deque(maxlen=limit)
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    recent.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return {"feedback": list(recent)}


@app.get("/starter-dataset")
def get_starter_dataset():
    """Get the starter instructions dataset."""
    dataset_file = os.path.join("data", "starter_instructions.jsonl")
    if not os.path.exists(dataset_file):
        raise HTTPException(status_code=404, detail="Starter dataset not found")

    results = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return {"dataset": results}
