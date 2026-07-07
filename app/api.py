# app/api.py
"""FastAPI REST API service for GPT-2 model serving.

Exposes endpoints for text generation, system health checks, and
training loss plot visualization. Supports lifespan checkpoint preloading.
"""

import logging
import os
import sys
import time
import json
import asyncio
import uuid
import queue
import threading
import tiktoken
import queue
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add root folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import GenerationRequest, GenerationResponse, FinetuneRequest, FinetuneStatus, FeedbackRequest
from app.finetune import run_lora_finetune_job
from app.inference import GPTInferenceEngine
from model.gpt import GPTModel, count_parameters
from model.tokenizer import GPT2Tokenizer

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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for FastAPI application startup and shutdown lifecycle events."""
    import threading
    
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
    app.state.total_requests = 0
    app.state.total_tokens_generated = 0
    app.state.total_time_taken = 0.0
    app.state.engine_semaphore = threading.Semaphore(1)
    app.state.finetune_jobs = {}
    app.state.finetune_lock = threading.Lock()

    # Define helper for async weight loading
    def load_weights_background(application):
        try:
            logger.info("Starting background download/mapping of pretrained weights...")
            from training.load_pretrained import main as load_weights
            load_weights()
            
            # Recheck and load
            small_path = os.path.join("checkpoints", "best_model.pt")
            if os.path.exists(small_path):
                logger.info("Loading background-downloaded model checkpoint from: %s", small_path)
                engine = GPTInferenceEngine(small_path)
                application.state.engine = engine
                application.state.status = "active"
                application.state.checkpoint_path = small_path
                application.state.parameter_count = engine.parameter_count
                application.state.device = str(engine.device)
                logger.info("Model loaded successfully in background thread.")
        except Exception as err:
            logger.error("Background weight load failed: %s", err)
            if not hasattr(application.state, "engine"):
                application.state.status = "error"
                application.state.error_msg = str(err)

    # 2. Try loading the model engine
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            logger.info("Loading model checkpoint from: %s", checkpoint_path)
            engine = GPTInferenceEngine(checkpoint_path)
            app.state.engine = engine
            app.state.status = "active"
            app.state.checkpoint_path = checkpoint_path
            app.state.parameter_count = engine.parameter_count
            app.state.device = str(engine.device)
            logger.info("Model loaded successfully (%d parameters)", engine.parameter_count)
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            app.state.status = "error"
            app.state.error_msg = str(e)
    else:
        # Checkpoint is missing! Boot fallback dummy model first to keep port open.
        logger.warning("No checkpoint found. Initializing untrained dummy model and starting background download.")
        try:
            dummy_cfg = {
                "vocab_size": 50257,
                "context_length": 256,
                "emb_dim": 64,
                "n_heads": 2,
                "n_layers": 1,
                "drop_rate": 0.0,
                "qkv_bias": False,
            }
            # Mock an inference engine manually
            class DummyEngine:
                def __init__(self, cfg):
                    self.device = torch.device("cpu")
                    self.model = GPTModel(cfg).eval()
                    self.tokenizer = GPT2Tokenizer()
                    self.context_size = 256
                    self.parameter_count = count_parameters(self.model)
                
                def generate(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50, top_p=None, repetition_penalty=1.0, use_cache=True):
                    input_ids = self.tokenizer.text_to_token_ids(prompt)
                    import time
                    start = time.perf_counter()
                    from model.gpt import generate as gpt_gen
                    output_ids = gpt_gen(
                        self.model, input_ids, max_new_tokens, self.context_size,
                        temperature, top_k, top_p, repetition_penalty, use_cache=use_cache
                    )
                    latency = time.perf_counter() - start
                    generated_text = self.tokenizer.token_ids_to_text(output_ids)
                    num_gen = output_ids.shape[1] - input_ids.shape[1]
                    return {
                        "prompt": prompt,
                        "generated_text": generated_text + " [Initializing Model... Please wait a few seconds]",
                        "tokens_generated": num_gen,
                        "time_taken_seconds": latency,
                        "tokens_per_second": num_gen / latency if latency > 0 else 0.0
                    }
            
            app.state.engine = DummyEngine(dummy_cfg)
            app.state.status = "loading"  # indicate it's loading the real weights in background
            app.state.checkpoint_path = "None (Downloading Real Weights...)"
            app.state.parameter_count = app.state.engine.parameter_count
            app.state.device = "cpu"
            
            # Start background thread to download and swap weights
            t = threading.Thread(target=load_weights_background, args=(app,), daemon=True)
            t.start()
        except Exception as e:
            logger.critical("Failed to build fallback model: %s", e)
            app.state.status = "failed"
            app.state.error_msg = str(e)
            
    yield
    # Cleanup on shutdown (if any)
    logger.info("Shutting down model serving API.")


# Instantiate FastAPI
app = FastAPI(
    title="GPT-2 Serving API",
    description="REST API for serving a custom GPT-2 model trained from scratch.",
    version="1.0.0",
    lifespan=lifespan,
)

# Setup Rate Limiting
limiter = Limiter(key_func=get_remote_address, enabled=(os.environ.get("TESTING") != "1"))
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,https://gpt-production-level.vercel.app")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Retrieve service health status, hardware device, and active checkpoint metadata."""
    status_code = status.HTTP_200_OK
    if getattr(app.state, "status", None) in ["error", "failed"]:
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

    return JSONResponse(
        status_code=status_code,
        content={
            "status": getattr(app.state, "status", "offline"),
            "checkpoint": getattr(app.state, "checkpoint_path", "unknown"),
            "parameters": getattr(app.state, "parameter_count", 0),
            "device": getattr(app.state, "device", "unknown"),
            "error_details": getattr(app.state, "error_msg", None),
            "uptime_seconds": uptime,
            "total_requests": total_reqs,
            "avg_tokens_per_second": avg_tokens_per_second,
        },
    )


def build_prompt_with_budget(original_prompt: str, max_new_tokens: int, sources: list, context_size: int) -> tuple[str, list]:
    enc = tiktoken.get_encoding("gpt2")
    base_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{original_prompt}\n\n"
        "### Response:\n"
    )
    
    if not sources:
        return base_prompt, sources

    template_start = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{original_prompt}\n\nWeb Search Context:\n"
    )
    template_end = "\n\n### Response:\n"
    
    fixed_tokens = len(enc.encode(template_start)) + len(enc.encode(template_end))
    available_tokens = context_size - max_new_tokens - fixed_tokens
    
    if available_tokens <= 0:
        return base_prompt, []

    valid_sources = []
    current_context_tokens = 0
    
    for src in sources:
        snippet_text = f"- {src['snippet']}"
        snippet_tokens = enc.encode(snippet_text)
        
        if current_context_tokens + len(snippet_tokens) + 1 <= available_tokens:
            valid_sources.append(src)
            current_context_tokens += len(snippet_tokens) + 1
        elif available_tokens - current_context_tokens > 20:
            allowed_tokens = available_tokens - current_context_tokens - 1
            trimmed_snippet = enc.decode(snippet_tokens[:allowed_tokens])
            trimmed_src = src.copy()
            trimmed_src['snippet'] = trimmed_snippet.strip() + "..."
            valid_sources.append(trimmed_src)
            break
        else:
            break
            
    if not valid_sources:
        return base_prompt, []
        
    context_str = "\n".join([f"- {res['snippet']}" for res in valid_sources])
    prompt_text = template_start + context_str + template_end
    
    return prompt_text, valid_sources


@app.post("/generate", response_model=GenerationResponse)
@limiter.limit("10/minute")
def generate_text(request: Request, body: GenerationRequest):
    """Generate text from a prompt using the loaded GPT model."""
    if getattr(app.state, "status", None) == "failed" or not hasattr(app.state, "engine"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model serving engine is offline or failed to initialize.",
        )

    req_id = str(uuid.uuid4())
    original_prompt = body.prompt
    sources = None
    
    if body.web_search:
        from app.search import web_search
        sources = web_search(original_prompt, max_results=3)
        
    context_size = getattr(app.state.engine, "context_size", 256)
    prompt_text, sources = build_prompt_with_budget(original_prompt, body.max_new_tokens, sources, context_size)
    
    acquired = app.state.engine_semaphore.acquire(timeout=30.0)
    if not acquired:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Server is currently busy generating another request. Please try again later."}
        )

    try:
        app.state.total_requests += 1

        response_data = app.state.engine.generate(
            prompt=prompt_text,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
            use_cache=body.use_cache,
        )
        
        # Unify response cleaning for instruction template
        gen_text = response_data["generated_text"]
        if "### Response:" in gen_text:
            answer = gen_text.split("### Response:")[-1].strip()
            # Remove EOS token if generated
            answer = answer.replace("<|endoftext|>", "").strip()
            response_data["generated_text"] = original_prompt + "\n\n" + answer
        else:
            # Fallback if tag is missing but starts with prompt_text
            if gen_text.startswith(prompt_text):
                clean_ans = gen_text[len(prompt_text):].strip().replace("<|endoftext|>", "")
                response_data["generated_text"] = original_prompt + "\n\n" + clean_ans
            else:
                response_data["generated_text"] = gen_text.replace("<|endoftext|>", "").strip()
                
        response_data["prompt"] = original_prompt
        response_data["sources"] = sources
        
        # Log metrics and update global state
        tokens_gen = response_data["tokens_generated"]
        latency = response_data["time_taken_seconds"]
        speed = response_data["tokens_per_second"]
        app.state.total_tokens_generated += tokens_gen
        app.state.total_time_taken += latency
        
        logger.info(
            "Generate | req_id=%s | prompt_len=%d | tokens=%d | latency_ms=%.1f | speed=%.1f t/s",
            req_id, len(original_prompt), tokens_gen, latency * 1000, speed
        )
        
        return response_data
    except Exception as e:
        logger.error("Generation error [req_id=%s]: %s", req_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )
    finally:
        app.state.engine_semaphore.release()


@app.post("/generate/stream")
@limiter.limit("10/minute")
async def generate_text_stream(request: Request, body: GenerationRequest):
    """Stream text generation from a prompt using the loaded GPT model."""
    if getattr(app.state, "status", None) == "failed" or not hasattr(app.state, "engine"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model serving engine is offline or failed to initialize.",
        )
        
    # Check if we are running DummyEngine (which doesn't have generate_stream)
    if not hasattr(app.state.engine, "generate_stream"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming is not available while the model is initializing.",
        )

    req_id = str(uuid.uuid4())
    
    original_prompt = body.prompt
    sources = None
    if body.web_search:
        from app.search import web_search
        sources = web_search(original_prompt, max_results=3)
        
    context_size = getattr(app.state.engine, "context_size", 256)
    prompt_text, sources = build_prompt_with_budget(original_prompt, body.max_new_tokens, sources, context_size)
    
    # Try to acquire semaphore asynchronously
    acquired = await asyncio.to_thread(app.state.engine_semaphore.acquire, timeout=30.0)
    if not acquired:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Server is currently busy generating another request. Please try again later."}
        )

    async def event_generator():
        q = queue.Queue(maxsize=100)
        stop_event = threading.Event()
        
        # Emit sources first
        if sources:
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            
        def run_generation():
            try:
                for text_chunk, latency, tokens_gen in app.state.engine.generate_stream(
                    prompt=prompt_text,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature,
                    top_k=body.top_k,
                    top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                    use_cache=body.use_cache,
                ):
                    if stop_event.is_set():
                        break
                    
                    if text_chunk == "<|endoftext|>":
                        continue
                    if "<|endoftext|>" in text_chunk:
                        text_chunk = text_chunk.replace("<|endoftext|>", "")
                        
                    q.put(("chunk", text_chunk, latency, tokens_gen))
                    
                if not stop_event.is_set():
                    q.put(("done", None, latency, tokens_gen))
            except Exception as e:
                logger.error("Streaming generation error [req_id=%s]: %s", req_id, e)
                q.put(("error", str(e), 0, 0))
            finally:
                q.put(("sentinel", None, 0, 0))

        thread = threading.Thread(target=run_generation)
        thread.start()
        
        try:
            app.state.total_requests += 1
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected during stream [req_id=%s]", req_id)
                    stop_event.set()
                    break
                    
                try:
                    item_type, text_chunk, latency, tokens_gen = await asyncio.to_thread(q.get, timeout=0.2)
                except queue.Empty:
                    continue
                    
                if item_type == "sentinel":
                    break
                    
                if item_type == "chunk":
                    if text_chunk:
                        yield f"data: {json.dumps({'token': text_chunk, 'index': tokens_gen})}\n\n"
                elif item_type == "done":
                    speed = tokens_gen / latency if latency > 0 else 0.0
                    app.state.total_tokens_generated += tokens_gen
                    app.state.total_time_taken += latency
                    
                    logger.info(
                        "Stream Generate | req_id=%s | prompt_len=%d | tokens=%d | latency_ms=%.1f | speed=%.1f t/s",
                        req_id, len(original_prompt), tokens_gen, latency * 1000, speed
                    )
                    
                    final_data = {
                        "done": True,
                        "tokens_generated": tokens_gen,
                        "time_taken_seconds": latency,
                        "tokens_per_second": speed,
                    }
                    if sources is not None:
                        final_data["sources"] = sources
                        
                    yield f"data: {json.dumps(final_data)}\n\n"
                    break
                elif item_type == "error":
                    yield f"data: {json.dumps({'error': text_chunk})}\n\n"
                    break
        finally:
            stop_event.set()
            await asyncio.to_thread(thread.join, timeout=1.0)
            app.state.engine_semaphore.release()
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/training/plot")
def get_training_plot():
    """Fetch the latest training loss curve plot image."""
    # Check potential plot locations
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
def start_finetuning(body: FinetuneRequest):
    """Start a background LoRA finetuning job."""
    if getattr(app.state, "status", None) == "failed" or not hasattr(app.state, "engine"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model serving engine is offline.",
        )
    
    with app.state.finetune_lock:
        active_jobs = [k for k, v in app.state.finetune_jobs.items() if v.get("status") in ("running", "queued")]
        if active_jobs:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A finetuning job is already running: {active_jobs[0]}"
            )
            
        job_id = f"job-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        job_state = {
            "id": job_id,
            "status": "queued",
            "step": 0,
            "total_steps": body.steps,
            "current_loss": None,
            "eta_seconds": None,
        }
        app.state.finetune_jobs[job_id] = job_state
        
    thread = threading.Thread(
        target=run_lora_finetune_job,
        args=(job_state, body, app.state.engine, app.state.finetune_lock),
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
    """List available LoRA adapters."""
    adapters_dir = os.path.join("checkpoints", "adapters")
    if not os.path.exists(adapters_dir):
        return {"adapters": []}
        
    adapters = []
    for f in os.listdir(adapters_dir):
        if f.endswith(".pt"):
            adapters.append(f[:-3])  # remove .pt
    return {"adapters": sorted(adapters)}


@app.post("/adapters/{name}/activate")
def activate_adapter(name: str):
    """Hot-swap an adapter into the running inference engine."""
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="Engine offline")
        
    adapter_path = os.path.join("checkpoints", "adapters", f"{name}.pt")
    if not os.path.exists(adapter_path):
        raise HTTPException(status_code=404, detail=f"Adapter {name} not found")
        
    acquired = app.state.engine_semaphore.acquire(timeout=30.0)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy")
        
    try:
        from model.lora import inject_lora
        
        checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
        lora_r = checkpoint.get("lora_r", 4)
        lora_alpha = checkpoint.get("lora_alpha", 8.0)
        
        model = app.state.engine.model
        # Check if already injected
        injected = False
        for mod_name, mod in model.named_modules():
            if "lora_layer" in str(type(mod)).lower() or "loralinear" in str(type(mod)).lower():
                injected = True
                break
                
        if not injected:
            inject_lora(model, r=lora_r, alpha=lora_alpha, target_modules=["W_query", "W_value"])
            
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return {"status": "success", "adapter": name}
    except Exception as e:
        logger.error(f"Failed to activate adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        app.state.engine_semaphore.release()


@app.post("/adapters/deactivate")
def deactivate_adapter():
    """Revert the inference engine to base behavior by zeroing LoRA matrices."""
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="Engine offline")
        
    acquired = app.state.engine_semaphore.acquire(timeout=30.0)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy")
        
    try:
        model = app.state.engine.model
        from model.lora import LoRALinear
        deactivated = 0
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.reset_parameters()
                deactivated += 1
                
        return {"status": "success", "layers_deactivated": deactivated}
    except Exception as e:
        logger.error(f"Failed to deactivate adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        app.state.engine_semaphore.release()


@app.post("/feedback")
def submit_feedback(body: FeedbackRequest):
    """Save feedback to data/feedback.jsonl."""
    os.makedirs("data", exist_ok=True)
    feedback_file = os.path.join("data", "feedback.jsonl")
    
    entry = {
        "timestamp": time.time(),
        "prompt": body.prompt,
        "response": body.response,
        "rating": body.rating,
        "correction": body.correction
    }
    
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        
    return {"status": "success"}


@app.get("/feedback")
def get_feedback():
    """Get stored feedback from data/feedback.jsonl."""
    feedback_file = os.path.join("data", "feedback.jsonl")
    if not os.path.exists(feedback_file):
        return {"feedback": []}
        
    results = []
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except:
                    pass
    return {"feedback": results}

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
                except:
                    pass
    return {"dataset": results}
