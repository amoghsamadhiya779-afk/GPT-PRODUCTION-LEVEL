# app/api.py
"""FastAPI REST API service for GPT-2 model serving.

Exposes endpoints for text generation, system health checks, and
training loss plot visualization. Supports lifespan checkpoint preloading.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Add root folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import GenerationRequest, GenerationResponse
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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Retrieve service health status, hardware device, and active checkpoint metadata."""
    status_code = status.HTTP_200_OK
    if app.state.status in ["error", "failed"]:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": app.state.status,
            "checkpoint": app.state.checkpoint_path,
            "parameters": getattr(app.state, "parameter_count", 0),
            "device": getattr(app.state, "device", "unknown"),
            "error_details": getattr(app.state, "error_msg", None),
        },
    )


@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    """Generate text from a prompt using the loaded GPT model."""
    if app.state.status == "failed" or not hasattr(app.state, "engine"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model serving engine is offline or failed to initialize.",
        )

    try:
        prompt_text = request.prompt
        sources = None
        
        if request.web_search:
            from app.search import web_search
            sources = web_search(prompt_text, max_results=3)
            if sources:
                context_str = "\n".join([f"- {res['snippet']}" for res in sources])
                prompt_text = (
                    "Context from Web Search:\n"
                    f"{context_str}\n\n"
                    "Write a factual response using the context if helpful:\n"
                    f"Prompt: {request.prompt}\n\n"
                    "Response:"
                )

        response_data = app.state.engine.generate(
            prompt=prompt_text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_cache=request.use_cache,
        )
        
        # If RAG was used, extract only the generated answer
        if request.web_search and sources:
            gen_text = response_data["generated_text"]
            if "Response:" in gen_text:
                answer = gen_text.split("Response:")[-1].strip()
                response_data["generated_text"] = request.prompt + " " + answer
            response_data["prompt"] = request.prompt
            
        response_data["sources"] = sources
        return response_data
    except Exception as e:
        logger.error("Generation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


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
