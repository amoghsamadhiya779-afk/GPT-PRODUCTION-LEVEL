---
title: GPT From Scratch
emoji: 🚀
colorFrom: gray
colorTo: gray
sdk: docker
pinned: false
---

# GPT-2 From Scratch: Serving API & Streamlit Playground

[![Live Demo](https://img.shields.io/badge/Live_Demo-GPT_Studio-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://gpt-production-level.vercel.app)

This project is a production-level, decoder-only transformer language model built entirely from scratch using PyTorch primitives, complete with a REST serving layer, containerized UI dashboard, and experiment tracking.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Community-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Built from scratch — no HuggingFace, no high-level wrapping. Includes KV-Caching optimizations, containerized microservices, Pydantic validation, and professional telemetry.*

---

## System Architecture

The project implements a decoupled microservices architecture designed to mimic production AI systems:

```
                  ┌──────────────────────────────┐
                  │      User Web Browser        │
                  └──────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
  ┌──────────────────────────────┐ ┌──────────────────────────────┐
  │        Next.js Chat UI       │ │     Streamlit Dashboard      │
  ├──────────────────────────────┤ ├──────────────────────────────┤
  │ * Deployed on Vercel         │ │ * Internal Telemetry         │
  │ * Beautiful animations       │ │ * Local Engine Fallback      │
  │ * Teach Mode & Grounding     │ │ * Playground Branch          │
  └──────────────┬───────────────┘ └──────────────┬───────────────┘
                 │                                │
                 └───────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │     FastAPI Model Server     │
                  ├──────────────────────────────┤
                  │ * Preloads weights once      │
                  │ * SSE Token Streaming        │
                  │ * KV-Cached GPT-2 Inference  │
                  │ * Background LoRA Fine-tuning│
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     GPT-2 From Scratch       │
                  └──────────────────────────────┘
```

- **Separation of Concerns**: Serves compute-heavy model inference (FastAPI) independently from user interfaces (Next.js & Streamlit). In production, this allows frontend instances to scale cheaply while keeping model weights in dedicated, GPU-accelerated replicas.
- **Fail-Safe Fallback**: If the FastAPI server is offline, the Streamlit dashboard automatically switches to **Standalone Standby Mode**, loading the model weights locally in CPU space to ensure 100% live uptime. The Next.js frontend gracefully falls back to a mocked offline mode.

---

## Key Optimization: KV-Caching

### The Bottleneck
Standard transformers recalculate the attention projections (Query, Key, Value) for *all past tokens* at each step of text generation. For a sequence of length $t$:
1. $Q, K, V$ are projected for all $t$ tokens.
2. Latency grows as $O(t^2)$, causing generation to slow down as output length increases.

### The Solution
We implemented a custom **Key-Value (KV) Cache** in the causal attention and model blocks. 
During generation:
1. **Pre-fill Step**: The initial prompt is run through the model, and all attention Key/Value tensors are saved.
2. **Decode Step**: At each step $t > 1$, we pass **only the single newest token** into the model.
3. The model projects $Q, K, V$ for the single new token, retrieves the cached $K, V$ values representing previous tokens, concatenates them, and computes the attention output.
4. Latency remains flat at $O(t)$ per token, yielding a significant speedup:

```
Without KV-Cache (Recalculating):
Token 1: [T1] -> Projects Q, K, V
Token 2: [T1, T2] -> Projects Q, K, V for T1, T2
Token 3: [T1, T2, T3] -> Projects Q, K, V for T1, T2, T3

With KV-Cache:
Token 1: [T1] -> Projects & Caches K1, V1
Token 2: [T2] -> Projects K2, V2 -> Attends Q2 to [K1+K2, V1+V2] -> Caches K2, V2
Token 3: [T3] -> Projects K3, V3 -> Attends Q3 to [K1+K2+K3, V1+V2+V3] -> Caches K3, V3
```

---

## Textbook & Prompts Pre-Training

To pre-train the model, we consolidate mathematical and educational textbooks from the **Hugging Face Cosmopedia** dataset (subsets: `auto_math_text`, `khanacademy`, `openstax`) along with persona prompt templates from **Awesome ChatGPT Prompts** (`fka/prompts.chat`). 

We support two modes of training:

### A. Local Pre-Compiled Corpus (Offline Mode)
First, compile the datasets locally to generate a consolidated text file:
```bash
python data/download_cosmopedia.py
```
This generates `data/cosmopedia_math.txt`. Next, launch the training script:
```bash
python training/train.py --config configs/gpt2_small.yaml --data data/cosmopedia_math.txt
```

### B. On-the-Fly Streaming (Zero Local Storage Mode)
If you want to train the model without downloading files locally (e.g. to avoid the 92 GB Cosmopedia download limit), you can stream data chunks directly from Hugging Face into memory. We use online split routing (modulus split) to divide the stream into 90% training and 10% validation batches:
```bash
python training/train.py --data stream_hf --data_type stream_hf --epochs 3 --batch_size 4 --steps_per_epoch 1000 --use_amp
```
*(Note: On Windows, use `py` instead of `python` if `python` is not in your PATH.)*

---

## Project Structure

```
GPT-PRODUCTION-LEVEL/
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT License
├── .gitignore
├── Dockerfile                  # Builds multi-entrypoint serving image
├── docker-compose.yml          # Orchestrates backend & frontend
├── notebooks/
│   └── GPT_Finetuning_Colab.ipynb # Google Colab pipeline for training/inference
│
├── configs/
│   ├── gpt2_small.yaml         # GPT-2 Small configuration (124M parameters)
│   └── gpt2_tiny.yaml          # Tiny model configuration (approx. 13M parameters)
│
├── frontend/                   # Next.js App Router Chat UI
│   ├── src/app/                # Next.js pages and layouts
│   ├── src/components/         # React components (ChatWindow, Composer, TeachView)
│   ├── src/lib/                # API clients and utilities
│   └── package.json            # Node dependencies
│
├── model/                      # PyTorch model definitions
│   ├── __init__.py             # Exports
│   ├── config.py               # GPTConfig & TrainingConfig dataclasses
│   ├── attention.py            # Causal Multi-Head Attention (with KV-Cache)
│   ├── layers.py               # LayerNorm, GELU, FeedForward, TransformerBlock
│   ├── lora.py                 # Parameter-efficient adapters
│   ├── gpt.py                  # GPTModel + cached generation utilities
│   └── tokenizer.py            # tiktoken BPE wrapper
│
├── training/                   # Model pre-training
│   ├── __init__.py
│   ├── train.py                # Pre-training & finetuning pipeline with MLflow
│   ├── load_pretrained.py      # Pre-trained weights loader
│   └── utils.py                # Loss calculation, evaluation, plotting
│
├── data/                       # Data loader & preprocessing
│   ├── __init__.py
│   ├── dataset.py              # GPTDataset & StreamedTextbookDataset loader
│   ├── download.py             # Demo text download utility
│   └── download_cosmopedia.py  # Cosmopedia & prompts dataset downloader
│
├── app/                        # Inference serving & user interfaces
│   ├── __init__.py
│   ├── generate.py             # CLI text generation
│   ├── inference.py            # Inference engine with time & speed benchmarks
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── api.py                  # FastAPI server endpoints
│   ├── finetune.py             # Background LoRA finetuning jobs
│   ├── search.py               # Vector search & retrieval modules
│   └── dashboard.py            # Streamlit interactive UI dashboard
│
└── tests/                      # Pytest unit testing suite
    ├── __init__.py
    ├── test_api.py             # API endpoints and SSE streaming
    ├── test_cosmopedia.py      # Cosmopedia loader and streamed training tests
    ├── test_dataset.py         # Static sliding window and instruction tests
    ├── test_model.py           # Model shapes, mask properties, and KV-cache equivalence
    ├── test_search.py          # Vector search utility verification tests
    ├── test_tokenizer.py       # Tokenizer encoding & decoding round-trip tests
    └── test_train.py           # Integration dry-run training loop tests
```

---

## Configuration

The application can be configured using environment variables:

| Variable | Default | Service | Description |
| :--- | :--- | :--- | :--- |
| `MODEL_CHECKPOINT` | `checkpoints/best_model.pt` | FastAPI / Streamlit | Path to the PyTorch model weights file. |
| `BACKEND_URL` | `http://localhost:8000` | Streamlit | The URL of the FastAPI model server. |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | FastAPI | Comma-separated list of CORS allowed origins. |
| `WEB_SEARCH_API_KEY` | *(None)* | FastAPI | Serper.dev API key for web search grounding. |
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Next.js Frontend | The URL of the FastAPI backend for the Next.js client. |

---

## API Documentation

The FastAPI backend exposes the following endpoints (default port: `8000`):

### 1. Model Health Check (`GET /health`)
Returns the active serving status, loaded model weights path, parameter size, and hardware device.

**Example Response**:
```json
{
  "status": "active",
  "checkpoint": "checkpoints/best_model.pt",
  "parameters": 124000000,
  "device": "cpu"
}
```

### 2. Generate Text (`POST /generate`)
Validates input parameters via Pydantic and executes text generation. Includes optional `web_search` for RAG grounding.

**Request Payload**:
```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_k": 50,
  "use_cache": true,
  "web_search": true
}
```

**Response Payload**:
```json
{
  "prompt": "Once upon a time",
  "generated_text": "Once upon a time Jack Gisburn rather a cheap genius...",
  "tokens_generated": 100,
  "time_taken_seconds": 0.908,
  "tokens_per_second": 110.2,
  "sources": [{"title": "Web Page", "snippet": "Context", "link": "https://..."}]
}
```

### 3. Generate Text Stream (`POST /generate/stream`)
Uses Server-Sent Events (SSE) to stream generation tokens one by one, providing immediate feedback.

**Example SSE Output**:
```text
data: {"sources": [{"title": "Web Page", "snippet": "Context", "link": "https://..."}]}

data: {"token": "Once", "index": 0}
data: {"token": " upon", "index": 1}

data: {"done": true, "tokens_generated": 2, "time_taken_seconds": 0.05, "tokens_per_second": 40.0}
```

### 4. Fine-tuning (`POST /finetune`)
Launch a background LoRA fine-tuning job on a set of examples.

**Request Payload**:
```json
{
  "examples": [{"instruction": "What is 2+2?", "response": "4"}],
  "adapter_name": "math_helper",
  "steps": 100,
  "lr": 0.001
}
```

### 5. Fine-tuning Status (`GET /finetune/{job_id}`)
Check the status of a running fine-tuning job.

**Example Response**:
```json
{
  "status": "running",
  "step": 45,
  "total_steps": 100,
  "current_loss": 2.14,
  "eta_seconds": 15
}
```

### 6. Adapters Management (`GET /adapters`)
List available LoRA adapters and activate/deactivate them using `POST /adapters/{name}/activate` and `POST /adapters/deactivate`.

### 7. Feedback (`POST /feedback`)
Submit user feedback on model responses.

**Request Payload**:
```json
{
  "prompt": "What is 2+2?",
  "response": "4",
  "rating": "up"
}
```

---

## Experiment Tracking with MLflow

Pre-training integrates `mlflow` to track parameters, logs, and artifacts automatically:
- **Logged Parameters**: Vocab size, context length, embedding dim, n_heads, n_layers, learning rate, weight decay, warmup steps.
- **Logged Metrics**: Train loss, validation loss, learning rate per step.
- **Logged Artifacts**: Model checkpoints (`best_model.pt`, `final_model.pt`), and loss graphs (`loss.png`).

To launch the MLflow dashboard locally:
```bash
mlflow ui
```
Open `http://localhost:5000` in your web browser to compare training runs and view loss metrics.

---

## Setup & Running

### 1. Local Run
Install dependencies locally:
```bash
# Setup virtual environment
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install requirements
pip install -r requirements-dev.txt

# Download local datasets (Optional)
python data/download.py
python data/download_cosmopedia.py

# Run a fast training smoke test
python training/train.py --config configs/gpt2_tiny.yaml

# Start FastAPI serving backend
python -m uvicorn app.api:app --reload --port 8000

# Start Streamlit UI dashboard
python -m streamlit run app/dashboard.py --server.port 8501
```

For the Next.js frontend:
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:3000` to see the Next.js UI, or `http://localhost:8501` for the Streamlit dashboard.

### 2. Multi-Service Container Run (Docker Compose)
To launch the complete system inside isolated containers:
```bash
docker compose up --build
```
This starts:
- FastAPI server on `http://localhost:8000`
- Streamlit dashboard on `http://localhost:8501`
Checkpoints and logs directories are automatically mapped into the containers to load local weights.

### 3. Running Unit Tests
To run unit tests verifying KV-cache mathematical equivalence, streaming, and layer shapes:
```bash
python -m pytest tests/ -v
```

---

## FAANG-Scale Production Roadmap

In a FAANG production setting, serving models with FastAPI and Streamlit is replaced with highly optimized infrastructure to support millions of concurrent users. Here is how this codebase would scale:

```
[Web UI] ──► [API Gateway (Kong)] ──► [Redis Cache]
                                             │
                                     (Cache Miss)
                                             │
                                             ▼
                                    [Triton/vLLM Cluster]
                                  (Kubernetes HPA Replicas)
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                       [PagedAttention]             [Continuous Batching]
```

### 1. Serving Engine Optimization (Triton / vLLM)
- **PagedAttention**: Standard KV-Cache causes memory fragmentation. vLLM uses PagedAttention (similar to virtual memory paging in OS) to partition key-value states in non-contiguous memory blocks, reducing memory waste by 96% and allowing larger batch sizes.
- **Continuous Batching**: FastAPI processes requests sequentially or concurrently via threads, but GPUs require large batches for maximum throughput. Production engines (Triton Inference Server, vLLM) use iteration-level scheduling to batch incoming requests dynamically mid-generation, rather than waiting for preceding generations to complete.

### 2. Horizontal Scaling & High Availability
- **Kubernetes Pod Auto-scaling (HPA)**: Deploy model instances as replica pods inside a Kubernetes cluster. Scale the pods dynamically using custom Prometheus metrics tracking request queue depth or GPU load.
- **Triton Model Registry**: Load models from central cloud storage (like S3/GCS) with automatic model versioning, staging rollouts, and multi-model hosting on single containers.

### 3. Load Balancing & Queueing
- **Asynchronous Task Queues**: For large-scale batch generation, requests are sent to a message broker (RabbitMQ/Kafka) and consumed by model serving queues to prevent blocking HTTP connections.
- **API Gateway**: Run an API Gateway (like Kong or AWS API Gateway) to manage rate limiting, API token verification, and request routing across model clusters.

---

## 👨‍💻 About the Author

### **Amogh Samadhiya** — *Backend & MLOps Engineer*

Final-year B.Tech student specializing in ML Systems, Distributed Systems, and Production MLOps.

| **Production Stack** | FastAPI • Docker • Kubernetes • MLflow • Apache Airflow • AWS • Python • C++17 |
| :--- | :--- |

**Featured Systems Engineering Projects:**
* 🚀 **[Voyage Analytics](https://github.com/amoghsamadhiya779-afk/voyage-analytics-mlops)** — End-to-end MLOps platform orchestrating automated model pipelines with Apache Airflow and scalable deployments on Kubernetes.
* 📈 **[Quantum Yield](https://github.com/amoghsamadhiya779-afk/quantitative-ml-trading-platform)** — Quantitative trading platform running BiLSTM forecasting models across 8 global market indices.
* ⚡ **[Core AI Microservice](https://github.com/amoghsamadhiya779-afk/fastapi-ml-microservice)** — High-throughput 3-tier FastAPI & MySQL inference microservice deployed on AWS EC2.

**Connect & Collaborate:**
* 📧 **Email**: [amoghsamadhiya779@gmail.com](mailto:amoghsamadhiya779@gmail.com)
* 🔗 **LinkedIn**: [amogh-samadhiya](https://www.linkedin.com/in/amogh-samadhiya-8890b82b8/)
* 💼 **Availability**: *Open to remote Backend / ML Engineering internships and opportunities.*
