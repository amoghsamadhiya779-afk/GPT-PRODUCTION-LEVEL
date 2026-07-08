---
title: GPT From Scratch
emoji: 🚀
colorFrom: gray
colorTo: gray
sdk: docker
pinned: false
---

# GPT-2 From Scratch: Serving API & Next.js Playground

[![Live Demo](https://img.shields.io/badge/Live_Demo-GPT_Studio-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://gpt-production-level.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

This project is a decoder-only transformer language model built entirely from scratch using PyTorch primitives, complete with a REST serving layer and a Next.js chat interface. 

*Built from scratch — no `transformers` library, no high-level wrapping. Features KV-Caching optimizations, LoRA hot-swapping, and grounded RAG search.*

---

## 1. System Architecture

The project decouples the Next.js frontend from the compute-heavy FastAPI model server.

```mermaid
graph TD
    User([User Browser])
    subgraph Frontend [Next.js UI - Vercel]
        UI[Chat Interface<br/>Persona Management]
    end
    subgraph Backend [FastAPI - Docker]
        API[FastAPI Server<br/>SSE Streaming & LoRA Routing]
    end
    subgraph Engine [Inference Layer]
        GPT2[Custom GPT-2 Engine<br/>406M Params]
        Search[Web Search<br/>Serper.dev API]
    end

    User -->|Interacts| UI
    UI -->|REST / SSE| API
    API -->|Context & Tokens| GPT2
    API -->|Retrieval| Search
```

---

## 2. Model & Inference Capabilities

### KV-Caching (Performance)
Standard transformers recalculate attention (Query, Key, Value) for all past tokens at every step ($O(t^2)$ latency). We implemented a **Key-Value Cache** that saves $K,V$ tensors for past tokens, passing only the single newest token into the model ($O(t)$ latency).

```mermaid
graph LR
    subgraph Pre-fill Phase
        P[Prompt] -->|Q,K,V Projection| Attn1[Compute Attention]
        Attn1 --> C1[(Store K,V in Cache)]
    end
    subgraph Decode Phase (Step t)
        T[New Token t] -->|Q,K,V Projection| Attn2[Compute Attention]
        C2[(Load K,V from Cache)] --> Attn2
        Attn2 --> C3[(Append new K,V to Cache)]
    end
    C1 -.-> C2
```

**Honest Benchmarks (Local CPU):**
- **GPT-2 Small (124M)**: ~21.4 tokens/second
- **GPT-2 Medium (406M)**: ~8.6 tokens/second

### Dynamic LoRA Personas
The backend supports hot-swapping LoRA (Low-Rank Adaptation) adapters at runtime without reloading the 406M parameter base model. 
- Personas (e.g., *Math Tutor*, *Physics Helper*) inject `[persona: <name>]` tags.
- The backend activates the corresponding adapter and restores the base state on exit.

### Grounded RAG Generation
When web search is enabled, the API:
1. Queries Serper.dev for live snippets.
2. Ranks and deduplicates snippets based on keyword overlap.
3. Pre-pends the snippets as context.
4. **Safety Net**: Computes extractive overlap on the generated answer; if overlap is near zero (hallucination), it prepends a direct quote from the sources.

---

## 3. Project Structure & Testing

The system is covered by a test suite (`pytest`) comprising **44 passing integration and unit tests**.

```
GPT-PRODUCTION-LEVEL/
├── app/                  # FastAPI server, inference engine, RAG search
├── model/                # PyTorch primitives (attention, layers, lora, gpt)
├── frontend/             # Next.js App Router (React)
├── data/                 # Datasets & tokenization utilities
├── training/             # Pre-training and LoRA fine-tuning scripts
├── tests/                # 44 unit & integration tests
└── checkpoints/          # Base models and adapter states
```

---

## 4. Setup & Running

**Prerequisites:** Python 3.10+, Node.js 18+

### Backend (FastAPI)
```bash
# Setup and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements-dev.txt

# Start FastAPI serving backend
python -m uvicorn app.api:app --reload --port 8000
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

---

## 5. Limitations & Reality Check

While this is a robust system, it is built for educational/portfolio purposes and is not a replacement for commercial LLMs:
- **CPU Bottleneck**: The backend currently targets CPU deployment (e.g. Hugging Face free tier). Real-world systems run on GPUs via Triton/vLLM.
- **Model Size**: 406M parameters is very small. It struggles with complex logical reasoning without RAG grounding.
- **Batching**: The FastAPI implementation handles requests sequentially or via threads. It lacks Continuous Batching (iteration-level scheduling) required for FAANG-scale throughput.
- **Generation Quality**: The custom LoRA finetuning on Cosmopedia text introduces style shifts but does not eliminate hallucinations entirely.

---

## 👨‍💻 About the Author

**Amogh Samadhiya** — *Backend & MLOps Engineer*

Final-year B.Tech student specializing in ML Systems, Distributed Systems, and Production MLOps.

| **Production Stack** | FastAPI • Docker • Kubernetes • MLflow • Apache Airflow • AWS • Python • C++17 |
| :--- | :--- |

**Connect & Collaborate:**
* 📧 **Email**: [amoghsamadhiya779@gmail.com](mailto:amoghsamadhiya779@gmail.com)
* 🔗 **LinkedIn**: [amogh-samadhiya](https://www.linkedin.com/in/amogh-samadhiya-8890b82b8/)
* 💼 **Availability**: *Open to remote Backend / ML Engineering internships and opportunities.*
