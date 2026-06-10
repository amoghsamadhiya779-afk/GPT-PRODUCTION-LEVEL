<div align="center">

# GPT-2 From Scratch

### A Production-Level Implementation of the GPT-2 Language Model in Pure PyTorch

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Built entirely from scratch — no HuggingFace, no external LLM libraries.*
*Every component hand-implemented for deep understanding.*

---

</div>

## Project Overview

This project is a complete, from-scratch implementation of the GPT-2 (124M parameter) language model using only PyTorch primitives. It covers the full LLM lifecycle:

- **Tokenization** — GPT-2 BPE via tiktoken
- **Model Architecture** — Multi-Head Causal Self-Attention, LayerNorm, GELU, Transformer Blocks
- **Pre-training** — Next-token prediction with cosine LR scheduling and gradient clipping
- **Text Generation** — Greedy decoding, temperature scaling, top-k sampling
- **Production Patterns** — YAML configs, structured logging, checkpointing, CLI interfaces

> **Why build from scratch?** To develop a deep, first-principles understanding of how modern LLMs work — not just how to call an API, but how every layer, every gradient, and every token is processed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GPT-2 (124M)                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input Token IDs ──► Token Embedding (50,257 × 768)     │
│                   + Positional Embedding (256 × 768)    │
│                   ──► Embedding Dropout                 │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │         × 12 Transformer Blocks                   │  │
│  │                                                   │  │
│  │  ┌─ LayerNorm ─► Multi-Head Attention (12 heads)  │  │
│  │  │               ──► Dropout ──► + Residual       │  │
│  │  │                                                │  │
│  │  └─ LayerNorm ─► Feed-Forward (768 → 3072 → 768) │  │
│  │                  ──► Dropout ──► + Residual        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Final LayerNorm ──► Linear Head ──► Logits (50,257)    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

| Component | Implementation | Why |
|-----------|----------------|-----|
| Attention | Separate Q/K/V projections | Clearer than fused QKV for learning |
| Normalization | Pre-LayerNorm (custom) | Matches GPT-2; hand-implemented for understanding |
| Activation | GELU (tanh approx.) | Hand-implemented, not standard PyTorch nn.GELU |
| Positional Encoding | Learned embeddings | Matches GPT-2 (not sinusoidal/RoPE) |
| Config | Python @dataclass + YAML | Type-safe, serializable, IDE-friendly |

---

## Project Structure

```
GPT-PRODUCTION-LEVEL/
├── README.md                 # You are here
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
├── .gitignore
│
├── configs/
│   ├── gpt2_small.yaml       # Model & training hyperparameters
│   └── gpt2_tiny.yaml        # Tiny model config for fast development
│
├── model/                    # Core model architecture
│   ├── __init__.py              # Package exports
│   ├── config.py                # GPTConfig & TrainingConfig dataclasses
│   ├── attention.py             # Multi-Head Causal Self-Attention
│   ├── layers.py                # LayerNorm, GELU, FeedForward, TransformerBlock
│   ├── gpt.py                   # GPTModel + generation functions
│   └── tokenizer.py             # GPT-2 BPE tokenizer wrapper
│
├── training/                 # Training pipeline
│   ├── __init__.py
│   ├── train.py                 # Full training loop with LR scheduling
│   └── utils.py                 # Loss computation, evaluation, plotting
│
├── data/                     # Data loading & processing
│   ├── __init__.py
│   ├── dataset.py               # GPTDataset + DataLoader factory
│   └── download.py              # Training data downloader
│
├── app/                      # Inference & serving
│   ├── __init__.py
│   └── generate.py              # Interactive text generation CLI
│
└── tests/                    # Test suite
    ├── __init__.py
    ├── test_model.py            # Architecture tests
    └── test_tokenizer.py        # Tokenizer tests
```

---

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/amoghsamadhiya779-afk/GPT-PRODUCTION-LEVEL.git
cd GPT-PRODUCTION-LEVEL

# Create virtual environment
py -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Training Data

```bash
py data/download.py
```

### 3. Train the Model

```bash
# Train with tiny model (fast validation run)
py training/train.py --config configs/gpt2_tiny.yaml

# Train with custom YAML config (GPT-2 Small)
py training/train.py --config configs/gpt2_small.yaml
```

### 4. Generate Text

```bash
# Generate with a custom prompt
py app/generate.py --checkpoint checkpoints_tiny/best_model.pt --prompt "Once upon a time"

# Adjust creativity and sampling
py app/generate.py --checkpoint checkpoints_tiny/best_model.pt --prompt "Once upon a time" --temperature 0.8 --top-k 40
```

### 5. Run Tests

```bash
py -m pytest tests/ -v
```

---

## Configuration

All hyperparameters are managed via YAML configuration files:

```yaml
# configs/gpt2_small.yaml
model:
  vocab_size: 50257
  context_length: 256
  emb_dim: 768
  n_heads: 12
  n_layers: 12
  dropout: 0.1

training:
  learning_rate: 5.0e-4
  num_epochs: 10
  batch_size: 2
  warmup_steps: 20
  max_grad_norm: 1.0
```

---

## Production-Level Features

| Feature | Description |
|---------|-------------|
| **Cosine LR Schedule** | Linear warmup + cosine decay for stable training |
| **Gradient Clipping** | Prevents exploding gradients (max_norm=1.0) |
| **Checkpointing** | Saves best model by validation loss + full state for resume |
| **Structured Logging** | Python logging module with timestamps |
| **YAML Configs** | Reproducible experiments without code changes |
| **Test Suite** | pytest-based tests for model architecture & tokenizer |
| **CLI Interfaces** | argparse-based CLIs for training, generation, data download |
| **Type Hints** | Throughout all modules for IDE support |

---

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | ~124M |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Embedding Dim | 768 |
| Attention Heads | 12 |
| Transformer Layers | 12 |
| Context Length | 256 tokens |
| Feed-Forward Dim | 3,072 (4x emb_dim) |
| Activation | GELU (tanh approx.) |

---

## Acknowledgments

- **Sebastian Raschka** — *Build a Large Language Model (From Scratch)* — the foundational reference for this implementation
- **OpenAI** — for the original GPT-2 paper and model architecture
- **Andrej Karpathy** — for making LLM internals accessible through educational content

---

## Author

**Amogh Samadhiya**

- GitHub: [@amoghsamadhiya779-afk](https://github.com/amoghsamadhiya779-afk)

---

<div align="center">

*Built with pure PyTorch*

</div>
