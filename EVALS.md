# Evaluations & Benchmarks

## 1. Correctness Suite (`pytest`)
- **Command:** `python -m pytest tests -v`
- **Result:** **140 passed** (all existing API, batching, retrieval, security, dataset, LoRA, speculative decoding, and tokenizer tests, plus 5 new KV systems tests).
- **New Verification Tests:**
  - `tests/test_paged_kv.py`:
    - `test_block_alloc_free_and_oom`: Verifies FIFO allocation, reference counting, double-free prevention, and `RuntimeError` on pool exhaustion.
    - `test_copy_on_write_clones_only_valid_tokens`: Verifies zero-copy fast path (`ref_count == 1`) and isolated partial-page cloning (`ref_count > 1`).
    - `test_paged_kv_matches_standard_generate`: Verifies bit-for-bit generated token equivalence (`torch.equal`) between `GPTModel(..., paged_kv=...)` and standard `generate(..., use_cache=True)`, plus zero block leaks after `free()`.
  - `tests/test_radix_cache.py`:
    - `test_radix_split_and_prefix_match`: Verifies block-aligned prefix hits, mid-edge splitting, and sibling branch insertion.
    - `test_radix_lru_eviction_and_locking`: Verifies `lock_node` protection of active nodes and cascading Min-Heap LRU eviction of unlocked parent/leaf chains.

## 2. Performance & Memory Benchmarks (`evals/run_kv_benchmark.py`)
- **Command:** `$env:PYTHONPATH = "."; python evals/run_kv_benchmark.py`
- **Hardware / Environment:** Local CPU, Python 3.12.0, PyTorch 2.12.1+cpu.

| Benchmark | Setup | Baseline | Ours (Paged + Radix) | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Memory Fragmentation** | 8 requests (total `1,535` tokens), `context_length = 512`, `block_size = 16` | Contiguous pre-alloc: `4,096` slots (**62.5% waste**) | Paged blocks: `1,568` slots (**2.1% waste**) | **61.7% fewer KV slots reserved** |
| **Shared-Prefix TTFT** | 4-layer `GPTModel` (`emb_dim=256`, `n_heads=4`), `192`-token shared prefix + `16`-token suffix | Standard prefill: **33.9 ms** | Radix hit + suffix prefill: **10.1 ms** | **3.34x faster TTFT** (`torch.allclose` verified) |
