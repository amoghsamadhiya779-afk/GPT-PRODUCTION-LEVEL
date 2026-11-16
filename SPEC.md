# Memory & KV-Cache Systems Specification

## 1. PagedAttention (`model/paged_kv.py`)
Implements OS-style virtual memory paging for Transformer Key/Value caches (Kwon et al., vLLM, SOSP '23).

### Physical Arena (`PhysicalBlockManager`)
- **Pre-allocated KV Pool Shape:** `(num_blocks, block_size, num_layers, num_heads, head_dim)` for both `k_pool` and `v_pool`.
- **Free-List:** `collections.deque[int]` initialized with `0 .. num_blocks - 1` for $O(1)$ `allocate_block()` and `free_block()`.
- **Reference Counting & Copy-on-Write:**
  - `ref_counts[block_id]` tracks how many sequences or `RadixNode` entries reference a physical block.
  - `copy_on_write(block_id, valid_tokens)` returns `block_id` unchanged if `ref_count == 1`; otherwise allocates a new block, clones slots `[:valid_tokens]` across all layers/heads, and decrements the shared block's count.

### Logical Block Table (`PagedKVCache`)
- **Per-Sequence State:** `block_table: list[int]` and `seq_len: int`.
- **Address Translation:** Token at position `pos` maps to:
  - `logical_block = pos // block_size`
  - `offset = pos % block_size`
- **Layer Synchronization:** Layer `0` owns block allocation, Copy-on-Write checks, and `seq_len` advancement; layers `1 .. n_layers - 1` write into the already-allocated slots at `start_pos = seq_len - num_tokens`.
- **Contiguous Gather:** `gather_kv(layer_idx)` collects `[:valid_len]` slices (`min(block_size, seq_len - i * block_size)`) into `(1, num_heads, seq_len, head_dim)` for `MultiHeadAttention`.

## 2. RadixAttention (`model/radix_cache.py`)
Implements cross-request prefix caching with a block-aligned Compressed Trie (Zheng et al., SGLang, NeurIPS '24).

- **Block-Aligned Nodes (`RadixNode`):**
  - Stores `tokens: tuple[int, ...]` (always a multiple of `block_size`) and matching `block_ids: list[int]`.
  - `children` is keyed by `tuple[int, ...]` (the first `block_size` tokens of the child edge).
  - Tracks monotonic `last_access` ticks and `lock_ref` pin counts.
- **Core Operations (`RadixTree`):**
  - `match_prefix(token_ids)`: Walks matching edges, refreshes `last_access`, and returns `(matched_block_ids, last_matched_node)` snapped to whole blocks.
  - `_split_node(node, split_blocks)`: Splits an edge mid-block-boundary while keeping `node.children` intact on the suffix child.
  - `insert(token_ids, block_ids)`: Inserts full blocks into the trie and pins new blocks via `manager.ref_block()`.
  - `lock_node(node, delta)`: Adjusts `lock_ref` along the ancestor chain from `node` up to `root`.
  - `evict_lru(num_blocks_needed)`: Uses a `heapq` Min-Heap over unlocked leaves (`lock_ref == 0`) ordered by `(last_access, id(node), node)` with cascading parent eviction.

## 3. Model Integration
- `GPTModel.forward(..., paged_kv=None)` computes position indices from `paged_kv.seq_len` and passes `paged_kv` + `layer_idx=i` through `TransformerBlock.forward` to `MultiHeadAttention.forward`.
- Default behavior (`paged_kv=None`) remains 100% backward-compatible with `layer_past` and `generate_stream()`.
