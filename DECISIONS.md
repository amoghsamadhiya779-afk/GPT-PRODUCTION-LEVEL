# Architectural Decisions (ADRs)

## ADR-001: Combined Multi-Layer Physical Pool Shape
- **Choice:** Allocate `k_pool` and `v_pool` as `(num_blocks, block_size, num_layers, num_heads, head_dim)` so Layer `0` drives block allocation and Copy-on-Write for all layers.
- **Alternative Rejected:** Separate block managers per layer.
- **Rationale:** All transformer layers in GPT-2 process the exact same token positions per step. Sharing one `block_table` per sequence eliminates per-layer metadata duplication and guarantees identical physical block lifetimes across layers.

## ADR-002: Keying `RadixNode.children` by First `block_size` Token Tuple
- **Choice:** Key `RadixNode.children` with `tuple[int, ...]` (`tokens[:block_size]`) rather than a single integer `tokens[0]`.
- **Rationale:** When `block_size > 1`, two prompts sharing only token `0` (e.g., `[10, 20]` vs. `[10, 99]` at `block_size = 2`) share zero complete physical blocks. Keying by a single token would overwrite the first branch and leak physical blocks; keying by the first block tuple avoids collisions while maintaining block alignment.

## ADR-003: Pure-PyTorch `gather_kv` vs. Custom CUDA Attention Kernel
- **Choice:** Materialize `(1, num_heads, seq_len, head_dim)` in `PagedKVCache.gather_kv()` via `torch.cat` before `Q @ K^T`.
- **Trade-off:** Unlike vLLM's custom CUDA PagedAttention kernel (which computes attention directly over scattered blocks without gathering), `gather_kv()` incurs one contiguous copy per attention step. However, on CPU/pure PyTorch it preserves exact `torch.equal` parity with standard `GPTModel` while eliminating $O(T^2)$ incremental `torch.cat` allocations and enabling block-level CoW + Radix prefix sharing.

## ADR-004: On-Demand Min-Heap Over Unlocked Leaves in `evict_lru`
- **Choice:** Store `last_access` ticks in $O(1)$ on reads/inserts and build a `heapq` Min-Heap over unlocked leaves (`lock_ref == 0`) inside `evict_lru()`.
- **Alternative Rejected:** Maintaining a persistent heap with manual bubble-up on every `match_prefix()` hit.
- **Rationale:** Eliminates $O(\log N)$ heap rebalancing on every prefix read and handles cascading parent-to-leaf promotion cleanly in $O(L + K \log L)$ when a parent loses its last child.
