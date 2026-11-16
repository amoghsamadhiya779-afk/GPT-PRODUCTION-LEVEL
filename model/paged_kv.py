#   model/paged_kv.py 

"""Paged Attention KV-Cache and Physical Block Manager (Kwon wt at ,  SOSP 2023) ."""

from collections import deque
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch import Tensor


class PhysicalBlockManager:
    """OS-style physical page allocator for Transformer KV-cache blocks."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # Pre-allocate the entire physical KV arena once at startup:
        # Shape allows 2D page indexing: pool[block_id, offset] -> (num_layers, num_heads, head_dim)
        pool_shape = (num_blocks, block_size, num_layers, num_heads, head_dim)
        self.k_pool: Tensor = torch.zeros(pool_shape, dtype=dtype, device=device)
        self.v_pool: Tensor = torch.zeros(pool_shape, dtype=dtype, device=device)

        # FIFO free-list queue of available physical block IDs + reference count table
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.ref_counts: list[int] = [0] * num_blocks

    def allocate_block(self) -> int:
        """Pop one free physical block ID in O(1) time and initialize its ref_count to 1."""
        if not self.free_blocks:
            raise RuntimeError("Out of physical KV cache blocks!")

        block_id = self.free_blocks.popleft()
        self.ref_counts[block_id] = 1

        # Zero out recycled block memory (like C calloc) so stale tokens never leak
        self.k_pool[block_id].zero_()
        self.v_pool[block_id].zero_()
        return block_id

    def free_block(self, block_id: int) -> None:
        """Decrement a block's reference count; return it to free_blocks when it hits 0."""
        if self.ref_counts[block_id] <= 0:
            raise ValueError(f"Block {block_id} is already free (ref_count={self.ref_counts[block_id]})")

        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)
    def ref_block(self, block_id: int) -> None:
        """Increment reference count when another sequence shares an allocated block."""
        if self.ref_counts[block_id] <= 0:
            raise ValueError(f"Cannot reference unallocated block {block_id}")
        self.ref_counts[block_id] += 1

    def copy_on_write(self, block_id: int, valid_tokens: int) -> int:
        """Return a writable block ID, cloning valid prefix slots if shared (ref_count > 1)."""
        if self.ref_counts[block_id] <= 0:
            raise ValueError(f"Block {block_id} is not allocated")

        # Fast path: only 1 owner, so this sequence can mutate the block in-place
        if self.ref_counts[block_id] == 1:
            return block_id

        # Slow path (Shared block): allocate a fresh page and copy valid token slots
        new_block_id = self.allocate_block()
        self.ref_counts[block_id] -= 1

        if valid_tokens > 0:
            self.k_pool[new_block_id, :valid_tokens].copy_(self.k_pool[block_id, :valid_tokens])
            self.v_pool[new_block_id, :valid_tokens].copy_(self.v_pool[block_id, :valid_tokens])

        return new_block_id


class PagedKVCache:
    """Per-Sequence logical-to-physical Block Table (vLLM Page Table) """

    def __init__ (self , manager: PhysicalBlockManager)  -> None :

       self.manager=manager
       self.block_table :list[int] = []
       self.seq_len:int =0

    def append_tokens(self,layer_idx:int, k_new:Tensor, v_new:Tensor) -> None:

        """ Wrtie new token K/V states of shape (1,num_heads,num_tokens, head_dim) into pages.""" 
        num_tokens = k_new.shape[2]
        #Layer 0 advances seq_len ; subsequent layers write to the same positions 
        start_pos = self.seq_len if layer_idx==0 else self.seq_len-num_tokens

        for t in range(num_tokens):
            pos=start_pos+t 
            logical_block=pos//self.manager.block_size
            offset=pos %self.manager.block_size


            #Allocate a new physical page only on layer0 when crossing a block boundary 
            if logical_block == len(self.block_table):
                self.block_table.append(self.manager.allocate_block())
            block_id = self.block_table[logical_block]     

            #If writing into a shared partial block on Layer 0 ,Trigger Copy-on-Write 
            if layer_idx==0 and self.manager.ref_counts[block_id]>1:
                block_id = self.manager.copy_on_write(block_id,valid_tokens=offset)
                self.block_table[logical_block]= block_id
            self.manager.k_pool[block_id,offset,layer_idx].copy_(k_new[0,:,t,:])
            self.manager.v_pool[block_id,offset,layer_idx].copy_(v_new[0,:,t,:])

        if layer_idx ==0:
             self.seq_len += num_tokens
    def gather_kv(self , layer_idx: int) -> tuple[Tensor, Tensor]:
        """ Reconstruct  contiguous (1,num_heads,seq_len, head_dim) K/V tensor for attention."""
        if self.seq_len == 0:
          empty = torch.empty(
            (1,self.manager.num_heads,0,self.manager.head_dim),
            dtype = self.manager.k_pool.dtype,
            device = self.manager.device,)
          return empty, empty.clone()
        k_chunks:list[Tensor]=[]
        v_chunks:list[Tensor]=[]
        block_size=self.manager.block_size

        for i, block_id in enumerate(self.block_table):
          #Full pages contribute 'block_size' tokens ; the final page contributes the remainder
         valid_len = min(block_size,self.seq_len-i * block_size)
         k_chunks.append(self.manager.k_pool[block_id,:valid_len, layer_idx ])
         v_chunks.append(self.manager.v_pool[block_id,:valid_len,layer_idx])
        # (seq_len,num_heads,head_dim) ->(1, num_heads , seq_len , head_dim)

        keys= torch.cat(k_chunks,dim=0).transpose(0,1).unsqueeze(0)
        values = torch.cat(v_chunks,dim=0).transpose(0,1).unsqueeze(0)
        return keys, values
    def free(self) -> None:
      for block_id in self.block_table:
        self.manager.free_block(block_id)
      self.block_table.clear()
      self.seq_len=0          

