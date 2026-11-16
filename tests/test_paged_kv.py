# tests/test_paged_kv.py 

""" Unit tests for Paged Attention PhysicalBlockManager and PagedKVCache."""

import pytest
import torch   
from model.paged_kv import PhysicalBlockManager 
from model.config import  GPTConfig
from model.gpt import GPTModel,generate
from model.paged_kv import PagedKVCache,PhysicalBlockManager
def test_block_alloc_free_and_oom() ->None:
  mgr =PhysicalBlockManager(num_blocks=2, block_size=4, num_layers=1, num_heads=2, head_dim=8)
   
  b0=mgr.allocate_block()
  b1=mgr.allocate_block()
  assert (b0,b1)==(0,1)

  with pytest.raises(RuntimeError):
    mgr.allocate_block()
  
  mgr.ref_block(b0)
  assert mgr.ref_counts[b0]==2

  mgr.free_block(b0)
  assert len(mgr.free_blocks)==0

  mgr.free_block(b0)
  assert len(mgr.free_blocks)==1


def test_copy_on_write_clones_only_valid_tokens()   -> None:
    mgr = PhysicalBlockManager(num_blocks=2, block_size=4, num_layers =1 , num_heads=2, head_dim=8)
    b0= mgr.allocate_block()
    mgr.k_pool[b0,:2] = 99.0
    mgr.k_pool[b0,2:]  = 42.0

    #Fast path : ref_count ==1 returns same block without copying
    assert mgr.copy_on_write(b0, valid_tokens=2)==b0

    #Shared path :ref_count ==2 ,allocated b2 and Copies only [:2]

    mgr.ref_block(b0)
    b1= mgr.copy_on_write(b0, valid_tokens=2)
    assert b1 != b0
    assert mgr.ref_counts[b0] ==1 and mgr.ref_counts[b1]==1
    assert torch.all(mgr.k_pool[b1, :2] ==99.0)
    assert torch.all (mgr.k_pool[b1,2:]==0)
@torch.no_grad()
def test_paged_kv_matches_standard_generate() -> None:
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=50, context_length=32, emb_dim=16, n_heads=2, n_layers=2, dropout=0.0)
    model = GPTModel(cfg)
    model.eval()

    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    standard_out = generate(model, prompt, max_new_tokens=4, context_size=32, use_cache=True)

    mgr = PhysicalBlockManager(num_blocks=8, block_size=3, num_layers=2, num_heads=2, head_dim=8)
    paged_kv = PagedKVCache(mgr)

    # Prefill the 5-token prompt, then decode 3 more tokens step by step
    logits = model(prompt, paged_kv=paged_kv)
    next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    paged_out = torch.cat((prompt, next_tok), dim=1)

    for _ in range(3):
        logits = model(next_tok, paged_kv=paged_kv)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        paged_out = torch.cat((paged_out, next_tok), dim=1)

    assert torch.equal(paged_out, standard_out)
    assert paged_kv.seq_len == 8 and len(paged_kv.block_table) == 3
    paged_kv.free()
    assert len(mgr.free_blocks) == 8
    


  
