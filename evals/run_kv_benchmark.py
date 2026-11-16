# evals /run_kv_benchmark.property

""" Benchmarks PagedAttention memory savings and RadixAttention prefic TTFT speedup."""

import math
import time 
import torch
from model.config import GPTConfig
from model.gpt import GPTModel
from model.paged_kv import PagedKVCache, PhysicalBlockManager
from model.radix_cache import RadixTree

def benchmark_memory(block_size:int =16 , context_length:int=512) ->None:
    lengths=[80,110,140,175,210,240,270,310]
    actual_tokens=sum(lengths)
    contig_slots=len(lengths) * context_length
    paged_slots= sum(math.ceil(n/block_size)*block_size for n in lengths)

    contig_waste = (contig_slots-actual_tokens)/contig_slots *100
    paged_waste=(paged_slots-actual_tokens)/paged_slots*100
    print(f"[Memory] Contiguous waste: {contig_waste:.1f}% ({contig_slots}slots)")
    print(f"[Memory] Paged wasted (B={block_size}):{paged_waste:.1f}% ({paged_slots}slots) ")

@torch.no_grad()
def benchmark_ttft (block_size:int =16) ->None:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50257,context_length=256,emb_dim=256,n_heads=4,n_layers=4,dropout=0.0)
    model=GPTModel(cfg).eval()
    mgr=PhysicalBlockManager(num_blocks=64,block_size=block_size,num_layers=4,num_heads=4,head_dim=64)
    tree=RadixTree(mgr)

    shared_prefix=torch.randint(0,50257,(1,192))
    prompt_1 = torch.cat((shared_prefix,torch.randint(0,50257,(1,16))),dim=1)
    prompt_2 = torch.cat((shared_prefix, torch.randint(0,50257,(1,16))),dim=1)

    #Warmup & populate RadixTree with Request 1

    kv1=PagedKVCache(mgr)
    _= model(prompt_1,paged_kv=kv1)
    tree.insert(prompt_1[0].tolist(),kv1.block_table)
    kv1.free()
    #Baseline:standard unshared prefill on Request 2
    t0= time.perf_counter()
    logits_std, _=model(prompt_2,use_cache=True)
    t_std=(time.perf_counter()-t0)*1000
    #RadixAttention :resue matched prefix blocks,prefill only the 16-token suffix
    t0=time.perf_counter()
    matched_blocks,_=tree.match_prefix(prompt_2[0].tolist())
    kv2 = PagedKVCache(mgr)
    for b in matched_blocks:
        mgr.ref_block(b)
        kv2.block_table.append(b)
    kv2.seq_len =len(matched_blocks)*block_size
    logits_radix = model(prompt_2[:,kv2.seq_len :],paged_kv=kv2)
    t_radix =(time.perf_counter()-t0) *1000

    assert torch.allclose(logits_radix[:,-1,:],logits_std[:,-1,:],atol=1e-4)
    kv2.free()
    print (f"[TTFT] Standard prefill : {t_std:.1f} ms|Radix hit: {t_radix:.1f}ms({t_std/t_radix:.2f}x faster)")

if __name__ == "__main__":
    benchmark_memory()
    benchmark_ttft()
