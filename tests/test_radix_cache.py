""" Unit tests for RadixAttention Prefix Matching , edge splitting , and LRU eviction."""

from model.paged_kv import PhysicalBlockManager
from model.radix_cache import RadixTree

def test_radix_split_and_prefix_match() -> None:
  mgr = PhysicalBlockManager(num_blocks=6,block_size=2,num_layers=1,num_heads=1,head_dim=4)
  tree = RadixTree(mgr)

  b0,b1 = mgr.allocate_block(), mgr.allocate_block()
  assert tree.insert([10,20,30,40],[b0,b1])==2
  mgr.free_block(b0)
  mgr.free_block(b1)


  #Prompt diverges inside the second block -> Matches only [b0]

  matched, _= tree.match_prefix([10,20,30,99])
  assert matched== [b0]

  b2=mgr.allocate_block()
  assert tree.insert([10,20,30,99],[b0,b2])==1
  mgr.free_block(b2)

  assert tree.match_prefix([10,20,30,40])[0]==[b0,b1]
  assert tree.match_prefix([10,20,30,99])[0]==[b0,b2]
def test_radix_lru_eviction_and_locking() ->None:
    mgr=PhysicalBlockManager(num_blocks=4,block_size=2,num_layers=1,num_heads=1,head_dim=4)
    tree=RadixTree(mgr)

    b0,b1=mgr.allocate_block(),mgr.allocate_block()
    tree.insert([10,20,30,40],[b0,b1])

    mgr.free_block(b0)
    mgr.free_block(b1)

    b2 = mgr.allocate_block()
    tree.insert([10,20,99,88],[b0,b2])
    mgr.free_block(b2)

    #Refresh and lock [10,20,30,40] -> only unlocked cold leaf [99,88] can evict
    _, node_b1= tree.match_prefix([10,20,30,40])
    tree.lock_node(node_b1,+1)
    assert tree.evict_lru(2)==1

    #Unlocking allows cascading eviction of leaf b1 AND its now-childless parent b0
    tree.lock_node(node_b1,-1)
    assert tree.evict_lru(2)==2
    assert len(mgr.free_blocks)==4

