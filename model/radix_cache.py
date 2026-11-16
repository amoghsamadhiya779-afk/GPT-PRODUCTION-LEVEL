""" RadixAttention Prefix KV-Cache using a Compressed Trie """


from __future__  import annotations

import heapq 
from model.paged_kv import PhysicalBlockManager
class  RadixNode:
    """Single compressed edge node storing token IDs and their physical KV block IDs."""
    def __init__(
        self,
        tokens:tuple[int,...]=(),
        block_ids: list[int] |None =None,
        parent :RadixNode | None =None,
        last_access :int =0,

    ) ->None:
       self.tokens:tuple[int,...]=tokens
       self.block_ids: list[int] = list(block_ids) if block_ids is not None else []
       self.parent:RadixNode |None=parent
       self.children :dict[tuple[int,...],RadixNode] ={}
       self.last_access:int = last_access
       self.lock_ref:int=0
class RadixTree:
    """Block-aligned compresse prefix trie for cross-request KV-cache reuse."""
    def __init__(self,manager:PhysicalBlockManager) ->None:
        self.manager=manager
        self.block_size=manager.block_size
        self._clock=0
        self.root=RadixNode()

    def _next_tick(self)->int:
        self._clock +=1
        return self._clock    

    def match_prefix(self,token_ids: list[int]) -> tuple[list[int], RadixNode]:
        """Return (matched_block_ids,last_matched_node)  for the longes block-aligned prefix"""
        matched_blocks:list[int]=[]
        node=self.root
        idx=0
        tick=self._next_tick()

        while idx+self.block_size <=len(token_ids):
            block_key=tuple(token_ids[idx:idx+self.block_size])
            if block_key not in node.children:
                break
            child=node.children[block_key]
            child.last_access=tick
            #Count matching token along this edge

            common =0
            limit =min(len(child.tokens),len(token_ids)-idx)
            while common <limit and child.tokens[common] ==token_ids[idx+common]:
                common +=1
            #Only reuse complete physical blocks so divergent tokens don't collide
            matched_num_blocks= common//self.block_size
            if matched_num_blocks==0:
                break
            matched_blocks.extend(child.block_ids[:matched_num_blocks])
            node=child
            if matched_num_blocks <len(child.block_ids):
                break
            idx += len(child.tokens)
        return matched_blocks,node    
    def _split_node(self,node:RadixNode,split_blocks:int) ->RadixNode:

        """ Split 'node' after 'split_blocks' blocks, return the new upper prefix node."""            

        split_tok=split_blocks*self.block_size
        new_node=RadixNode(
            tokens=node.tokens[:split_tok],
            block_ids=node.block_ids[:split_blocks],
            parent = node.parent,
            last_access= node.last_access,
        )
        new_node.lock_ref=node.lock_ref

        #Shrink existing node into the Suffix child so its own child stay attached
        node.tokens=node.tokens[split_tok:]
        node.block_ids=node.block_ids[split_blocks:]
        node.parent=new_node

        new_node.children[node.tokens[:self.block_size]] =node
        if new_node.parent is not None:
           new_node.parent.children[new_node.tokens[:self.block_size]] = new_node
        return new_node

    def insert (self,token_ids:list[int], block_ids:list[int])-> int:
        """Insert full blocks for token_ids; return number of newly cache blocks."""
        num_full_blocks= min(len(token_ids)//self.block_size,len(block_ids))
        if num_full_blocks ==0:
            return 0

        tokens= tuple(token_ids[: num_full_blocks* self.block_size]) 
        blocks=list(block_ids[:num_full_blocks])
        node= self.root
        idx,block_idx=0,0
        tick=self._next_tick()

        while block_idx<num_full_blocks:
            block_key=tokens[idx:idx+self.block_size]
            if block_key not in node.children:
                break
            child=node.children[block_key]
            child.last_access=tick
            common=0
            limit=min(len(child.tokens), len(tokens)-idx)
            while common< limit and child.tokens[common]==tokens[idx+common]:
                common+=1
            common_blocks=common//self.block_size

            if common_blocks<len(child.block_ids):
                child=self._split_node(child, common_blocks)

            idx += common_blocks*self.block_size
            block_idx +=common_blocks
            node=child
        added_blocks= blocks[block_idx:]
        if added_blocks:
            leaf=RadixNode(
                tokens=tokens[idx:],block_ids=added_blocks,parent=node, last_access=tick)
            node.children[leaf.tokens[:self.block_size]]=leaf
            for b in added_blocks:
                self.manager.ref_block(b)
        return len(added_blocks)        
    
    def lock_node(self, node :RadixNode,delta:int =1) -> None:
        """ Adjust lock_ref from 'node' up to root (+1 to pint , -1 to unpin)."""
        while node is not self.root and node is not None:
            node.lock_ref +=delta 
            node = node.parent
    def _collect_leaves(self,node:RadixNode, leaves:list[RadixNode])  ->None:
        """ Collect all unlocked leaf nodes eligible for LRU eviction."""
        if node is not self.root and not node.children:
            if node.lock_ref== 0:
                leaves.append(node)
            return
        for child in node.children.values():
            self._collect_leaves(child,leaves)
    def evict_lru(self,num_blocks_needed:int) ->int:
        """ Evict coldest unlocked leaves via Min Head Untill 'num_blocks_needed' Are freed. """

        if num_blocks_needed <=0:
            return 0
        leaves:list[RadixNode]=[]
        self._collect_leaves(self.root,leaves)

        #Tuple (last_access, id(leaf, leaf) prevents comparing RadixNode objects on tick ties) 
        heap: list[tuple [int,int,RadixNode]]=[
            (leaf.last_access,id(leaf),leaf) for leaf in leaves
        ]   
        heapq.heapify(heap)

        evicted_blocks=0
        while evicted_blocks<num_blocks_needed and heap:
            _,_, leaf = heapq.heappop(heap)
            for b in leaf.block_ids:
                self.manager.free_block(b)
            evicted_blocks += len(leaf.block_ids)

            parent = leaf.parent
            if parent is not None:
                parent.children.pop(leaf.tokens[:self.block_size],None)
                # IF parent now has 0 children is unlocked ,it becomes an evictable leaf
                if parent is not self.root and not parent.children and parent.lock_ref ==0:
                    heapq.heappush(heap,(parent.last_access,id(parent),parent))
        return evicted_blocks          


        



        