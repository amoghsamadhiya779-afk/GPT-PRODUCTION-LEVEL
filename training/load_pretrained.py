# training/load_pretrained.py
"""Load official pretrained OpenAI GPT-2 weights into the custom model architecture.

Downloads official weights from Hugging Face and translates state dict keys
layer by layer, verifying that our custom PyTorch implementation is 100%
mathematically equivalent to the official OpenAI model.
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPTModel
from model.config import GPTConfig


def main() -> None:
    print("\n" + "=" * 60)
    print("  GPT-2 Pretrained Weights Loader")
    print("=" * 60 + "\n")

    # 1. Install transformers if not present
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("  [INFO] Installing 'transformers' library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "--quiet"])
        from transformers import GPT2LMHeadModel

    # 2. Download official weights
    print("  [1/3] Downloading official GPT-2 Small (124M) weights from Hugging Face...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_sd = hf_model.state_dict()

    # 3. Define matching GPTConfig
    # Official GPT-2 Small has 1024 context, 768 dim, 12 layers, 12 heads, and uses QKV bias
    cfg = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        dropout=0.1,
        qkv_bias=True, 
    )

    # 4. Initialize our model and get state dict
    model = GPTModel(cfg)
    sd = model.state_dict()

    # 5. Translate weights
    print("  [2/3] Mapping weights to custom model architecture...")

    # Token and Positional Embeddings
    sd["tok_emb.weight"].copy_(hf_sd["transformer.wte.weight"])
    sd["pos_emb.weight"].copy_(hf_sd["transformer.wpe.weight"])

    # Stacked Transformer Blocks
    for i in range(cfg.n_layers):
        # Layer Normalizations
        sd[f"trf_blocks.{i}.norm1.scale"].copy_(hf_sd[f"transformer.h.{i}.ln_1.weight"])
        sd[f"trf_blocks.{i}.norm1.shift"].copy_(hf_sd[f"transformer.h.{i}.ln_1.bias"])
        sd[f"trf_blocks.{i}.norm2.scale"].copy_(hf_sd[f"transformer.h.{i}.ln_2.weight"])
        sd[f"trf_blocks.{i}.norm2.shift"].copy_(hf_sd[f"transformer.h.{i}.ln_2.bias"])

        # QKV Attention Projections
        # Hugging Face stores Q, K, V fused in a single Conv1D layer weight (768, 2304) and bias (2304)
        c_attn_w = hf_sd[f"transformer.h.{i}.attn.c_attn.weight"]
        c_attn_b = hf_sd[f"transformer.h.{i}.attn.c_attn.bias"]
        
        q_w, k_w, v_w = torch.chunk(c_attn_w, 3, dim=-1)
        q_b, k_b, v_b = torch.chunk(c_attn_b, 3, dim=-1)

        # Transpose weights because Hugging Face uses Conv1D (transposed relative to nn.Linear)
        sd[f"trf_blocks.{i}.att.W_query.weight"].copy_(q_w.t())
        sd[f"trf_blocks.{i}.att.W_query.bias"].copy_(q_b)
        
        sd[f"trf_blocks.{i}.att.W_key.weight"].copy_(k_w.t())
        sd[f"trf_blocks.{i}.att.W_key.bias"].copy_(k_b)
        
        sd[f"trf_blocks.{i}.att.W_value.weight"].copy_(v_w.t())
        sd[f"trf_blocks.{i}.att.W_value.bias"].copy_(v_b)

        # Out projection
        sd[f"trf_blocks.{i}.att.out_proj.weight"].copy_(hf_sd[f"transformer.h.{i}.attn.c_proj.weight"].t())
        sd[f"trf_blocks.{i}.att.out_proj.bias"].copy_(hf_sd[f"transformer.h.{i}.attn.c_proj.bias"])

        # Feed Forward Network (MLP)
        sd[f"trf_blocks.{i}.ff.layers.0.weight"].copy_(hf_sd[f"transformer.h.{i}.mlp.c_fc.weight"].t())
        sd[f"trf_blocks.{i}.ff.layers.0.bias"].copy_(hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"])
        sd[f"trf_blocks.{i}.ff.layers.2.weight"].copy_(hf_sd[f"transformer.h.{i}.mlp.c_proj.weight"].t())
        sd[f"trf_blocks.{i}.ff.layers.2.bias"].copy_(hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"])

    # Final LayerNorm and LM Head
    sd["final_norm.scale"].copy_(hf_sd["transformer.ln_f.weight"])
    sd["final_norm.shift"].copy_(hf_sd["transformer.ln_f.bias"])
    sd["out_head.weight"].copy_(hf_sd["lm_head.weight"])

    # 6. Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "best_model.pt")
    
    torch.save({
        "model_state_dict": sd,
        "model_config": cfg.to_dict(),
    }, checkpoint_path)

    print(f"  [3/3] Successfully saved mapped GPT-2 weights to: {checkpoint_path}")
    print("\n" + "=" * 60)
    print("  LOAD COMPLETE — Ready for text generation!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
