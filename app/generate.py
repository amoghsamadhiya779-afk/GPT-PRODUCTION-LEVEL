# app/generate.py
"""Interactive text generation CLI for GPT-2.

Load a trained model checkpoint and generate text with configurable
temperature, top-k sampling, and maximum token count.

Usage:
    py app/generate.py --prompt "Once upon a time"
    py app/generate.py --checkpoint checkpoints/best_model.pt --temperature 0.8 --top-k 40
"""

import argparse
import os
import sys
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import GPTConfig
from model.gpt import GPTModel, generate, count_parameters
from model.tokenizer import GPT2Tokenizer


def load_model(checkpoint_path: str, device: torch.device) -> GPTModel:
    """Load a GPT model from a training checkpoint, handling LoRA if present."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = checkpoint["model_config"]
    model = GPTModel(cfg)
    
    is_lora = checkpoint.get("is_lora", False)
    if is_lora:
        from model.lora import inject_lora
        lora_r = checkpoint.get("lora_r")
        lora_alpha = checkpoint.get("lora_alpha")
        if lora_r is None:
            # Find any lora_A key to check its shape
            lora_A_keys = [k for k in checkpoint["model_state_dict"].keys() if "lora_A" in k]
            if lora_A_keys:
                lora_r = checkpoint["model_state_dict"][lora_A_keys[0]].shape[0]
            else:
                lora_r = 4
        if lora_alpha is None:
            lora_alpha = float(lora_r * 2)
            
        inject_lora(model, r=lora_r, alpha=lora_alpha, target_modules=["W_query", "W_value"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        
    model.to(device)
    model.eval()
    return model


def main() -> None:
    # Ensure stdout and stderr use UTF-8 encoding to avoid Windows encoding crashes
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="Generate text with a trained GPT-2 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "best_model.pt"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="Every effort moves you",
        help="Text prompt to seed generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.0 = greedy, higher = more random).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (only sample from top k tokens).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: cpu, cuda, or auto.",
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Header
    print("\n" + "=" * 60)
    print("  GPT-2 Text Generator")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Top-k       : {args.top_k}")
    print(f"  Max tokens  : {args.max_tokens}")
    print("=" * 60 + "\n")

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"  [ERROR] Checkpoint not found: {args.checkpoint}")
        print(f"  Train a model first: py training/train.py")
        sys.exit(1)

    # Load model
    print("  Loading model...")
    model = load_model(args.checkpoint, device)
    n_params = count_parameters(model)
    print(f"  Parameters  : {n_params:,} ({n_params / 1e6:.1f}M)")

    # Tokenize prompt
    tokenizer = GPT2Tokenizer()
    input_ids = tokenizer.text_to_token_ids(args.prompt).to(device)
    context_size = model.pos_emb.weight.shape[0]

    # Generate
    print(f"\n  Prompt: \"{args.prompt}\"\n")
    print("-" * 60)

    torch.manual_seed(42)
    start_time = time.time()

    output_ids = generate(
        model=model,
        idx=input_ids,
        max_new_tokens=args.max_tokens,
        context_size=context_size,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    elapsed = time.time() - start_time
    generated_text = tokenizer.token_ids_to_text(output_ids)
    num_generated = output_ids.shape[1] - input_ids.shape[1]

    print(f"\n{generated_text}")
    print("\n" + "-" * 60)
    print(f"  Generated {num_generated} tokens in {elapsed:.2f}s")
    print(f"  Speed: {num_generated / elapsed:.1f} tokens/sec")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
