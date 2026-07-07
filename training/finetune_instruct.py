import argparse
import os
import sys
import time
import math
import logging
import json
import csv
import torch
import torch.nn.functional as F
import mlflow
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from model.gpt import GPTModel, count_parameters
from model.tokenizer import GPT2Tokenizer
from model.lora import inject_lora, mark_only_lora_as_trainable, get_lora_state_dict
from app.inference import GPTInferenceEngine
from training.utils import calc_loss_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

EVAL_PROMPTS = [
    "what is gravity",
    "who are you",
    "hi",
    "write a haiku about nature",
    "explain what a neural network is",
    "write a joke",
    "what is the capital of France",
    "summarize Cinderella in one sentence"
]

def format_alpaca(instruction: str, response: str) -> str:
    """Matches the byte-exact Alpaca template used in app/api.py /generate."""
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}<|endoftext|>"
    )

class InstructDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.input_ids = []
        self.target_ids = []
        
        for ex in examples:
            text = format_alpaca(ex["instruction"], ex["response"])
            token_ids = tokenizer.text_to_token_ids(text).squeeze(0)
            
            if token_ids.size(0) > max_length:
                token_ids = token_ids[:max_length]
                
            if token_ids.size(0) > 1:
                self.input_ids.append(token_ids[:-1])
                self.target_ids.append(token_ids[1:])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def collate_fn_instruct(batch, pad_token_id):
    inputs, targets = zip(*batch)
    max_len = max([x.size(0) for x in inputs])
    
    padded_inputs = []
    padded_targets = []
    
    for i in range(len(inputs)):
        inp = inputs[i]
        tgt = targets[i]
        pad_len = max_len - inp.size(0)
        
        if pad_len > 0:
            inp = F.pad(inp, (0, pad_len), value=pad_token_id)
            tgt = F.pad(tgt, (0, pad_len), value=-100)
            
        padded_inputs.append(inp)
        padded_targets.append(tgt)
        
    return torch.stack(padded_inputs), torch.stack(padded_targets)

def get_lr(step: int, max_steps: int, max_lr: float, min_lr: float = 1e-6) -> float:
    warmup_steps = int(0.05 * max_steps)
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def evaluate_generation(model, tokenizer, device, context_size):
    model.eval()
    logger.info("--- EVALUATION ---")
    for prompt in EVAL_PROMPTS:
        template = format_alpaca(prompt, "").replace("<|endoftext|>", "")
        input_ids = tokenizer.text_to_token_ids(template).to(device)
        
        with torch.no_grad():
            from model.gpt import generate
            output_ids = generate(
                model=model,
                idx=input_ids,
                max_new_tokens=50,
                context_size=context_size,
                temperature=1.0,
                top_k=1, # greedy
                use_cache=True
            )
        
        generated_text = tokenizer.token_ids_to_text(output_ids)
        if "### Response:\n" in generated_text:
            ans = generated_text.split("### Response:\n")[-1].strip()
        else:
            ans = generated_text.replace(template, "").strip()
            
        if "<|endoftext|>" in ans:
            ans = ans.split("<|endoftext|>")[0]
            
        logger.info(f"Q: {prompt}")
        logger.info(f"A: {ans}")
        logger.info("-" * 40)
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda)")
    parser.add_argument("--data", type=str, default="data/sft_mix.jsonl", help="Path to local dataset file.")
    parser.add_argument("--model-size", type=str, default="small", help="Base model size (small, medium, tiny).")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimization steps.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest adapter checkpoint if exists.")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    tokenizer = GPT2Tokenizer()
    pad_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    checkpoint_dir = "checkpoints" if args.model_size != "tiny" else "checkpoints_tiny"
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Base checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    
    if model_config.get("model_size", "small") != args.model_size:
        logger.warning(f"Warning: model_size config in checkpoint ({model_config.get('model_size')}) does not match --model-size {args.model_size}")
        
    max_length = model_config.get("context_length", 256)

    examples = []
    
    for data_path in args.data.split(","):
        data_path = data_path.strip()
        if not data_path: continue
        logger.info(f"Loading local dataset from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                examples.append(item)

    seen = set()
    clean_examples = []
    for ex in examples:
        pair_hash = (ex["instruction"].strip(), ex["response"].strip())
        if pair_hash in seen:
            continue
        
        full_text = format_alpaca(ex["instruction"], ex["response"])
        tokens = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
        
        if len(tokens) <= max_length:
            seen.add(pair_hash)
            clean_examples.append(ex)
            
    examples = clean_examples
    if not examples:
        logger.error("No valid examples found after filtering!")
        sys.exit(1)
        
    logger.info(f"Dataset Stats: {len(examples)} instruction pairs")
    
    train_ds = InstructDataset(examples, tokenizer, max_length=max_length)
    collate = lambda b: collate_fn_instruct(b, pad_id)

    logger.info(f"Loading base model from {checkpoint_path}")
    gpt_cfg_dict = model_config.copy()
        
    model = GPTModel(gpt_cfg_dict)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    logger.info("=== EVALUATION (BEFORE LORA) ===")
    evaluate_generation(model, tokenizer, device, max_length)

    logger.info("Injecting LoRA adapters (r=16, alpha=32) into W_query and W_value...")
    inject_lora(model, r=16, alpha=32.0, target_modules=["W_query", "W_value"])
    mark_only_lora_as_trainable(model)
    model.to(device)

    adapters_dir = os.path.join("checkpoints", "adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    adapter_name = f"sft_v1_{args.model_size}"
    adapter_path = os.path.join(adapters_dir, f"{adapter_name}.pt")
    
    global_step = 0
    if args.resume and os.path.exists(adapter_path):
        logger.info(f"Resuming from checkpoint {adapter_path}")
        lora_ckpt = torch.load(adapter_path, map_location=device, weights_only=True)
        model.load_state_dict(lora_ckpt["model_state_dict"], strict=False)
        if "step" in lora_ckpt:
            global_step = lora_ckpt["step"]
            logger.info(f"Resumed from step {global_step}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if args.model_size == "medium":
        physical_batch_size = 1
    else:
        physical_batch_size = 4
    accum_steps = max(1, 32 // physical_batch_size)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=physical_batch_size, 
        shuffle=True, 
        collate_fn=collate
    )

    epochs = args.epochs
    lr = 1e-4
    total_steps = (len(train_loader) // accum_steps) * epochs

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("LoRA Instruction Tuning")
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.start_run(run_name=adapter_name)
    
    mlflow.log_params({
        "epochs": epochs,
        "effective_batch_size": 32,
        "learning_rate": lr,
        "lora_r": 16,
        "lora_alpha": 32.0,
        "dataset_size": len(train_ds),
        "model_size": args.model_size
    })

    logger.info("Starting training loop...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        for step, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
            else:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                
            loss = loss / accum_steps
            loss.backward()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                current_lr = get_lr(global_step, total_steps, lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                loss_val = loss.item() * accum_steps
                mlflow.log_metric("train_loss", loss_val, step=global_step)
                mlflow.log_metric("lr", current_lr, step=global_step)
                
                if global_step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | Loss: {loss_val:.4f} | LR: {current_lr:.2e}")
                
                global_step += 1
                
                # Resumable checkpointing every 200 steps
                if global_step % 200 == 0:
                    checkpoint_out = {
                        "model_config": model_config,
                        "model_state_dict": get_lora_state_dict(model),
                        "is_lora": True,
                        "lora_r": 16,
                        "lora_alpha": 32.0,
                        "step": global_step
                    }
                    torch.save(checkpoint_out, adapter_path)
                    logger.info(f"Saved resumable checkpoint at step {global_step} to {adapter_path}")
                
                if args.max_steps is not None and global_step >= args.max_steps:
                    logger.info(f"Reached max steps: {args.max_steps}. Stopping early.")
                    break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.1f}s")
    mlflow.end_run()

    logger.info("=== EVALUATION (AFTER LORA) ===")
    evaluate_generation(model, tokenizer, device, max_length)

    logger.info("=== POST-TRAINING LoRA CHECK ===")
    from model.lora import LoRALinear
    lora_modules_found = 0
    total_delta_norm = 0.0
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules_found += 1
            # L2 norm of (lora_B @ lora_A)
            delta = module.lora_B @ module.lora_A
            total_delta_norm += delta.norm().item()
            
    assert lora_modules_found > 0, "No LoRA modules found in the model!"
    logger.info(f"Found {lora_modules_found} LoRA modules. Sum of delta L2 norms: {total_delta_norm:.6f}")
    assert total_delta_norm > 0, "LoRA delta is exactly zero, the adapter was not trained or not applied!"

    checkpoint_out = {
        "model_config": model_config,
        "model_state_dict": get_lora_state_dict(model),
        "is_lora": True,
        "lora_r": 16,
        "lora_alpha": 32.0,
        "step": global_step
    }
    
    torch.save(checkpoint_out, adapter_path)
    logger.info(f"Successfully saved final LoRA adapter to {adapter_path}")

if __name__ == "__main__":
    main()
