# app/finetune.py
import logging
import os
import time
import copy
import threading

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from app.schemas import FinetuneRequest
from model.gpt import GPTModel
from model.lora import inject_lora, mark_only_lora_as_trainable, get_lora_state_dict, strip_lora_wrapper_keys
from model.tokenizer import GPT2Tokenizer

logger = logging.getLogger(__name__)

class AlpacaDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.input_ids = []
        self.target_ids = []
        
        for ex in examples:
            text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{ex.instruction}\n\n"
                f"### Response:\n{ex.response}<|endoftext|>"
            )
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

def collate_fn(batch, pad_token_id=50256):
    inputs, targets = zip(*batch)
    
    max_len_in_batch = max([x.size(0) for x in inputs])
    
    padded_inputs = []
    padded_targets = []
    
    for i in range(len(inputs)):
        inp = inputs[i]
        tgt = targets[i]
        
        pad_len = max_len_in_batch - inp.size(0)
        
        if pad_len > 0:
            inp = F.pad(inp, (0, pad_len), value=pad_token_id)
            tgt = F.pad(tgt, (0, pad_len), value=-100)
            
        padded_inputs.append(inp)
        padded_targets.append(tgt)
        
    return torch.stack(padded_inputs), torch.stack(padded_targets)


def run_lora_finetune_job(job_state: dict, req: FinetuneRequest, base_engine, lock: threading.Lock):
    """Background thread to fine-tune the model using LoRA."""
    try:
        logger.info(f"Starting LoRA fine-tuning job for adapter: {req.adapter_name}")

        device = base_engine.device

        model = GPTModel(base_engine.model_config)
        # base_engine.model may currently have LoRA layers injected (e.g. an
        # active persona/adapter), whose weights are saved under wrapped key
        # names (`...W_query.linear.weight`). strip_lora_wrapper_keys maps
        # those back to plain names so the base weights actually load instead
        # of silently staying at random init under strict=False.
        base_state = strip_lora_wrapper_keys(base_engine.model.state_dict())
        missing, unexpected = model.load_state_dict(base_state, strict=False)
        if missing:
            logger.warning(f"Fine-tune base model load left params unmatched (random init): {missing}")
        model.to(device)
        
        inject_lora(model, r=4, alpha=8.0, target_modules=["W_query", "W_value"])
        mark_only_lora_as_trainable(model)
        
        tokenizer = GPT2Tokenizer()
        pad_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        dataset = AlpacaDataset(req.examples, tokenizer, max_length=base_engine.model_config["context_length"])
        
        if len(dataset) == 0:
            raise ValueError("All examples were empty or invalid.")
            
        dataloader = DataLoader(
            dataset, 
            batch_size=min(4, len(dataset)), 
            shuffle=True, 
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id)
        )
        
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=req.lr,
            weight_decay=0.01
        )
        
        model.train()
        total_steps = req.steps
        
        with lock:
            job_state["status"] = "running"
            job_state["total_steps"] = total_steps
            job_state["step"] = 0
            
        start_time = time.time()
        step_times = []
        
        current_step = 0
        while current_step < total_steps:
            for input_batch, target_batch in dataloader:
                if current_step >= total_steps:
                    break
                    
                step_start = time.time()
                
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                
                optimizer.zero_grad()
                logits = model(input_batch)
                
                loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100)
                
                loss.backward()
                optimizer.step()
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                avg_step_time = sum(step_times[-10:]) / len(step_times[-10:])
                eta = avg_step_time * (total_steps - current_step)
                
                current_step += 1
                
                with lock:
                    job_state["step"] = current_step
                    job_state["current_loss"] = loss.item()
                    job_state["eta_seconds"] = eta
                    
        os.makedirs("checkpoints/adapters", exist_ok=True)
        adapter_path = os.path.join("checkpoints", "adapters", f"{req.adapter_name}.pt")
        
        checkpoint = {
            "model_config": base_engine.model_config,
            "model_state_dict": get_lora_state_dict(model),
            "is_lora": True,
            "lora_r": 4,
            "lora_alpha": 8.0,
        }
        torch.save(checkpoint, adapter_path)
        
        with lock:
            job_state["status"] = "done"
            job_state["eta_seconds"] = 0.0
            
        logger.info(f"Successfully finished finetuning and saved adapter to {adapter_path}")
        
    except Exception as e:
        logger.error(f"Finetuning job failed: {e}")
        with lock:
            job_state["status"] = "failed"
