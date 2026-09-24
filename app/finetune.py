# app/finetune.py
import logging
import os
import time
import threading

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.adapters import ADAPTERS_DIR, adapter_path
from app.schemas import FinetuneRequest
from data.sft import IGNORE_INDEX, SFTDataset, collate_sft
from model.gpt import GPTModel
from model.lora import LORA_TARGET_MODULES, inject_lora, mark_only_lora_as_trainable, get_lora_state_dict, strip_lora_wrapper_keys
from model.tokenizer import GPT2Tokenizer

TEACH_LORA_R = 4
TEACH_LORA_ALPHA = 8.0

logger = logging.getLogger(__name__)

class FinetuneJobError(Exception):
    """A failure whose message is safe to show to the client."""


def run_lora_finetune_job(job_state: dict, req: FinetuneRequest, base_engine, lock: threading.Lock, engine_gate=None):
    """Background thread to fine-tune the model using LoRA."""
    try:
        logger.info(f"Starting LoRA fine-tuning job for adapter: {req.adapter_name}")

        device = base_engine.device

        model = GPTModel(base_engine.model_config)
        # Snapshot the base weights while holding the engine gate: requests
        # swap LoRA wrappers in and out of base_engine.model, and reading its
        # state_dict mid-swap would copy a half-rewired module tree.
        # base_engine.model may have an adapter applied, whose weights are
        # saved under wrapped key names (`...W_query.linear.weight`);
        # strip_lora_wrapper_keys maps those back so the base weights actually
        # load instead of silently staying at random init under strict=False.
        if engine_gate is not None and not engine_gate.acquire(timeout=120.0):
            raise FinetuneJobError("The model was busy for too long; please retry.")
        try:
            base_state = strip_lora_wrapper_keys(base_engine.model.state_dict())
            missing, unexpected = model.load_state_dict(base_state, strict=False)
        finally:
            if engine_gate is not None:
                engine_gate.release()
        if missing:
            raise RuntimeError(f"Fine-tune base model load left params unmatched: {missing}")

        inject_lora(model, r=TEACH_LORA_R, alpha=TEACH_LORA_ALPHA, target_modules=LORA_TARGET_MODULES)
        mark_only_lora_as_trainable(model)
        model.to(device)

        tokenizer = GPT2Tokenizer()
        pad_id = tokenizer.eos_id
        # Loss on response tokens only (prompt positions are IGNORE_INDEX).
        dataset = SFTDataset(
            ((ex.instruction, ex.response) for ex in req.examples),
            tokenizer,
            max_length=base_engine.model_config["context_length"],
        )

        if len(dataset) == 0:
            raise FinetuneJobError("All examples were empty or too long to leave room for a response.")

        dataloader = DataLoader(
            dataset,
            batch_size=min(4, len(dataset)),
            shuffle=True,
            collate_fn=lambda b: collate_sft(b, pad_token_id=pad_id)
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
                
                loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=IGNORE_INDEX)
                
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
                    
        checkpoint = {
            "model_config": base_engine.model_config,
            "model_state_dict": {k: v.detach().cpu() for k, v in get_lora_state_dict(model).items()},
            "is_lora": True,
            "lora_r": TEACH_LORA_R,
            "lora_alpha": TEACH_LORA_ALPHA,
        }
        final_path = adapter_path(req.adapter_name)
        os.makedirs(ADAPTERS_DIR, exist_ok=True)
        # Write to a temp file and rename into place, so a crash or a
        # concurrent reader never sees a truncated adapter. Refuse to replace
        # an existing adapter -- the endpoint checks too, but only this check
        # runs after training, right before the write.
        tmp_path = f"{final_path}.{job_state['id']}.tmp"
        torch.save(checkpoint, tmp_path)
        try:
            with lock:
                if os.path.exists(final_path):
                    raise FinetuneJobError("An adapter with that name already exists.")
                os.replace(tmp_path, final_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        with lock:
            job_state["status"] = "done"
            job_state["eta_seconds"] = 0.0

        logger.info(f"Successfully finished finetuning and saved adapter to {final_path}")

    except FinetuneJobError as e:
        logger.warning(f"Finetuning job {job_state.get('id')} rejected: {e}")
        with lock:
            job_state["status"] = "failed"
            job_state["error"] = str(e)
    except Exception:
        logger.exception(f"Finetuning job {job_state.get('id')} failed")
        with lock:
            job_state["status"] = "failed"
            job_state["error"] = "Training failed due to a server error."
