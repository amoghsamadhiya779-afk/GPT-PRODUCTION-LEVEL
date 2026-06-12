# training/train_classifier.py
"""GPT sequence classification training and fine-tuning pipeline.

Supports:
    - Training custom GPTClassificationModel (with LoRA, head-only freezing, or full finetuning)
    - Fine-tuning Hugging Face sequence classification models (with PEFT/QLoRA)
    - Gradient accumulation and AMP mixed precision
    - MLflow experiment tracking
"""

import argparse
import logging
import os
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mlflow

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import GPTConfig, TrainingConfig, load_config_from_yaml
from model.gpt import GPTModel
from model.tokenizer import GPT2Tokenizer
from model.classification import GPTClassificationModel
from model.lora import inject_lora
from data.classification_dataset import create_classification_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_hf: bool = False,
) -> tuple[float, float]:
    """Evaluate sequence classification model loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_hf:
                outputs = model(inputs)
                logits = outputs.logits
            else:
                logits = model(inputs)

            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    model.train()
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train_classifier(
    model_cfg: GPTConfig,
    train_cfg: TrainingConfig,
    data_path: str,
    checkpoint_path: str | None = None,
    use_lora: bool = False,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    freeze_base: bool = False,
    use_hf: bool = False,
    hf_model: str = "deepseek-ai/deepseek-llm-7b-chat",
    use_amp: bool = False,
    accum_steps: int = 1,
) -> None:
    """Run sequence classification fine-tuning."""
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Prepare Datasets (90/10 split)
    logger.info("Parsing CSV dataset from: %s", data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    
    if len(reader) < 2:
        raise ValueError("Dataset is too small for classification splitting.")

    train_ratio = 0.90
    split_idx = int(train_ratio * len(reader))
    train_rows = reader[:split_idx]
    val_rows = reader[split_idx:]

    os.makedirs("tests", exist_ok=True)
    temp_train_path = "tests/temp_class_train.csv"
    temp_val_path = "tests/temp_class_val.csv"

    # Write split datasets to temp files
    fieldnames = reader[0].keys()
    with open(temp_train_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)
    with open(temp_val_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)

    # 2. Setup Tokenizer & Model
    if use_hf:
        logger.info("Loading Hugging Face model/tokenizer for: %s", hf_model)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load Sequence Classification Model (3 classes)
        # In case of large models like DeepSeek, QLoRA is required
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model,
            num_labels=3,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Wrap with PEFT LoRA if requested
        if use_lora:
            logger.info("Wrapping HF model with PEFT LoRA...")
            from peft import LoraConfig, get_peft_model, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "W_query", "W_value"],  # covers both deepseek & custom
            )
            model = get_peft_model(model, peft_config)
    else:
        # Load custom GPT classifier
        tokenizer = GPT2Tokenizer()
        cfg_dict = model_cfg.to_dict()

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info("Loading base GPT weights from checkpoint: %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            cfg_dict = checkpoint["model_config"]
            model_cfg = GPTConfig(**cfg_dict)
            base_model = GPTModel(cfg_dict)
            base_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            logger.info("Initializing base GPT model from scratch.")
            base_model = GPTModel(cfg_dict)

        # Inject LoRA into base attention layers if requested
        if use_lora:
            logger.info("Injecting LoRA adapters (rank=%d) into custom attention blocks...", lora_r)
            inject_lora(base_model, r=lora_r, alpha=lora_alpha, target_modules=["W_query", "W_value"])

        model = GPTClassificationModel(base_model, num_classes=3)
        model.to(device)

        # Freeze base parameters if requested
        if freeze_base:
            logger.info("Freezing base GPT model parameters. Only training classifier head.")
            for name, param in model.named_parameters():
                if "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif use_lora:
            # When LoRA is used on base model, classifier head parameters must still be trainable
            logger.info("LoRA enabled. Ensuring classifier head and LoRA parameters are trainable.")
            for name, param in model.named_parameters():
                if "lora_" in name or "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    # 3. Create Dataloaders
    train_loader = create_classification_dataloader(
        temp_train_path,
        tokenizer=tokenizer,
        batch_size=train_cfg.batch_size,
        max_length=model_cfg.context_length if not use_hf else 512,
        shuffle=True,
        is_hf_tokenizer=use_hf,
    )
    val_loader = create_classification_dataloader(
        temp_val_path,
        tokenizer=tokenizer,
        batch_size=train_cfg.batch_size,
        max_length=model_cfg.context_length if not use_hf else 512,
        shuffle=False,
        is_hf_tokenizer=use_hf,
    )

    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ─── Optimizer & Scheduler ───────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    # MLflow Setup
    mlflow.set_experiment("GPT Classifier Finetuning")
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.start_run(run_name="classifier-training")
    
    mlflow.log_params({
        "learning_rate": train_cfg.learning_rate,
        "num_epochs": train_cfg.num_epochs,
        "batch_size": train_cfg.batch_size,
        "use_lora": use_lora,
        "lora_r": lora_r if use_lora else 0,
        "freeze_base": freeze_base,
        "use_hf": use_hf,
        "hf_model": hf_model if use_hf else "None",
        "use_amp": use_amp,
    })

    best_val_loss = float("inf")
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    start_time = time.time()

    for epoch in range(train_cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and use_amp)):
                if use_hf:
                    outputs = model(inputs)
                    logits = outputs.logits
                else:
                    logits = model(inputs)
                
                loss = F.cross_entropy(logits, targets)
                # Scale loss by gradient accumulation steps
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if not use_hf:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps * inputs.size(0)

        # Epoch Metrics
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_classifier(model, val_loader, device, is_hf=use_hf)
        
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%%",
            epoch + 1, train_cfg.num_epochs, train_loss, val_loss, val_acc * 100
        )
        
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_file = os.path.join(train_cfg.save_dir, "classifier_best.pt")
            logger.info("Saving best model checkpoint to %s", checkpoint_file)
            
            # Extract state dict (just LoRA/Classifier weights if frozen/PEFT)
            if use_hf:
                # Save Hugging Face state dict
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "is_hf": True,
                    "hf_model_name": hf_model,
                }, checkpoint_file)
            else:
                # Save custom model
                state_dict = model.state_dict()
                if freeze_base or use_lora:
                    # Save only classifier and LoRA parameters to save disk space
                    state_dict = {k: v for k, v in state_dict.items() if "lora_" in k or "classifier" in k}
                
                torch.save({
                    "model_state_dict": state_dict,
                    "model_config": model_cfg.to_dict(),
                    "is_lora": use_lora,
                    "is_hf": False,
                    "freeze_base": freeze_base,
                }, checkpoint_file)

    logger.info("Training complete in %.2f seconds.", time.time() - start_time)
    mlflow.end_run()

    # Clean up temp split files
    for path in [temp_train_path, temp_val_path]:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT / DeepSeek for Sequence Classification")
    parser.add_argument("--config", type=str, default="configs/gpt2_tiny.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, required=True, help="Path to train.csv dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Base model checkpoint path")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA adaptation")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA alpha")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze base model weights")
    parser.add_argument("--use_hf", action="store_true", help="Train Hugging Face sequence classification model")
    parser.add_argument("--hf_model", type=str, default="deepseek-ai/deepseek-llm-7b-chat", help="Hugging Face model ID")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    
    args = parser.parse_args()

    model_cfg, train_cfg = load_config_from_yaml(args.config)
    
    # Overrides
    if args.epochs is not None:
        train_cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr

    train_classifier(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        freeze_base=args.freeze_base,
        use_hf=args.use_hf,
        hf_model=args.hf_model,
        use_amp=args.use_amp,
        accum_steps=args.accum_steps,
    )
