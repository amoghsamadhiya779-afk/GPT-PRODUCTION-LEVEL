# data/classification_dataset.py
"""Dataset loader and utilities for sequence classification (Chatbot Arena Human Preference).

Parses prompt-response CSVs, formats input prompts, performs smart proportional truncation,
and maps target labels (winner_model_a, winner_model_b, winner_tie) to (0, 1, 2).
"""

import os
import csv
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class LLMClassificationDataset(Dataset):
    """Dataset for training a sequence classifier on Chatbot Arena preferences.

    Formats inputs as:
      <|endoftext|>Prompt: {prompt}

      Response A: {response_a}

      Response B: {response_b}
    
    Supports both custom tokenizers (tiktoken wrappers) and Hugging Face tokenizers.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 256,
        is_hf_tokenizer: bool = False,
    ) -> None:
        self.input_ids: list[Tensor] = []
        self.labels: list[int] = []

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        # Try to find pad/eos token from tokenizer
        if is_hf_tokenizer:
            eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        else:
            # Tiktoken GPT-2 BPE special token ID
            try:
                eos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
            except Exception:
                eos_token_id = 50256  # standard fallback for GPT-2

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Read text columns
                prompt = row.get("prompt", "")
                response_a = row.get("response_a", "")
                response_b = row.get("response_b", "")

                # Determine target label
                # winner_model_a, winner_model_b, winner_tie
                w_a = float(row.get("winner_model_a", 0))
                w_b = float(row.get("winner_model_b", 0))
                w_tie = float(row.get("winner_tie", 0))

                if w_a >= 1.0:
                    label = 0
                elif w_b >= 1.0:
                    label = 1
                else:
                    label = 2  # Tie or fallback

                # Format prompts and tokenize parts
                prompt_text = f"Prompt:\n{prompt}\n\nResponse A:\n"
                res_a_text = f"{response_a}\n\nResponse B:\n"
                res_b_text = f"{response_b}"

                if is_hf_tokenizer:
                    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    res_a_ids = tokenizer.encode(res_a_text, add_special_tokens=False)
                    res_b_ids = tokenizer.encode(res_b_text, add_special_tokens=False)
                else:
                    prompt_ids = tokenizer.encode(prompt_text)
                    res_a_ids = tokenizer.encode(res_a_text)
                    res_b_ids = tokenizer.encode(res_b_text)

                # Prepend special token
                prompt_ids = [eos_token_id] + prompt_ids

                # Smart proportional truncation
                total_len = len(prompt_ids) + len(res_a_ids) + len(res_b_ids)
                if total_len > max_length:
                    available = max_length - len(prompt_ids)
                    if available < 20:
                        # Fallback to simple truncation
                        combined = prompt_ids + res_a_ids + res_b_ids
                        input_ids = combined[:max_length]
                    else:
                        half_share = available // 2
                        if len(res_a_ids) > half_share and len(res_b_ids) > half_share:
                            res_a_ids = res_a_ids[:half_share]
                            res_b_ids = res_b_ids[:available - half_share]
                        elif len(res_a_ids) > half_share:
                            res_b_len = len(res_b_ids)
                            res_a_ids = res_a_ids[:available - res_b_len]
                        else:
                            res_a_len = len(res_a_ids)
                            res_b_ids = res_b_ids[:available - res_a_len]
                        input_ids = prompt_ids + res_a_ids + res_b_ids
                else:
                    input_ids = prompt_ids + res_a_ids + res_b_ids

                # Padding if shorter than max_length
                if len(input_ids) < max_length:
                    padding_len = max_length - len(input_ids)
                    # For causal decoder models, we usually pad with eos_token_id
                    input_ids = input_ids + [eos_token_id] * padding_len

                self.input_ids.append(torch.tensor(input_ids))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.input_ids[idx], torch.tensor(self.labels[idx], dtype=torch.long)


def create_classification_dataloader(
    csv_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    shuffle: bool = True,
    drop_last: bool = False,
    is_hf_tokenizer: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for sequence classification training."""
    dataset = LLMClassificationDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_hf_tokenizer=is_hf_tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
