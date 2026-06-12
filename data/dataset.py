# data/dataset.py
"""GPT training dataset and DataLoader utilities.

Implements a sliding-window dataset that creates (input, target) pairs
for next-token prediction, where targets are shifted by one position.
"""

import tiktoken
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset


class GPTDataset(Dataset):
    """Sliding-window dataset for GPT pre-training.

    Tokenizes the input text and creates overlapping sequences of
    (input, target) pairs where the target is shifted by one token.

    Args:
        txt: Raw text string to tokenize and chunk.
        tokenizer: tiktoken tokenizer instance.
        max_length: Maximum sequence length per sample.
        stride: Number of tokens to advance the window between samples.
    """

    def __init__(
        self,
        txt: str,
        tokenizer,
        max_length: int,
        stride: int,
    ) -> None:
        self.input_ids: list[Tensor] = []
        self.target_ids: list[Tensor] = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Create sliding window chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for GPT pre-training.

    Args:
        txt: Raw text string.
        batch_size: Number of sequences per batch.
        max_length: Maximum sequence length.
        stride: Window stride (set equal to max_length for no overlap).
        shuffle: Whether to shuffle the dataset.
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Configured DataLoader instance.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


class StreamedTextbookDataset(IterableDataset):
    """Iterable Dataset for streaming Hugging Face datasets on-the-fly.

    Streams educational textbook datasets (auto_math_text, khanacademy, openstax)
    from HuggingFaceTB/cosmopedia and merges them with awesome chatbot prompts
    from fka/prompts.chat without writing raw data files to local disk.
    """

    def __init__(self, tokenizer, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        from datasets import load_dataset
        import pandas as pd

        token_buffer = []
        subsets = ["auto_math_text", "khanacademy", "openstax"]
        eos_token_id = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        # 1. Stream Cosmopedia math subsets
        for subset in subsets:
            try:
                ds = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)
                for row in ds:
                    text = row.get("text", "").strip()
                    if text:
                        tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"}) + [eos_token_id]
                        token_buffer.extend(tokens)

                        while len(token_buffer) >= self.max_length + 1:
                            inputs = token_buffer[:self.max_length]
                            targets = token_buffer[1 : self.max_length + 1]
                            token_buffer = token_buffer[self.max_length:]
                            yield torch.tensor(inputs), torch.tensor(targets)
            except Exception:
                # Silently skip if loading errors occur
                pass

        # 2. Stream prompts from fka/prompts.chat using pandas hf:// protocol
        try:
            df = pd.read_csv("hf://datasets/fka/prompts.chat/prompts.csv")
            for _, row in df.iterrows():
                act = str(row.get("act", "")).strip()
                prompt = str(row.get("prompt", "")).strip()
                if act and prompt:
                    text = f"Persona: {act}\nPrompt: {prompt}"
                    tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"}) + [eos_token_id]
                    token_buffer.extend(tokens)

                    while len(token_buffer) >= self.max_length + 1:
                        inputs = token_buffer[:self.max_length]
                        targets = token_buffer[1 : self.max_length + 1]
                        token_buffer = token_buffer[self.max_length:]
                        yield torch.tensor(inputs), torch.tensor(targets)
        except Exception:
            pass


class SplitStreamedDataset(IterableDataset):
    """Splits a streamed IterableDataset into train/validation on-the-fly."""

    def __init__(self, base_dataset: IterableDataset, split: str = "train", split_ratio: float = 0.9) -> None:
        self.base_dataset = base_dataset
        self.split = split
        self.split_ratio = split_ratio

    def __iter__(self):
        count = 0
        for inputs, targets in self.base_dataset:
            # Modulus split routing
            is_train = (count % 10 < int(self.split_ratio * 10))
            if (self.split == "train" and is_train) or (self.split == "val" and not is_train):
                yield inputs, targets
            count += 1


def create_streamed_dataloader(
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    split: str = "train",
    split_ratio: float = 0.9,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader that streams and splits the Cosmopedia & Prompts dataset."""
    base_dataset = StreamedTextbookDataset(tokenizer, max_length=max_length)
    split_dataset = SplitStreamedDataset(base_dataset, split=split, split_ratio=split_ratio)
    dataloader = DataLoader(
        split_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return dataloader


class InstructionDataset(Dataset):
    """Dataset for instruction finetuning with target loss masking.

    Loads a JSON file containing instruction-response pairs (Alpaca style),
    formats them into structured prompts, tokenizes them, and sets up
    target labels where prompt tokens are masked with -100 to prevent computing gradients.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 256,
        ignore_index: int = -100,
    ) -> None:
        self.input_ids: list[Tensor] = []
        self.labels: list[Tensor] = []

        import json
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        eos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            # Format prompt text
            if input_text:
                prompt = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
                )
            else:
                prompt = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n### Response:\n"
                )

            # Tokenize prompt and response
            prompt_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
            response_ids = tokenizer.encode(output, allowed_special={"<|endoftext|>"}) + [eos_token_id]

            # Form full input sequence
            input_sequence = prompt_ids + response_ids

            # Truncate if it exceeds max_length
            if len(input_sequence) > max_length:
                input_sequence = input_sequence[:max_length]

            # Pad if shorter than max_length
            padding_len = max_length - len(input_sequence)
            input_sequence_padded = input_sequence + [eos_token_id] * padding_len

            # Calculate target labels (shifted by 1 for next-token prediction)
            target = []
            for i in range(len(input_sequence_padded) - 1):
                next_token = input_sequence_padded[i + 1]
                # Mask out prompt and padding tokens with ignore_index (-100)
                if i + 1 < len(prompt_ids):
                    target.append(ignore_index)
                elif i + 1 < len(input_sequence):
                    target.append(next_token)
                else:
                    target.append(ignore_index)

            # Input is sequence[:-1], target is sequence[1:]
            self.input_ids.append(torch.tensor(input_sequence_padded[:-1]))
            self.labels.append(torch.tensor(target))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.labels[idx]


def create_instruction_dataloader(
    data_path: str,
    batch_size: int = 4,
    max_length: int = 256,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for instruction finetuning."""
    from model.tokenizer import GPT2Tokenizer
    tokenizer = GPT2Tokenizer()
    dataset = InstructionDataset(data_path, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
