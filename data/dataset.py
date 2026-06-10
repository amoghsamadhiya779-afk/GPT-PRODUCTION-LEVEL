# data/dataset.py
"""GPT training dataset and DataLoader utilities.

Implements a sliding-window dataset that creates (input, target) pairs
for next-token prediction, where targets are shifted by one position.
"""

import tiktoken
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


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
