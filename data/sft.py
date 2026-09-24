# data/sft.py
"""Supervised fine-tuning (SFT) example encoding with prompt-masked labels.

Single source of truth for the Alpaca-style instruction template, shared by
the SFT training script (training/finetune_instruct.py), Teach Mode
(app/finetune.py) and serving (app/api.py) so the prompt a model is trained
on is byte-for-byte the prompt it is served with.

Loss is computed on response tokens only. Supervising the prompt as well
spends most of the training signal on predicting the fixed template and the
user's instruction (two-thirds of all tokens in data/sft_mix.jsonl) instead
of on answering -- and teaches the model to write instructions.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset

# Label value skipped by F.cross_entropy (its default ignore_index).
IGNORE_INDEX = -100

PROMPT_HEADER = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
)


def format_prompt(instruction: str) -> str:
    """The prompt the model sees; the response follows it directly."""
    return f"{PROMPT_HEADER}{instruction}\n\n### Response:\n"


def encode_sft_example(tokenizer, instruction: str, response: str, max_length: int) -> tuple[list[int], list[int]] | None:
    """Encode one example as (input_ids, labels) for next-token training.

    The sequence is prompt + response + EOS; labels are the next token at
    each position, with every position that would predict a prompt token set
    to IGNORE_INDEX. Returns None if truncation to `max_length` input tokens
    leaves no response token to learn from.

    Prompt and response are tokenized separately: the prompt then encodes
    exactly as it does at inference (where it is encoded on its own and the
    model continues from there), rather than letting BPE merge across the
    boundary. Both use encode_ordinary, so a literal "<|endoftext|>" in user
    data stays plain text and only the appended terminator is the real EOS.
    """
    prompt_ids = tokenizer.encode_ordinary(format_prompt(instruction))
    sequence = (prompt_ids + tokenizer.encode_ordinary(response) + [tokenizer.eos_id])[: max_length + 1]

    input_ids = sequence[:-1]
    labels = [
        token if position + 1 >= len(prompt_ids) else IGNORE_INDEX
        for position, token in enumerate(sequence[1:])
    ]
    if all(label == IGNORE_INDEX for label in labels):
        return None
    return input_ids, labels


class SFTDataset(Dataset):
    """Prompt-masked instruction dataset over (instruction, response) pairs."""

    def __init__(self, pairs, tokenizer, max_length: int = 256) -> None:
        self.input_ids: list[Tensor] = []
        self.labels: list[Tensor] = []
        for instruction, response in pairs:
            encoded = encode_sft_example(tokenizer, instruction, response, max_length)
            if encoded is not None:
                self.input_ids.append(torch.tensor(encoded[0]))
                self.labels.append(torch.tensor(encoded[1]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.labels[idx]


def collate_sft(batch, pad_token_id: int) -> tuple[Tensor, Tensor]:
    """Right-pad a batch; padded label positions are ignored by the loss."""
    inputs, labels = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    padded_inputs = torch.full((len(inputs), max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(inputs), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, (inp, lab) in enumerate(zip(inputs, labels)):
        padded_inputs[i, : inp.size(0)] = inp
        padded_labels[i, : lab.size(0)] = lab
    return padded_inputs, padded_labels
