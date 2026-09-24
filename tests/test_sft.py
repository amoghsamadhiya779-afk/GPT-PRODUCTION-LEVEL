# tests/test_sft.py
"""Tests for prompt-masked SFT encoding (data/sft.py)."""

import torch
import torch.nn.functional as F

from app.api import build_prompt_with_budget
from data.sft import IGNORE_INDEX, SFTDataset, collate_sft, encode_sft_example, format_prompt
from model.gpt import GPTModel
from model.tokenizer import GPT2Tokenizer
from training.utils import calc_loss_batch

tok = GPT2Tokenizer()


def test_labels_mask_prompt_and_supervise_only_the_response():
    instruction, response = "What is 2 + 2?", "It is 4."
    input_ids, labels = encode_sft_example(tok, instruction, response, max_length=256)

    prompt_ids = tok.encode_ordinary(format_prompt(instruction))
    response_ids = tok.encode_ordinary(response) + [tok.eos_id]
    assert input_ids == (prompt_ids + response_ids)[:-1]
    assert len(labels) == len(input_ids)

    # Positions predicting prompt tokens are ignored; the last prompt position
    # predicts the first response token, so supervision starts there.
    n_masked = len(prompt_ids) - 1
    assert labels[:n_masked] == [IGNORE_INDEX] * n_masked
    assert labels[n_masked:] == response_ids


def test_loss_is_computed_on_response_tokens_only():
    torch.manual_seed(0)
    model = GPTModel({"vocab_size": 50257, "context_length": 128, "emb_dim": 32, "n_heads": 2,
                      "n_layers": 1, "drop_rate": 0.0, "qkv_bias": False}).eval()
    ds = SFTDataset([("Name a colour.", "Blue.")], tok, max_length=128)
    inputs, labels = collate_sft([ds[0]], pad_token_id=tok.eos_id)

    loss = calc_loss_batch(inputs, labels, model, torch.device("cpu"))

    n_masked = len(tok.encode_ordinary(format_prompt("Name a colour."))) - 1
    with torch.no_grad():
        logits = model(inputs)[0, n_masked:]
    expected = F.cross_entropy(logits, torch.tensor(tok.encode_ordinary("Blue.") + [tok.eos_id]))
    assert torch.allclose(loss, expected)


def test_training_prompt_matches_the_served_prompt():
    """Train/serve skew guard: the tokens a model is trained to continue from
    must be exactly what app/api.py sends it at inference time."""
    instruction = "Explain gravity in one sentence."
    served_prompt, _ = build_prompt_with_budget(instruction, 50, None, 1024)
    assert served_prompt == format_prompt(instruction)

    input_ids, _ = encode_sft_example(tok, instruction, "Mass attracts mass.", max_length=1024)
    served_ids = tok.encode_ordinary(served_prompt)
    assert input_ids[: len(served_ids)] == served_ids


def test_endoftext_text_in_user_data_is_not_an_eos_token():
    input_ids, labels = encode_sft_example(tok, "say <|endoftext|>", "ok <|endoftext|> done", max_length=256)
    supervised = [t for t in labels if t != IGNORE_INDEX]
    assert tok.eos_id not in input_ids
    assert supervised.count(tok.eos_id) == 1 and supervised[-1] == tok.eos_id


def test_truncation_keeps_context_and_drops_examples_without_response_tokens():
    long_response = "word " * 500
    input_ids, labels = encode_sft_example(tok, "Write a lot.", long_response, max_length=64)
    assert len(input_ids) == len(labels) == 64
    assert any(label != IGNORE_INDEX for label in labels)

    # A prompt that alone fills the window leaves nothing to learn from.
    assert encode_sft_example(tok, "instruction " * 200, "answer", max_length=64) is None
    assert len(SFTDataset([("instruction " * 200, "answer"), ("hi", "hello")], tok, max_length=64)) == 1


def test_collate_pads_inputs_and_ignores_padded_labels():
    ds = SFTDataset([("hi", "hello"), ("Tell me a longer story please.", "Once upon a time there was a model.")], tok)
    inputs, labels = collate_sft([ds[0], ds[1]], pad_token_id=tok.eos_id)
    assert inputs.shape == labels.shape
    short_len = ds[0][0].size(0)
    assert torch.all(inputs[0, short_len:] == tok.eos_id)
    assert torch.all(labels[0, short_len:] == IGNORE_INDEX)
    assert torch.equal(labels[1], ds[1][1])
