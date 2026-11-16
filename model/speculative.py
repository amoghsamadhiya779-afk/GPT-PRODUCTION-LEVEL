# model/speculative.py
"""Speculative Decoding & Rejection Sampling for GPTModel.

Replicates Leviathan et al. (ICML 2023), "Fast Inference from Transformers via
Speculative Decoding", using exact residual rejection sampling and O(1)
tensor-view KV-cache rollback on token rejection.

Core invariant during each speculative iteration:
    Given committed sequence `seq` of length L:
      - `target_kv` and `draft_kv` store cached K/V states for `seq[:, :-1]`
        (exact length L - 1).
      - `seq[:, -1:]` is the single unconsumed token fed as the next query root.
"""

from __future__ import annotations

import torch
from torch import Tensor

from model.gpt import GPTModel


KVCache = list[tuple[Tensor, Tensor]]


def crop_kv_cache(past_key_values: KVCache | None, max_len: int) -> KVCache | None:
    """Truncate cached key/value tensors to `max_len` tokens in O(1) time.

    Returns tensor views sliced along the sequence dimension (`dim=-2`) without
    copying data or recomputing prefix activations.
    """
    if past_key_values is None or max_len <= 0:
        return None
    return [
        (keys[:, :, :max_len, :], values[:, :, :max_len, :])
        for keys, values in past_key_values
    ]


def _logits_to_probs(
    logits: Tensor,
    temperature: float,
    top_k: int | None = None,
) -> Tensor:
    """Convert raw logits of shape (batch, vocab) into a normalized probability distribution."""
    scaled = logits.clone()
    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.shape[-1])
        top_logits, _ = torch.topk(scaled, k, dim=-1)
        min_val = top_logits[:, -1:]
        scaled = torch.where(
            scaled < min_val,
            torch.tensor(float("-inf"), dtype=scaled.dtype, device=scaled.device),
            scaled,
        )

    if temperature <= 0.0:
        # Degenerate one-hot distribution at argmax
        probs = torch.zeros_like(scaled)
        idx = torch.argmax(scaled, dim=-1, keepdim=True)
        probs.scatter_(-1, idx, 1.0)
        return probs

    scaled = scaled / temperature
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    return torch.softmax(scaled, dim=-1)


def sample_residual_distribution(target_probs: Tensor, draft_probs: Tensor) -> Tensor:
    """Sample a replacement token from norm(max(0, p(x) - q(x))).

    Implements Theorem 1 / Algorithm 1 of Leviathan et al. (2023) so that the
    marginal output distribution of speculative decoding matches `target_probs`
    mathematically without bias. Falls back cleanly to `target_probs` if the
    residual mass is numerically zero.
    """
    diff = torch.clamp(target_probs - draft_probs, min=0.0)
    diff_sum = diff.sum(dim=-1, keepdim=True)
    # Guard against numerical underflow when p and q are nearly identical
    residual_probs = torch.where(
        diff_sum > 1e-12,
        diff / torch.clamp(diff_sum, min=1e-12),
        target_probs,
    )
    return torch.multinomial(residual_probs, num_samples=1)


@torch.no_grad()
def speculative_generate(
    target_model: GPTModel,
    draft_model: GPTModel,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
    gamma: int = 4,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> tuple[Tensor, dict]:
    """Generate tokens using speculative decoding with rejection sampling.

    Args:
        target_model: High-capacity verifier GPTModel (runs single batched pass over
            `gamma + 1` tokens per outer step).
        draft_model: Lightweight speculative GPTModel (autoregressively proposes up to
            `gamma` candidate tokens per outer step).
        idx: Prompt token indices of shape `(1, prompt_len)`.
        max_new_tokens: Maximum number of new tokens to generate.
        context_size: Configured maximum context window.
        gamma: Number of speculative draft tokens proposed per step (K).
        temperature: Sampling temperature (`0.0` for exact greedy equivalence).
        top_k: Optional top-k truncation applied consistently to draft & target probs.
        eos_id: Optional end-of-sequence token id to stop generation early.

    Returns:
        Tuple of `(extended_sequence, telemetry_metrics)`.
    """
    if idx.dim() != 2 or idx.shape[0] != 1:
        raise ValueError(f"Expected idx shape (1, seq_len), got {tuple(idx.shape)}")
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma}")

    seq = idx.clone()
    prompt_len = seq.shape[1]

    metrics = {
        "tokens_generated": 0,
        "draft_tokens_proposed": 0,
        "draft_tokens_accepted": 0,
        "bonus_tokens": 0,
        "acceptance_rate": 0.0,
        "target_forward_passes": 0,
        "speculative_steps": 0,
        "tokens_per_target_pass": 0.0,
    }
    if max_new_tokens <= 0:
        return seq, metrics

    effective_ctx = min(
        context_size,
        target_model.pos_emb.weight.shape[0],
        draft_model.pos_emb.weight.shape[0],
    )

    # Initial prefill over seq[:, :-1] when inside effective_ctx so seq[:, -1:]
    # stays as the shared query root. If already at/past effective_ctx, defer to
    # the single-pass sliding window branch below.
    target_kv: KVCache | None = None
    draft_kv: KVCache | None = None
    if 1 < seq.shape[1] < effective_ctx:
        prefix = seq[:, :-1]
        _, target_kv = target_model(prefix, use_cache=True, past_key_values=None)
        _, draft_kv = draft_model(prefix, use_cache=True, past_key_values=None)
        metrics["target_forward_passes"] += 1

    generated = 0
    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        cache_len = target_kv[0][0].shape[-2] if target_kv is not None else 0

        # Once committed sequence reaches effective_ctx, sliding the context window
        # shifts every token's absolute position index, invalidating prior KV caches.
        # Fall back to exact single-pass target decoding on seq[:, -effective_ctx:].
        if seq.shape[1] >= effective_ctx:
            window = seq[:, -effective_ctx:]
            logits = target_model(window, use_cache=False)
            metrics["target_forward_passes"] += 1

            step_logits = logits[:, -1, :]
            if temperature <= 0.0:
                next_tok = torch.argmax(step_logits, dim=-1, keepdim=True)
            else:
                p_step = _logits_to_probs(step_logits, temperature, top_k)
                next_tok = torch.multinomial(p_step, num_samples=1)

            seq = torch.cat((seq, next_tok), dim=1)
            generated += 1
            if eos_id is not None and next_tok.item() == eos_id:
                break
            target_kv = None
            draft_kv = None
            continue

        # How many draft tokens can fit in `effective_ctx` alongside `cache_len` and `seq[:, -1:]`?
        # Verification consumes (gamma_step + 1) query tokens on top of `cache_len`.
        room_for_draft = effective_ctx - cache_len - 1
        gamma_step = min(gamma, remaining, room_for_draft)

        if gamma_step <= 0:
            # Exactly 1 position left in effective_ctx: run single target step.
            logits, target_kv = target_model(seq[:, -1:], use_cache=True, past_key_values=target_kv)
            metrics["target_forward_passes"] += 1
            step_logits = logits[:, -1, :]
            if temperature <= 0.0:
                next_tok = torch.argmax(step_logits, dim=-1, keepdim=True)
            else:
                p_step = _logits_to_probs(step_logits, temperature, top_k)
                next_tok = torch.multinomial(p_step, num_samples=1)
            seq = torch.cat((seq, next_tok), dim=1)
            generated += 1
            if eos_id is not None and next_tok.item() == eos_id:
                break
            continue

        # ------------------------------------------------------------------
        # 1. Draft Phase: draft_model proposes `gamma_step` tokens one-by-one.
        #    After completion, `draft_kv` has length (cache_len + gamma_step),
        #    leaving the final drafted token unconsumed.
        # ------------------------------------------------------------------
        draft_tokens: list[Tensor] = []
        draft_probs: list[Tensor] = []
        draft_input = seq[:, -1:]

        for _ in range(gamma_step):
            d_logits, draft_kv = draft_model(
                draft_input, use_cache=True, past_key_values=draft_kv
            )
            step_d_logits = d_logits[:, -1, :]
            if temperature <= 0.0:
                d_tok = torch.argmax(step_d_logits, dim=-1, keepdim=True)
                q_dist = None
            else:
                q_dist = _logits_to_probs(step_d_logits, temperature, top_k)
                d_tok = torch.multinomial(q_dist, num_samples=1)

            draft_tokens.append(d_tok)
            if q_dist is not None:
                draft_probs.append(q_dist)
            draft_input = d_tok

        metrics["draft_tokens_proposed"] += gamma_step
        metrics["speculative_steps"] += 1

        # ------------------------------------------------------------------
        # 2. Target Verification Phase: single batched forward pass over
        #    [seq[:, -1], d_0, d_1, ..., d_{gamma_step - 1}] (length gamma_step + 1).
        # ------------------------------------------------------------------
        draft_tensor = torch.cat(draft_tokens, dim=1)  # (1, gamma_step)
        verify_input = torch.cat((seq[:, -1:], draft_tensor), dim=1)  # (1, gamma_step + 1)

        target_logits, target_kv = target_model(
            verify_input, use_cache=True, past_key_values=target_kv
        )
        metrics["target_forward_passes"] += 1

        # ------------------------------------------------------------------
        # 3. Acceptance & Exact Residual Rejection Sampling Loop
        # ------------------------------------------------------------------
        n_accepted = 0
        hit_eos = False
        rejected = False

        for k in range(gamma_step):
            t_logits_k = target_logits[:, k, :]
            d_tok_k = draft_tokens[k]

            if temperature <= 0.0:
                t_tok_k = torch.argmax(t_logits_k, dim=-1, keepdim=True)
                accept = bool(t_tok_k.item() == d_tok_k.item())
                replacement_tok = t_tok_k
            else:
                p_k = _logits_to_probs(t_logits_k, temperature, top_k)
                q_k = draft_probs[k]
                tok_id = d_tok_k.item()
                p_val = p_k[0, tok_id]
                q_val = torch.clamp(q_k[0, tok_id], min=1e-12)
                accept_prob = torch.clamp(p_val / q_val, max=1.0)
                r = torch.rand((), device=seq.device)
                accept = bool(r < accept_prob)
                replacement_tok = None if accept else sample_residual_distribution(p_k, q_k)

            if accept:
                seq = torch.cat((seq, d_tok_k), dim=1)
                n_accepted += 1
                generated += 1
                metrics["draft_tokens_accepted"] += 1
                if eos_id is not None and d_tok_k.item() == eos_id:
                    hit_eos = True
                    break
                if generated >= max_new_tokens:
                    break
            else:
                assert replacement_tok is not None
                seq = torch.cat((seq, replacement_tok), dim=1)
                generated += 1
                rejected = True
                if eos_id is not None and replacement_tok.item() == eos_id:
                    hit_eos = True
                break

        # ------------------------------------------------------------------
        # 4. Bonus Token (only when all `gamma_step` tokens accepted & budget remains)
        # ------------------------------------------------------------------
        all_accepted = (not rejected) and (n_accepted == gamma_step)
        took_bonus = False
        if all_accepted and (not hit_eos) and (generated < max_new_tokens):
            bonus_logits = target_logits[:, gamma_step, :]
            if temperature <= 0.0:
                bonus_tok = torch.argmax(bonus_logits, dim=-1, keepdim=True)
            else:
                p_bonus = _logits_to_probs(bonus_logits, temperature, top_k)
                bonus_tok = torch.multinomial(p_bonus, num_samples=1)

            seq = torch.cat((seq, bonus_tok), dim=1)
            generated += 1
            metrics["bonus_tokens"] += 1
            took_bonus = True
            if eos_id is not None and bonus_tok.item() == eos_id:
                hit_eos = True

        if hit_eos or generated >= max_new_tokens:
            break

        # ------------------------------------------------------------------
        # 5. O(1) KV-Cache Rollback to Restore `len(kv) == cache_len + 1 + n_accepted`
        #    Note: the committed sequence grew by (n_accepted + 1) tokens in total
        #    (`n_accepted` draft tokens + either 1 corrected or 1 bonus token),
        #    so the cached prefix preceding the new `seq[:, -1:]` is always
        #    `cache_len + 1 + n_accepted`.
        # ------------------------------------------------------------------
        if seq.shape[1] >= effective_ctx:
            target_kv = None
            draft_kv = None
            continue

        new_cache_len = cache_len + 1 + n_accepted
        target_kv = crop_kv_cache(target_kv, new_cache_len)

        if took_bonus:
            # Advance `draft_kv` (currently at cache_len + gamma_step) over the
            # last accepted draft token so `draft_kv` matches `new_cache_len` too.
            _, draft_kv = draft_model(
                draft_tokens[-1], use_cache=True, past_key_values=draft_kv
            )
        else:
            draft_kv = crop_kv_cache(draft_kv, new_cache_len)

    metrics["tokens_generated"] = seq.shape[1] - prompt_len
    if metrics["draft_tokens_proposed"] > 0:
        metrics["acceptance_rate"] = (
            metrics["draft_tokens_accepted"] / metrics["draft_tokens_proposed"]
        )
    if metrics["target_forward_passes"] > 0:
        metrics["tokens_per_target_pass"] = (
            metrics["tokens_generated"] / metrics["target_forward_passes"]
        )

    return seq, metrics
