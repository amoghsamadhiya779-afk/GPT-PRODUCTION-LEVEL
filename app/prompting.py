# app/prompting.py
"""Prompt construction shared by the serving API and the eval harness.

Everything that decides what text the model actually sees -- the instruction
template, conversation history, web-source grounding, and the token budget --
lives here, so evaluations measure exactly the prompts users are served.
"""

import re
from collections.abc import Sequence

import tiktoken

# The instruction template shared with SFT training, so serving prompts are
# byte-for-byte what the adapters were trained on.
from data.sft import format_prompt

STOP_WORDS = {"a", "an", "the", "and", "but", "if", "or", "because", "as", "what", "which", "this", "that", "these", "those", "then", "just", "so", "than", "such", "both", "through", "about", "for", "is", "of", "while", "during", "to", "in", "it", "on", "with"}

def check_grounding_safety(prompt: str, answer: str, sources: list) -> str:
    if not sources:
        return ""

    def get_words(text):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return set(w for w in words if w not in STOP_WORDS)

    gen_words = get_words(answer)
    source_words = set()
    for s in sources:
        source_words.update(get_words(s['snippet']))

    overlap = len(gen_words.intersection(source_words))
    overlap_ratio = overlap / len(gen_words) if gen_words else 0.0
    # Trigger when the answer barely echoes the sources -- either in absolute
    # terms (very few shared substantive words, catches short answers that
    # are entirely off-topic) or proportionally (mostly made up of words that
    # appear nowhere in the sources, catches longer answers that drift).
    if overlap < 3 or (len(gen_words) >= 6 and overlap_ratio < 0.2):
        best_sentence = ""
        max_overlap = -1
        prompt_words = get_words(prompt)

        for s in sources:
            sentences = re.split(r'(?<=[.!?]) +', s['snippet'])
            for sent in sentences:
                sent_words = get_words(sent)
                o = len(prompt_words.intersection(sent_words))
                if o > max_overlap:
                    max_overlap = o
                    best_sentence = sent.strip()

        if best_sentence:
            return f"From the sources: {best_sentence}.\n\n"
    return ""


def history_pairs(turns) -> list[tuple[str, str]]:
    """Pair each user turn with the assistant reply that directly follows it.

    Unanswered user turns (e.g. a failed generation) and assistant turns with
    no preceding question (e.g. a persona's welcome message) are dropped, so
    the prompt always alternates Instruction/Response.
    """
    pairs, question = [], None
    for turn in turns:
        text = turn.content.strip()
        if turn.role == "user":
            question = text or None
        elif question is not None and text:
            pairs.append((question, text))
            question = None
    return pairs


def build_prompt_with_budget(
    original_prompt: str,
    max_new_tokens: int,
    sources: list | None,
    context_size: int,
    history: Sequence[tuple[str, str]] = (),
) -> tuple[str, list | None]:
    """Build the model prompt so prompt + completion fit the context window.

    Priority when space runs out: the current prompt, then its web sources,
    then earlier conversation turns (most recent first; the oldest are
    dropped). User text and web snippets are counted with encode_ordinary,
    exactly as the engine will encode them (special-token strings stay text).
    """
    enc = tiktoken.get_encoding("gpt2")

    def n_tokens(text: str) -> int:
        return len(enc.encode_ordinary(text))

    # Reserve room for the completion, but always keep at least half the
    # window for the prompt (generation past the window slides it forward).
    budget = max(context_size - max_new_tokens, context_size // 2)

    # A prompt longer than the budget used to be cropped from the *left* by
    # generate(), silently cutting off the instruction header so the model no
    # longer saw the template it was tuned on. Keep the template intact and
    # drop the oldest part of the user's text instead.
    user_budget = max(budget - n_tokens(format_prompt("")), 1)
    prompt_ids = enc.encode_ordinary(original_prompt)
    while len(prompt_ids) > user_budget:
        original_prompt = enc.decode(prompt_ids[-user_budget:]).lstrip("\ufffd")
        # BPE merges across the template boundary can shift the count by a
        # token or two, so re-check the assembled prompt.
        excess = n_tokens(format_prompt(original_prompt)) - budget
        if excess <= 0 or user_budget == 1:
            break
        user_budget = max(user_budget - excess, 1)

    instruction = original_prompt
    if sources:
        valid_sources = sources.copy()
        while valid_sources:
            context_str = ""
            for i, res in enumerate(valid_sources, 1):
                context_str += f"[{i}] {res['snippet']}\n"
            grounded = f"{context_str}\nQuestion: {original_prompt}"

            prompt_tokens = n_tokens(format_prompt(grounded))
            if prompt_tokens <= budget:
                instruction = grounded
                break

            excess = prompt_tokens - budget

            last_src = valid_sources[-1]
            raw_tokens = enc.encode_ordinary(last_src['snippet'])

            trim_len = len(raw_tokens) - excess - 2
            if trim_len <= 0:
                valid_sources.pop()
            else:
                new_src = last_src.copy()
                new_src['snippet'] = enc.decode(raw_tokens[:trim_len]).strip() + "..."
                valid_sources[-1] = new_src
        sources = valid_sources

    # Earlier turns get whatever room is left, newest first.
    kept: list[tuple[str, str]] = []
    for turn in reversed(history):
        if n_tokens(format_prompt(instruction, [turn] + kept)) > budget:
            break
        kept.insert(0, turn)

    return format_prompt(instruction, kept), sources
