# evals/harness.py
"""Evaluation harness for the GPT serving stack.

Every suite runs through the production code path -- GPTInferenceEngine for
generation and app.prompting for prompt construction -- so the numbers
describe what users are actually served, not an idealized offline setup.

Suites
------
heldout   Response-only cross-entropy and perplexity on held-out instruction
          data (data/sft_eval.jsonl), with the same prompt masking as SFT.
mc        Multiple-choice questions scored by log-likelihood of each choice.
          A stable, generation-free signal even for small models.
behavior  Greedy generations on fixed prompts checked by rules: factual
          recall, instruction following, chitchat, multi-turn memory, RAG
          grounding on invented facts, plus hygiene on every output (template
          leakage, empty output, failure to stop, repetition).
retrieval Ranking of web-search candidates against relevance labels (MRR,
          precision@1, recall@3), including trap results and a keyword-stuffed
          injection attempt. Model-free unless RAG_EMBED_MODEL is set.

A report is a flat dict of metrics plus the provenance needed to decide
whether two reports are comparable (eval-data hashes, decoding settings).
compare_to_baseline() turns a pair of reports into a pass/fail regression gate.
"""

import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from app.citations import cite
from app.prompting import build_prompt_with_budget
from app.schemas import GenerationRequest
from data.sft import IGNORE_INDEX, SFTDataset, collate_sft, format_prompt

HARNESS_VERSION = 1
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILES = {
    "heldout": os.path.join(REPO_ROOT, "data", "sft_eval.jsonl"),
    "mc": os.path.join(REPO_ROOT, "evals", "data", "multiple_choice.jsonl"),
    "behavior": os.path.join(REPO_ROOT, "evals", "data", "behavior.jsonl"),
    "retrieval": os.path.join(REPO_ROOT, "evals", "data", "retrieval.jsonl"),
}
ALL_SUITES = tuple(DATA_FILES)

# Text that must never appear in a completion: it means the model is
# continuing the prompt format instead of answering.
TEMPLATE_MARKERS = ("### Instruction", "### Response", "Below is an instruction", "<|endoftext|>")

# Metrics the regression gate checks: direction and allowed slack. Small
# suites get more slack -- one behavior case is ~4 points of pass rate.
GATED_METRICS = {
    "heldout.response_loss": ("lower", 0.02),
    "mc.accuracy_norm": ("higher", 0.05),
    "behavior.pass_rate": ("higher", 0.05),
    "behavior.template_leak_rate": ("lower", 0.0),
    "behavior.empty_rate": ("lower", 0.0),
    "retrieval.mrr": ("higher", 0.0),  # deterministic: any drop is a real change
}


def serving_decoding(max_new_tokens: int = 64) -> dict:
    """The API's default sampling settings, made greedy so runs are repeatable."""
    defaults = GenerationRequest(prompt="x").model_dump()
    keys = ("top_k", "top_p", "repetition_penalty", "frequency_penalty", "presence_penalty",
            "no_repeat_ngram_size", "min_new_tokens", "use_cache")
    return {**{k: defaults[k] for k in keys}, "temperature": 0.0, "max_new_tokens": max_new_tokens}


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
    except Exception:
        return None


# ── Suites ────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_heldout(engine, rows: list[dict], batch_size: int = 8) -> tuple[dict, dict]:
    """Token-weighted response-only loss, exactly as SFT computes it."""
    tok = engine.tokenizer
    triples = [(r["instruction"], r["response"], [tuple(t) for t in r.get("history", [])]) for r in rows]
    dataset = SFTDataset(triples, tok, max_length=engine.context_size)

    total_loss, total_tokens = 0.0, 0
    engine.model.eval()
    for start in range(0, len(dataset), batch_size):
        inputs, labels = collate_sft([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))], tok.eos_id)
        logits = engine.model(inputs.to(engine.device))
        labels = labels.to(engine.device)
        total_loss += F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                      ignore_index=IGNORE_INDEX, reduction="sum").item()
        total_tokens += (labels != IGNORE_INDEX).sum().item()

    loss = total_loss / max(total_tokens, 1)
    metrics = {
        "heldout.response_loss": loss,
        "heldout.perplexity": math.exp(min(loss, 50.0)),
    }
    details = {"examples": len(dataset), "skipped_too_long": len(rows) - len(dataset), "response_tokens": total_tokens}
    return metrics, details


@torch.no_grad()
def _choice_logprobs(engine, prompt_ids: list[int], choice: str) -> tuple[float, int]:
    # The choice is scored as the response, which directly follows
    # "### Response:\n" -- so it's encoded as-is, with no leading space.
    choice_ids = engine.tokenizer.encode_ordinary(choice)
    ids = (prompt_ids + choice_ids)[-engine.context_size - 1:]
    n_choice = min(len(choice_ids), len(ids) - 1)
    inputs = torch.tensor([ids[:-1]], device=engine.device)
    logprobs = torch.log_softmax(engine.model(inputs)[0].float(), dim=-1)
    targets = torch.tensor(ids[1:], device=engine.device)
    token_lp = logprobs[torch.arange(len(targets)), targets][-n_choice:]
    return token_lp.sum().item(), n_choice


def eval_multiple_choice(engine, rows: list[dict]) -> tuple[dict, list]:
    """Pick the choice with the highest log-likelihood as the model's answer.

    accuracy uses the summed log-probability; accuracy_norm divides by the
    choice's token count, so longer answers aren't penalized just for length.
    """
    engine.model.eval()
    correct = correct_norm = 0
    details = []
    for row in rows:
        prompt_ids = engine.tokenizer.encode_ordinary(format_prompt(row["question"]))
        scored = [_choice_logprobs(engine, prompt_ids, c) for c in row["choices"]]
        pick = max(range(len(scored)), key=lambda i: scored[i][0])
        pick_norm = max(range(len(scored)), key=lambda i: scored[i][0] / max(scored[i][1], 1))
        correct += pick == row["answer"]
        correct_norm += pick_norm == row["answer"]
        details.append({"id": row["id"], "answer": row["answer"], "pick": pick, "pick_norm": pick_norm})
    n = max(len(rows), 1)
    return {"mc.accuracy": correct / n, "mc.accuracy_norm": correct_norm / n}, details


def _distinct_2(text: str) -> float:
    words = text.split()
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 1.0


def contains_term(text: str, term: str) -> bool:
    """Case-insensitive whole-term match: "47" must not match "475", nor
    "red" match "reduced" -- plain substring checks let random text pass."""
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None


def check_case(case: dict, completion: str, tokens_generated: int, max_new_tokens: int, sources: list | None) -> dict:
    """Rule checks for one behavior case. Returns the per-case result record."""
    text = completion.strip()
    failures = []

    if not text:
        failures.append("empty output")
    # Markers are matched as raw substrings: any fragment of the template is a leak.
    leaked = [m for m in TEMPLATE_MARKERS if m.lower() in text.lower()]
    if leaked:
        failures.append(f"template leak: {leaked[0]!r}")
    expect_any = case.get("expect_any") or []
    if expect_any and not any(contains_term(text, e) for e in expect_any):
        failures.append(f"expected any of {expect_any}")
    for banned in case.get("expect_none") or []:
        if contains_term(text, banned):
            failures.append(f"must not contain {banned!r}")

    result = {
        "id": case["id"],
        "category": case["category"],
        "completion": completion,
        "passed": not failures,
        "failures": failures,
        "stopped": tokens_generated < max_new_tokens,
        "distinct_2": _distinct_2(text),
        "leaked": bool(leaked),
        "empty": not text,
    }
    if sources:
        cited = cite(case["prompt"], text, sources)
        # Grounded: the serving safety net didn't need to lead with a quote.
        result["grounded"] = not cited.safety_net_prefix
        result["cited"] = bool(cited.citations)
        result["cited_text"] = cited.full_text
    return result


def eval_behavior(engine, cases: list[dict], decoding: dict) -> tuple[dict, list]:
    results = []
    for case in cases:
        sources = [{"title": "", "snippet": s, "link": ""} for s in case.get("sources", [])] or None
        history = [tuple(turn) for turn in case.get("history", [])]
        prompt_text, sources = build_prompt_with_budget(
            case["prompt"], decoding["max_new_tokens"], sources, engine.context_size, history
        )
        out = engine.generate(prompt=prompt_text, **decoding)
        results.append(check_case(case, out["completion_text"], out["tokens_generated"],
                                  decoding["max_new_tokens"], sources))

    n = max(len(results), 1)
    metrics = {
        "behavior.pass_rate": sum(r["passed"] for r in results) / n,
        "behavior.template_leak_rate": sum(r["leaked"] for r in results) / n,
        "behavior.empty_rate": sum(r["empty"] for r in results) / n,
        "behavior.stop_rate": sum(r["stopped"] for r in results) / n,
        "behavior.distinct_2": sum(r["distinct_2"] for r in results) / n,
    }
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["passed"])
    for category, passed in sorted(by_category.items()):
        metrics[f"behavior.pass_rate.{category}"] = sum(passed) / len(passed)
    grounded = [r["grounded"] for r in results if "grounded" in r]
    if grounded:
        metrics["behavior.rag_grounded_rate"] = sum(grounded) / len(grounded)
        metrics["behavior.rag_cited_rate"] = sum(r["cited"] for r in results if "cited" in r) / len(grounded)
    return metrics, results


def eval_retrieval(rows: list[dict]) -> tuple[dict, list]:
    """Run each query's candidates through the production selection pipeline
    (sanitize, drop injection attempts, rank, de-duplicate). Dropped
    candidates count as ranked last.

    Candidates are shuffled with a fixed per-query seed first: ranking ties
    keep input order, so the file's own ordering must not leak the labels.
    """
    from app.retrieval import get_dense_reranker, select_sources

    reranker = get_dense_reranker()
    mrr = p_at_1 = recall_at_3 = 0.0
    details = []
    for i, row in enumerate(rows):
        order = list(range(len(row["candidates"])))
        random.Random(i).shuffle(order)
        candidates = [{**row["candidates"][j], "link": "", "_id": j} for j in order]
        kept = select_sources(row["query"], candidates, max_results=len(candidates),
                              reranker=reranker, dedup_ratio=1.01)  # rank everything; no dedup
        ranked = [c["_id"] for c in kept] + [j for j in order if j not in {c["_id"] for c in kept}]
        relevant = set(row["relevant"])
        first_hit = next((k for k, cid in enumerate(ranked) if cid in relevant), None)
        mrr += 1 / (first_hit + 1) if first_hit is not None else 0.0
        p_at_1 += ranked[0] in relevant
        recall_at_3 += len(relevant & set(ranked[:3])) / len(relevant)
        details.append({"query": row["query"], "ranking": ranked, "relevant": sorted(relevant),
                        "top_correct": ranked[0] in relevant})
    n = max(len(rows), 1)
    metrics = {"retrieval.mrr": mrr / n, "retrieval.p_at_1": p_at_1 / n, "retrieval.recall_at_3": recall_at_3 / n}
    return metrics, details


# ── Runner ────────────────────────────────────────────────────────────


def run_eval(engine, suites=ALL_SUITES, limit: int | None = None, max_new_tokens: int = 64,
             checkpoint: str | None = None, data_files: dict | None = None) -> dict:
    """Run the selected suites against an engine and return a report."""
    data_files = {**DATA_FILES, **(data_files or {})}
    decoding = serving_decoding(max_new_tokens)
    torch.manual_seed(0)

    report = {
        "harness_version": HARNESS_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "model": {
            "checkpoint": checkpoint,
            "adapter": engine.active_adapter,
            "config": engine.model_config,
            "parameters": engine.parameter_count,
        },
        "decoding": decoding,
        "limit": limit,
        "data": {},
        "metrics": {},
        "details": {},
        "timings_seconds": {},
    }

    runners = {
        "heldout": lambda rows: eval_heldout(engine, rows),
        "mc": lambda rows: eval_multiple_choice(engine, rows),
        "behavior": lambda rows: eval_behavior(engine, rows, decoding),
        "retrieval": eval_retrieval,
    }
    for suite in suites:
        path = data_files[suite]
        start = time.perf_counter()
        metrics, details = runners[suite](load_jsonl(path, limit))
        report["timings_seconds"][suite] = round(time.perf_counter() - start, 2)
        report["data"][suite] = {"path": os.path.relpath(path, REPO_ROOT), "sha256": _sha256(path)}
        report["metrics"].update(metrics)
        report["details"][suite] = details
    return report


# ── Baseline gate ─────────────────────────────────────────────────────


class IncomparableReports(ValueError):
    """The two reports were produced on different eval data or settings."""


def compare_to_baseline(report: dict, baseline: dict) -> tuple[list[dict], list[str]]:
    """Compare gated metrics. Returns (rows for display, names of regressed metrics).

    Refuses to compare reports built on different eval data, limits, or
    decoding settings: a "regression" between those would be meaningless.
    """
    for key in ("decoding", "limit"):
        if report.get(key) != baseline.get(key):
            raise IncomparableReports(f"'{key}' differs between report and baseline")
    for suite, info in report["data"].items():
        base_info = baseline.get("data", {}).get(suite)
        if base_info and base_info["sha256"] != info["sha256"]:
            raise IncomparableReports(f"eval data for suite '{suite}' changed since the baseline")

    rows, regressions = [], []
    for metric, (direction, tolerance) in GATED_METRICS.items():
        if metric not in report["metrics"] or metric not in baseline.get("metrics", {}):
            continue
        now, before = report["metrics"][metric], baseline["metrics"][metric]
        delta = now - before
        worse = delta > tolerance if direction == "lower" else -delta > tolerance
        rows.append({"metric": metric, "baseline": before, "current": now, "delta": delta,
                     "direction": direction, "tolerance": tolerance, "regressed": worse})
        if worse:
            regressions.append(metric)
    return rows, regressions


def format_summary(report: dict, comparison: list[dict] | None = None) -> str:
    """Markdown summary for terminals and CI job summaries."""
    model = report["model"]
    lines = [
        f"## Eval report — adapter: `{model['adapter'] or 'base model'}`",
        "",
        f"checkpoint `{model['checkpoint']}` · git `{(report['git_sha'] or 'unknown')[:10]}` · "
        f"greedy, max_new_tokens={report['decoding']['max_new_tokens']}"
        + (f" · limit={report['limit']}" if report["limit"] else ""),
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.4f} |" for k, v in report["metrics"].items()]

    if comparison is not None:
        lines += ["", "| Gated metric | Baseline | Current | Δ | Status |", "|---|---|---|---|---|"]
        for row in comparison:
            status = "❌ regressed" if row["regressed"] else "✅ ok"
            lines.append(f"| {row['metric']} | {row['baseline']:.4f} | {row['current']:.4f} | "
                         f"{row['delta']:+.4f} | {status} |")

    failed = [r for r in report["details"].get("behavior", []) if not r["passed"]]
    if failed:
        lines += ["", "<details><summary>Failed behavior cases</summary>", ""]
        lines += [f"- `{r['id']}`: {'; '.join(r['failures'])} — {r['completion'].strip()[:120]!r}" for r in failed]
        lines += ["", "</details>"]
    return "\n".join(lines)
