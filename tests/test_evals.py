# tests/test_evals.py
"""Tests for the eval harness (evals/): scoring rules, suites, gate, CLI."""

import copy
import json
import math

import pytest
import torch

from app.inference import GPTInferenceEngine
from data.sft import format_prompt
from evals import harness
from evals.__main__ import main as cli_main
from evals.harness import (
    IncomparableReports, check_case, compare_to_baseline, contains_term, format_summary, run_eval,
)
from model.gpt import GPTModel

TINY = {"vocab_size": 50257, "context_length": 256, "emb_dim": 32, "n_heads": 2,
        "n_layers": 1, "drop_rate": 0.0, "qkv_bias": False, "model_size": "tiny"}


@pytest.fixture(scope="module")
def engine():
    torch.manual_seed(0)
    return GPTInferenceEngine.from_config(TINY)


@pytest.fixture(scope="module")
def report(engine):
    return run_eval(engine, limit=4, max_new_tokens=6, checkpoint="tiny")


# ── Scoring rules ────────────────────────────────────────────────────


def test_whole_term_matching_avoids_false_passes():
    assert contains_term("The answer is 47.", "47")
    assert not contains_term("model output 475IG", "47")
    assert not contains_term("Reduced", "red")
    assert contains_term("Mira Okonkwo found it", "okonkwo")


def test_check_case_rules():
    case = {"id": "c", "category": "factual", "prompt": "Capital of France?", "expect_any": ["Paris"]}
    assert check_case(case, "It is Paris.", 5, 64, None)["passed"]

    miss = check_case(case, "It is Lyon.", 5, 64, None)
    assert not miss["passed"] and "expected any" in miss["failures"][0]

    leak = check_case(case, "Paris.\n\n### Instruction:\nMore", 64, 64, None)
    assert not leak["passed"] and leak["leaked"] and not leak["stopped"]

    assert check_case(case, "   ", 1, 64, None)["empty"]
    banned = {**case, "expect_none": ["London"]}
    assert not check_case(banned, "Paris, not London", 5, 64, None)["passed"]

    sources = [{"snippet": "The Zorblax festival is held in Quillmoor every spring."}]
    rag = {"id": "r", "category": "rag", "prompt": "Where is the Zorblax festival held?"}
    assert check_case(rag, "The Zorblax festival is held in Quillmoor every spring.", 5, 64, sources)["grounded"]
    assert not check_case(rag, "I like cake and ice cream a lot today.", 5, 64, sources)["grounded"]


# ── Suites on a tiny random model ────────────────────────────────────


def test_report_structure_and_sanity(report):
    m = report["metrics"]
    assert math.isclose(m["heldout.perplexity"], math.exp(m["heldout.response_loss"]), rel_tol=1e-9)
    # A random model is near uniform over the vocabulary.
    assert abs(m["heldout.response_loss"] - math.log(TINY["vocab_size"])) < 1.0
    for key in ("mc.accuracy", "mc.accuracy_norm", "behavior.pass_rate", "behavior.template_leak_rate"):
        assert 0.0 <= m[key] <= 1.0
    assert report["decoding"]["temperature"] == 0.0
    assert set(report["data"]) == {"heldout", "mc", "behavior"}
    assert all(len(d["sha256"]) == 64 for d in report["data"].values())
    assert len(report["details"]["behavior"]) == 4 and len(report["details"]["mc"]) == 4


def test_generation_is_deterministic(engine):
    first = run_eval(engine, suites=["behavior"], limit=3, max_new_tokens=6)
    second = run_eval(engine, suites=["behavior"], limit=3, max_new_tokens=6)
    assert [r["completion"] for r in first["details"]["behavior"]] == [r["completion"] for r in second["details"]["behavior"]]


def test_choice_logprob_matches_manual_computation(engine):
    prompt_ids = engine.tokenizer.encode_ordinary(format_prompt("What is 2 + 2?"))
    choice_ids = engine.tokenizer.encode_ordinary("Four")
    total, n = harness._choice_logprobs(engine, prompt_ids, "Four")

    ids = prompt_ids + choice_ids
    with torch.no_grad():
        logprobs = torch.log_softmax(engine.model(torch.tensor([ids[:-1]]))[0], dim=-1)
    expected = sum(logprobs[len(prompt_ids) - 1 + i, t].item() for i, t in enumerate(choice_ids))
    assert n == len(choice_ids) and math.isclose(total, expected, rel_tol=1e-4)


def test_behavior_uses_the_serving_prompt_and_scores_by_category(engine, monkeypatch, tmp_path):
    cases = [
        {"id": "f", "category": "factual", "prompt": "Capital of France?", "expect_any": ["Paris"]},
        {"id": "m", "category": "multi_turn", "prompt": "What is my name?",
         "history": [["My name is Ada.", "Hi Ada!"]], "expect_any": ["Ada"]},
        {"id": "r", "category": "rag", "prompt": "Where is it held?",
         "sources": ["The festival is held in Quillmoor."], "expect_any": ["Quillmoor"]},
    ]
    path = tmp_path / "behavior.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases))
    answers = {"f": "Paris.", "m": "I don't know.", "r": "It is held in Quillmoor."}
    prompts = []

    def fake_generate(prompt, **kwargs):
        prompts.append(prompt)
        case_id = "m" if "Ada" in prompt else ("r" if "Quillmoor" in prompt else "f")
        return {"completion_text": answers[case_id], "tokens_generated": 3}

    monkeypatch.setattr(engine, "generate", fake_generate)
    out = run_eval(engine, suites=["behavior"], data_files={"behavior": str(path)})
    m = out["metrics"]
    assert m["behavior.pass_rate.factual"] == 1.0
    assert m["behavior.pass_rate.multi_turn"] == 0.0
    assert m["behavior.pass_rate.rag"] == 1.0
    assert math.isclose(m["behavior.pass_rate"], 2 / 3)
    assert prompts[1] == format_prompt("What is my name?", [("My name is Ada.", "Hi Ada!")])
    assert "[1] The festival is held in Quillmoor." in prompts[2]


# ── Regression gate ──────────────────────────────────────────────────


def test_gate_flags_regressions_beyond_tolerance(report):
    baseline = copy.deepcopy(report)
    rows, regressions = compare_to_baseline(report, baseline)
    assert regressions == [] and rows

    worse = copy.deepcopy(report)
    worse["metrics"]["heldout.response_loss"] += 0.5          # lower is better
    worse["metrics"]["mc.accuracy_norm"] -= 0.5               # higher is better
    worse["metrics"]["behavior.pass_rate"] -= 0.01            # within tolerance
    _, regressions = compare_to_baseline(worse, baseline)
    assert sorted(regressions) == ["heldout.response_loss", "mc.accuracy_norm"]

    better = copy.deepcopy(report)
    better["metrics"]["heldout.response_loss"] -= 0.5
    assert compare_to_baseline(better, baseline)[1] == []
    assert "regressed" in format_summary(worse, compare_to_baseline(worse, baseline)[0])


def test_gate_refuses_incomparable_reports(report):
    changed_data = copy.deepcopy(report)
    changed_data["data"]["mc"]["sha256"] = "0" * 64
    with pytest.raises(IncomparableReports, match="mc"):
        compare_to_baseline(report, changed_data)

    changed_decoding = copy.deepcopy(report)
    changed_decoding["decoding"]["max_new_tokens"] = 999
    with pytest.raises(IncomparableReports, match="decoding"):
        compare_to_baseline(report, changed_decoding)


def test_cli_exit_codes(tmp_path):
    ckpt = tmp_path / "tiny.pt"
    torch.manual_seed(0)
    torch.save({"model_state_dict": GPTModel(TINY).state_dict(), "model_config": TINY}, ckpt)
    out = tmp_path / "report.json"
    args = ["--checkpoint", str(ckpt), "--suites", "heldout,mc", "--limit", "2"]

    assert cli_main(args + ["--out", str(out)]) == 0
    assert cli_main(args + ["--baseline", str(out)]) == 0

    baseline = json.loads(out.read_text())
    baseline["metrics"]["heldout.response_loss"] -= 1.0  # baseline was much better -> regression
    (tmp_path / "better.json").write_text(json.dumps(baseline))
    assert cli_main(args + ["--baseline", str(tmp_path / "better.json")]) == 1

    baseline["data"]["heldout"]["sha256"] = "0" * 64
    (tmp_path / "other.json").write_text(json.dumps(baseline))
    assert cli_main(args + ["--baseline", str(tmp_path / "other.json")]) == 2
