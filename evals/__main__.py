# evals/__main__.py
"""Run the eval harness from the command line (from the repo root).

    python -m evals --checkpoint checkpoints/best_model.pt --adapter sft_v1_small \
        --out reports/sft_v1_small.json --baseline evals/baselines/sft_v1_small.json

Exit status: 0 = ok, 1 = a gated metric regressed past its tolerance,
2 = the baseline was produced on different eval data or settings.
"""

import argparse
import json
import os
import sys

from evals.harness import ALL_SUITES, IncomparableReports, compare_to_baseline, format_summary, run_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint (.pt).")
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter name in checkpoints/adapters/ (omit or 'none' for the base model).")
    parser.add_argument("--suites", default=",".join(ALL_SUITES),
                        help=f"Comma-separated subset of: {', '.join(ALL_SUITES)}.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N items of each suite.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length for behavior cases.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--out", default=None, help="Write the full JSON report here.")
    parser.add_argument("--baseline", default=None, help="Baseline report to gate against.")
    args = parser.parse_args(argv)

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = sorted(set(suites) - set(ALL_SUITES))
    if unknown:
        parser.error(f"unknown suites: {', '.join(unknown)}")

    from app.inference import GPTInferenceEngine

    engine = GPTInferenceEngine(args.checkpoint, device=args.device)
    if args.adapter and args.adapter.lower() != "none":
        engine.set_adapter(engine.adapters.get(args.adapter))

    report = run_eval(engine, suites, limit=args.limit, max_new_tokens=args.max_new_tokens,
                      checkpoint=args.checkpoint)

    comparison, exit_code = None, 0
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)
        try:
            comparison, regressions = compare_to_baseline(report, baseline)
            exit_code = 1 if regressions else 0
        except IncomparableReports as e:
            print(f"Cannot compare against {args.baseline}: {e}", file=sys.stderr)
            exit_code = 2

    summary = format_summary(report, comparison)
    print(summary)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(summary + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
