# data/add_citations.py
"""Add inline [n] citations to grounded SFT examples.

The grounded examples in data/sft_mix.jsonl use the serving format
("[1] snippet\\n[2] snippet\\n\\nQuestion: ...") but their responses never
cite, so a model tuned on them never writes [n] markers. This rewrites each
grounded response with citations from the same attribution the API applies
at serving time (app/citations.py), so an adapter retrained on the output
learns to cite on its own. Non-grounded examples pass through unchanged.

    python data/add_citations.py --data data/sft_mix.jsonl --out data/sft_mix_cited.jsonl
    python training/finetune_instruct.py --data data/sft_mix_cited.jsonl ...
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.citations import cite

_SOURCE_RE = re.compile(r"^\[(\d+)\] ", re.MULTILINE)


def parse_grounded_instruction(instruction: str) -> tuple[list[dict], str] | None:
    """Split "[1] ... [n] ...\\n\\nQuestion: q" into (sources, question)."""
    if not instruction.startswith("[1] ") or "\nQuestion: " not in instruction:
        return None
    context, question = instruction.rsplit("\nQuestion: ", 1)
    starts = [m for m in _SOURCE_RE.finditer(context)]
    if [int(m.group(1)) for m in starts] != list(range(1, len(starts) + 1)):
        return None  # not a clean 1..n numbering
    sources = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(context)
        sources.append({"snippet": context[match.end():end].strip()})
    return sources, question.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/sft_mix.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    total = grounded = with_citations = sentences_cited = 0
    with open(args.data, encoding="utf-8") as src, open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            example = json.loads(line)
            total += 1
            parsed = parse_grounded_instruction(example["instruction"])
            if parsed:
                grounded += 1
                sources, question = parsed
                cited = cite(question, example["response"], sources)
                # Only the inline markers: a "From the sources:" prefix is a
                # serving-time fallback, not something to teach the model.
                example = {**example, "response": cited.text}
                with_citations += bool(cited.citations)
                sentences_cited += len(cited.citations)
            dst.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"{total} examples, {grounded} grounded; {with_citations} of those now cite "
          f"({with_citations / max(grounded, 1):.1%}), {sentences_cited} citations total -> {args.out}")


if __name__ == "__main__":
    main()
