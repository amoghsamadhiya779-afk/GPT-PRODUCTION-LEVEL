# app/citations.py
"""Inline citations for grounded (web-search) answers.

The SFT data holds 4,011 grounded examples in the exact format the API serves
("[1] snippet ... Question: ..."), but none of their responses cite sources,
so the model doesn't write [n] markers on its own. Citations are attached
after generation instead, by attribution: each answer sentence is matched to
the numbered source that contains most of its content words, using the same
term normalization as retrieval (app/retrieval.py).

The same pass also:
- validates markers the model does write (after retraining on data from
  data/add_citations.py it will), dropping ones that point at a source that
  doesn't exist -- a hallucinated citation is worse than none;
- decides when an answer is too ungrounded to stand alone, in which case the
  best-matching source sentence is shown first ("From the sources: ...").
"""

import re
from dataclasses import dataclass, field

from app.retrieval import terms

# Share of a sentence's content terms that must appear in a source for the
# sentence to be attributed to it.
SUPPORT_THRESHOLD = 0.5
# Below this share of supported sentences, lead with a quote from the sources.
MIN_GROUNDED_FRACTION = 0.5

_MARKER_RE = re.compile(r"\s*\[(\d{1,2})\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_KEEP_SEP_RE = re.compile(r"((?<=[.!?])\s+)")


@dataclass
class CitedAnswer:
    text: str                                   # answer with [n] markers
    citations: list[dict] = field(default_factory=list)  # {"sentence", "source", "support"}
    grounded_fraction: float = 1.0              # supported / checkable sentences
    safety_net_prefix: str = ""                 # quote shown first when ungrounded

    @property
    def full_text(self) -> str:
        return self.safety_net_prefix + self.text


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text.strip()) if s]


def support(sentence: str, source: str) -> float:
    """Share of the sentence's distinct content terms found in the source."""
    sentence_terms = set(terms(sentence))
    if not sentence_terms:
        return 0.0
    return len(sentence_terms & set(terms(source))) / len(sentence_terms)


def _with_marker(sentence: str, number: int) -> str:
    """Insert " [n]" before the sentence's closing punctuation, if any."""
    match = re.search(r"[.!?]+[\"')\]]*$", sentence)
    if match:
        return f"{sentence[:match.start()]} [{number}]{sentence[match.start():]}"
    return f"{sentence} [{number}]"


def _best_quote(question: str, sources: list[dict]) -> tuple[str, int] | None:
    """The source sentence sharing the most content terms with the question."""
    question_terms = set(terms(question))
    best, best_overlap = None, 0
    for number, source in enumerate(sources, 1):
        for sentence in split_sentences(source["snippet"]):
            overlap = len(question_terms & set(terms(sentence)))
            if overlap > best_overlap:
                best, best_overlap = (sentence.strip(), number), overlap
    return best


def cite(question: str, answer: str, sources: list[dict]) -> CitedAnswer:
    """Attach [n] citations to `answer`, whose numbers match `sources` order."""
    if not sources:
        return CitedAnswer(text=answer)

    # Alternating [sentence, separator, sentence, ...] so the original
    # whitespace (newlines in lists and paragraphs) survives annotation.
    pieces = _SENTENCE_KEEP_SEP_RE.split(answer.strip())
    sentences, separators = pieces[0::2], pieces[1::2] + [""]

    citations, annotated = [], []
    checkable = supported = 0
    for index, sentence in enumerate(sentences):
        # Keep model-written markers only when they point at a real source.
        claimed = [int(n) for n in _MARKER_RE.findall(sentence) if 1 <= int(n) <= len(sources)]
        bare = _MARKER_RE.sub("", sentence).strip()
        if len(terms(bare)) < 2:  # "Yes." / "Sure!" -- nothing to attribute
            annotated.append(bare)
            continue

        checkable += 1
        scores = [support(bare, s["snippet"]) for s in sources]
        best = max(range(len(sources)), key=lambda i: scores[i])
        if claimed:
            number = claimed[0]
            supported += scores[number - 1] >= SUPPORT_THRESHOLD
        elif scores[best] >= SUPPORT_THRESHOLD:
            number = best + 1
            supported += 1
        else:
            annotated.append(bare)
            continue
        annotated.append(_with_marker(bare, number))
        citations.append({"sentence": index, "source": number, "support": round(scores[number - 1], 3)})

    grounded = supported / checkable if checkable else 0.0
    text = "".join(sentence + sep for sentence, sep in zip(annotated, separators))
    result = CitedAnswer(text=text, citations=citations, grounded_fraction=grounded)
    if grounded < MIN_GROUNDED_FRACTION:
        quote = _best_quote(question, sources)
        if quote:
            sentence, number = quote
            sentence = sentence if re.search(r"[.!?]$", sentence) else sentence + "."
            result.safety_net_prefix = f"From the sources: {sentence} [{number}]\n\n"
    return result
