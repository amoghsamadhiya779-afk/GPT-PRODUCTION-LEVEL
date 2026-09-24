# app/retrieval.py
"""Ranking and hardening of web-search results for grounded generation.

Search providers return a handful of short title + snippet results. Before
they reach the prompt they are:

1. sanitized -- snippets are untrusted third-party text, so anything that
   imitates our prompt template ("### Response:", "Below is an
   instruction...") or hides text (control / zero-width characters) is
   removed, and results phrased as instructions to the model ("ignore all
   previous instructions...") are dropped;
2. ranked by query-term coverage -- the fraction of the query's content
   words (stopwords removed, lightly stemmed) found in title + snippet.
   The previous ranker counted raw shared words including stopwords, so
   "What causes what?" outranked an actual explanation of the seasons;
3. optionally re-ranked by a dense embedding model (RAG_EMBED_MODEL): the
   final score is a weighted sum of coverage and min-max-normalized cosine
   similarity, so a paraphrase no lexical method can match ("tallest" vs
   "highest") can overturn the lexical order. (Reciprocal-rank fusion was
   avoided: with two rankers, any pair they disagree on ties exactly, so
   the dense model could never correct a lexical mistake.)
4. de-duplicated -- near-identical snippets waste the context window.

evals/data/retrieval.jsonl is a labeled set for measuring changes here
(python -m evals --suites retrieval). On it this pipeline scores MRR 0.872
(precision@1 0.792, recall@3 0.938) vs 0.783 (0.625, 0.854) for the
previous ranker; its remaining misses are synonyms ("tallest"/"highest"),
irregular verbs ("sink"/"sank") and exact ties -- what the optional dense
re-ranker is for. Okapi BM25 was tried and scored 0.689: with
IDF computed over only ~5 candidates, a query word that appears in just one
result -- typically an off-topic result matching a single word, like
"Fast food" for "How fast does light travel?" -- gets the highest weight.
Coverage also ignores term frequency, so keyword stuffing doesn't pay.
"""

import difflib
import logging
import os
import re
import threading
import unicodedata

import snowballstemmer

logger = logging.getLogger(__name__)

# Share of the dense score when an embedding model is configured. Untuned:
# no embedding model was available when this was built, so it's an even
# split -- measure with `python -m evals --suites retrieval` and adjust.
DENSE_WEIGHT = float(os.environ.get("RAG_DENSE_WEIGHT", 0.5))

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself yourselves many much
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Text in a snippet that imitates our prompt format could steer the model
# (prompt injection) or corrupt answer parsing.
_TEMPLATE_RE = re.compile(
    r"(#{2,}\s*(instruction|response|input)\s*:?|below is an instruction[^.]*\.?|<\|endoftext\|>)",
    re.IGNORECASE,
)

# Phrasing typical of prompt-injection attempts planted in web pages. A
# heuristic, so it can be evaded -- but a snippet that talks to the model
# instead of about the topic is never a useful source, so dropping it is cheap.
_INJECTION_RE = re.compile(
    r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+"
    r"(instructions?|prompts?|messages?|context)"
    r"|\b(system|developer)\s+prompt\b"
    r"|\bnew\s+instructions?\s*:"
    r"|\bdo\s+not\s+(tell|reveal\s+to)\s+the\s+user\b",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    return _INJECTION_RE.search(text) is not None


def sanitize_snippet(text: str) -> str:
    """Neutralize prompt-template imitation and invisible characters."""
    # Format characters (zero-width, bidi overrides, soft hyphens) are deleted,
    # not spaced out, so "ig\u200bnore" can't split a word past the checks
    # below; control characters (tabs, newlines, bells) become spaces.
    text = "".join(
        "" if (category := unicodedata.category(ch)) == "Cf" else " " if category == "Cc" else ch
        for ch in text
    )
    text = _TEMPLATE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# Snowball (Porter2) English stemmer: maps "causes"/"caused"/"cause" and
# "travels"/"travelling" to shared stems. A hand-rolled suffix stripper was
# tried first and kept producing inconsistent stems ("causes" vs "caused").
_STEMMER = snowballstemmer.stemmer("english")


def _stem(token: str) -> str:
    return _STEMMER.stemWord(token)


def terms(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def coverage_scores(query: str, documents: list[str]) -> list[float]:
    """Fraction of the query's distinct content terms present in each document."""
    query_terms = set(terms(query))
    if not query_terms:
        return [0.0] * len(documents)
    return [len(query_terms & set(terms(d))) / len(query_terms) for d in documents]


class DenseReranker:
    """Optional semantic re-ranking with a sentence-embedding model.

    Enabled when RAG_EMBED_MODEL names a sentence-transformers model (e.g.
    sentence-transformers/all-MiniLM-L6-v2) and that package is installed;
    otherwise ranking is lexical (coverage) only. Loaded lazily, once.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def similarities(self, query: str, documents: list[str]) -> list[float]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode([query] + documents, normalize_embeddings=True)
        return [float(vectors[0] @ v) for v in vectors[1:]]


_reranker_lock = threading.Lock()
_reranker: DenseReranker | None | bool = False  # False = not resolved yet


def get_dense_reranker() -> DenseReranker | None:
    global _reranker
    with _reranker_lock:
        if _reranker is False:
            name = os.environ.get("RAG_EMBED_MODEL", "").strip()
            _reranker = DenseReranker(name) if name else None
        return _reranker


def _normalize(scores: list[float]) -> list[float]:
    low, high = min(scores), max(scores)
    return [(x - low) / (high - low) if high > low else 0.0 for x in scores]


def rank_results(query: str, results: list[dict], reranker: DenseReranker | None = None) -> list[dict]:
    """Order results best-first. Ties keep the provider's order, which
    already encodes the search engine's own relevance signal."""
    if not results:
        return []
    documents = [f"{r['title']} {r['snippet']}" for r in results]
    scores = coverage_scores(query, documents)

    if reranker is not None:
        try:
            dense = _normalize(reranker.similarities(query, documents))
            scores = [(1 - DENSE_WEIGHT) * lex + DENSE_WEIGHT * sem for lex, sem in zip(scores, dense)]
        except Exception:
            logger.exception("Dense re-ranking failed; using lexical ranking only")

    order = sorted(range(len(results)), key=lambda i: (-scores[i], i))
    return [results[i] for i in order]


def select_sources(query: str, results: list[dict], max_results: int = 3,
                   reranker: DenseReranker | None = None, dedup_ratio: float = 0.8) -> list[dict]:
    """Sanitize, rank, de-duplicate, and keep the top `max_results`."""
    cleaned = []
    for r in results:
        snippet = sanitize_snippet(re.sub(r"(?:\.{3,}|…)", "", r.get("snippet", "")))
        title = sanitize_snippet(r.get("title", ""))
        # Checked after sanitizing: that's the text the model will see.
        if looks_like_injection(f"{title} {snippet}"):
            logger.warning("Dropped a search result that looks like a prompt-injection attempt")
            continue
        if snippet and title:
            cleaned.append({**r, "title": title, "snippet": snippet})

    final = []
    for candidate in rank_results(query, cleaned, reranker):
        if any(difflib.SequenceMatcher(None, candidate["snippet"], kept["snippet"]).ratio() > dedup_ratio
               for kept in final):
            continue
        final.append(candidate)
        if len(final) == max_results:
            break
    return final
