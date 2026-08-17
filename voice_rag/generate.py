"""Grounded answer generation.

Default: **extractive** — the answer is the single passage sentence whose
token-F1 over the query is highest, taken from what retrieval actually
returned. Being a span of the context by construction, it *cannot*
hallucinate; ``confidence`` is the F1 and the guardrail refuses when it is too
low.

Optional: a fluent **LLM** mode reuses the same OpenAI-compatible proxy the
sibling ``assessoraudit`` project calls. Its output is *not* trusted as-is —
``guardrails.grounded_result`` must confirm every claim attaches to a
retrieved chunk or the harness falls back to extractive/refusal.
"""

import os
from dataclasses import dataclass

import httpx

from .chunk import approx_tokens, split_sentences
from .config import Settings
from .text import tokenize

_LANG_NAMES = {"hi": "Hindi", "en": "English", "ta": "Tamil", "bn": "Bengali",
               "te": "Telugu", "mr": "Marathi", "kn": "Kannada",
               "gu": "Gujarati"}


@dataclass
class Extracted:
    text: str
    confidence: float
    citations: list[dict]
    source_chunk_id: str
    strategy: str = "extractive"


def _shingles(text: str) -> set[str]:
    """Character bigrams of the space-stripped lowercase text (padded).

    Shingles survive Hindi compound spellings the way token-F1 cannot:
    ``ताज महल`` and ``ताजमहल`` produce the same bigrams, and minor
    transliteration noise barely moves the score."""
    s = "".join(text.lower().split())
    if not s:
        return set()
    if len(s) == 1:
        s = s * 2
    s = "#" + s + "#"
    return {s[i:i + 2] for i in range(len(s) - 1)}


def extract_overlap(query: str, sentence: str) -> float:
    sa, sb = _shingles(query), _shingles(sentence)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return 2.0 * inter / (len(sa) + len(sb))


def extractive_answer(query: str, hits, min_overlap: float = 0.18,
                      top_n: int = 3, min_tokens: int = 6,
                      max_tokens: int = 45) -> Extracted | None:
    """Best sentence from the top retrieved contexts, scored by query F1.

    If the best sentence is a short fragment (common when the query is a
    "what is X" and the passage opens with ``X.`` + definition), it is
    extended with the following sentence of the same chunk so the answer
    carries the definition rather than a bare noun.
    """
    qt = tokenize(query)
    best: tuple[float, int, int, int] | None = None  # (f1, rank, sent_idx, len)
    for rank, h in enumerate(hits[:top_n]):
        for si, (_, _, sent) in enumerate(split_sentences(h.chunk.text)):
            f1 = extract_overlap(query, sent)
            cand = (f1, rank, si, len(tokenize(sent)))
            if best is None or cand > best:  # tuple order: f1 > rank > idx > len
                best = cand
    if best is None or best[0] < min_overlap:
        return None
    f1, rank, si, _ = best
    text = hits[rank].chunk.text
    sents = split_sentences(text)
    s, e, sent = sents[si]
    if len(tokenize(sent)) < min_tokens and si + 1 < len(sents):
        _, e2, _ = sents[si + 1]
        extended = text[s:e2].strip()
        if approx_tokens(extended) <= max_tokens:
            sent = extended

    citations = _citations_for(hits, rank)
    return Extracted(
        text=sent, confidence=float(max(f1, extract_overlap(query, sent))),
        citations=citations, source_chunk_id=hits[rank].chunk.id)


def _citations_for(hits, up_to_rank: int) -> list[dict]:
    out = []
    seen: set[str] = set()
    for h in hits[: up_to_rank + 1]:
        if h.chunk.id in seen:
            continue
        seen.add(h.chunk.id)
        out.append({
            "chunk_id": h.chunk.id, "passage_id": h.chunk.passage_id,
            "text": h.chunk.text[:400],
            "vector_sim": round(h.vector_sim, 4),
            "bm25_score": round(h.bm25_score, 4),
            "rank": len(out),
        })
    return out


class LLMGenerator:
    """Optional fluent-answer mode over the pluggable OpenAI-compatible proxy."""

    def __init__(self, settings: Settings):
        self.base = (settings.llm_base_url or "").rstrip("/")
        self.key = settings.llm_api_key
        self.model = settings.llm_model or os.getenv("ANTHROPIC_MODEL", "")
        self.lang = settings.lang

    @property
    def available(self) -> bool:
        return bool(self.base and self.key)

    def generate(self, query: str, hits, max_tokens: int = 90,
                 timeout: float = 20.0) -> tuple[str, list[dict]] | None:
        """Return (answer_text, citations) or None on any failure."""
        if not self.available or not hits:
            return None
        lang = _LANG_NAMES.get(self.lang, self.lang)
        system = (
            f"You answer questions grounded ONLY in the numbered passages "
            f"below, in {lang}. "
            "Answer in 1-3 sentences. After each factual sentence, add the "
            "citation of the passage it came from as [n]. If no passage "
            "supports the answer, reply exactly: NO_GROUNDED_ANSWER")
        ctx = "\n\n".join(
            f"[{i+1}] {h.chunk.text.strip()}" for i, h in enumerate(hits[:6]))
        user = f"Question: {query}\n\nPassages:\n{ctx}"
        url = f"{self.base}/chat/completions"
        r = httpx.post(
            url, headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}],
                "max_tokens": max_tokens, "temperature": 0.0},
            timeout=timeout)
        r.raise_for_status()
        text = (r.json().get("choices") or [{}])[0].get("message", {}).get(
            "content", "")
        if not text or "NO_GROUNDED_ANSWER" in text.upper():
            return None
        return text.strip(), [{
            "chunk_id": h.chunk.id, "passage_id": h.chunk.passage_id,
            "text": h.chunk.text[:400],
            "vector_sim": round(h.vector_sim, 4),
            "bm25_score": round(h.bm25_score, 4), "rank": i}
            for i, h in enumerate(hits[:6])]