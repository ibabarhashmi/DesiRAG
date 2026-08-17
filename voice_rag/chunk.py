"""Multi-strategy, metadata-aware chunking.

Three strategies are implemented and *evaluated* (scripts/evaluate.py) rather
than assumed:

- ``fixed``      — naive fixed-size windows with word overlap (baseline; kept
                   so the comparison has a floor).
- ``recursive``  — paragraph -> sentence splitting, merging sentences bottom-up
                   up to ``max_chunk_tokens`` with ``overlap_words`` carried
                   across boundaries.
- ``semantic``   — sentences embedded with the retrieval encoder; adjacent
                   sentences whose cosine gap crosses a threshold start a new
                   chunk (topic-aware, handles Indic scripts via the shared
                   multilingual embedder).

A length-aware router indexes short passages whole (no pointless
over-fragmentation of a 60-word IR passage) and only chunk-splits passages
longer than ``short_passage_words``. Every chunk keeps its passage metadata
(gold query ids, answers, query type) so retrieval evaluation can score
"retrieved chunk -> gold passage" and answer recall directly.
"""

import re
from dataclasses import dataclass, field

import numpy as np

_SENT_BOUNDARIES = re.compile(r"(?<=[।?!.…])\s+")
_TOKEN = re.compile(r"\S+")


@dataclass
class ChunkConfig:
    strategy: str = "semantic"
    max_chunk_tokens: int = 220
    min_chunk_tokens: int = 110
    short_passage_tokens: int = 80     # under this -> single whole chunk
    overlap_tokens: int = 20           # carried between chunks (recursive/fixed)
    semantic_gap_threshold: float = 0.5  # 1 - cos(adjacent sentence pair)


@dataclass
class Chunk:
    id: str
    passage_id: str
    text: str
    start: int                # char offset of chunk within passage
    end: int
    tokens: int
    meta: dict = field(default_factory=dict)
    strategy: str = ""


def approx_tokens(text: str) -> int:
    """Token-count proxy: whitespace-delimited units. Works for Indic scripts
    (Devanagari words are space-separated too), good enough to size chunks."""
    return len(_TOKEN.findall(text))


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Sentence splitter for Indic + Latin punctuation, with char offsets."""
    out = []
    pos = 0
    for part in _SENT_BOUNDARIES.split(text):
        part = (part or "").strip()
        if not part:
            continue
        start = text.find(part, pos)
        if start < 0:
            start = pos
        end = start + len(part)
        out.append((start, end, part))
        pos = end
    if not out and text.strip():
        out.append((0, len(text), text.strip()))
    return out


def _meta_for(passage) -> dict:
    return {
        "passage_id": passage.pid,
        "answers": passage.answers,
        "query_ids": passage.query_ids,
        "query_types": passage.query_types,
        "lang": "hi",
        "eng": passage.eng,
    }


def _unit_chunk(passage, start: int, end: int, text: str, strategy: str) -> Chunk:
    base = passage.pid.replace(":", "_")
    tokens = approx_tokens(text)
    return Chunk(
        id=f"{base}-{strategy}-{start}-{end}", passage_id=passage.pid,
        text=text.strip(), start=start, end=end, tokens=tokens,
        meta=_meta_for(passage), strategy=strategy)


# --- fixed-size baseline ---------------------------------------------------

def chunk_fixed(passage, cfg: ChunkConfig) -> list[Chunk]:
    words = _TOKEN.findall(passage.text)
    size = max(1, cfg.max_chunk_tokens)
    overlap = max(0, cfg.overlap_tokens)
    chunks = []
    start_i = 0
    n = len(words)
    while start_i < n:
        end_i = min(start_i + size, n)
        seg = " ".join(words[start_i:end_i])
        start = passage.text.find(seg) if seg else 0
        chunks.append(_unit_chunk(passage, start, start + len(seg), seg, "fixed"))
        if end_i >= n:
            break
        start_i = end_i - overlap if end_i - overlap > start_i else end_i
    return chunks


# --- recursive (paragraph -> sentence, sentential merge + overlap) ----------

def sentences_(para: str) -> list[str]:
    return [s for _, _, s in split_sentences(para)]


def _flush_rec(passage, buf: list[str], carry: list[str],
               cfg: ChunkConfig) -> tuple[Chunk, list[str]]:
    text = " ".join(carry + buf).strip()
    start = passage.text.find(text)
    if start < 0:
        start = 0
    ch = _unit_chunk(passage, start, start + len(text), text, "recursive")
    words = _TOKEN.findall(text)
    new_carry = " ".join(words[-cfg.overlap_tokens:]) if (
        cfg.overlap_tokens and len(words) > cfg.overlap_tokens) else ""
    return ch, [new_carry] if new_carry else []


def chunk_recursive(passage, cfg: ChunkConfig) -> list[Chunk]:
    """Sentential top-down: grow a chunk sentence-by-sentence up to
    ``max_chunk_tokens``; the last ``overlap_tokens`` words of a flushed chunk
    are carried into the next one so no answer straddles a boundary."""
    paras = [p.strip() for p in passage.text.split("\n") if p.strip()]
    if not paras:
        paras = [passage.text]
    chunks: list[Chunk] = []
    carry: list[str] = []
    for para in paras:
        buf: list[str] = []
        for sent in sentences_(para):
            if approx_tokens(" ".join(buf + [sent])) > cfg.max_chunk_tokens \
                    and buf:
                ch, carry = _flush_rec(passage, buf, carry, cfg)
                chunks.append(ch)
            buf.append(sent)
        if buf:
            ch, carry = _flush_rec(passage, buf, carry, cfg)
            chunks.append(ch)
    if not chunks:
        chunks.append(_unit_chunk(passage, 0, len(passage.text),
                                  passage.text, "recursive"))
    return chunks


# --- semantic (sentence-embedding discontinuity) ----------------------------

def chunk_semantic(passage, cfg: ChunkConfig, embed_fn=None) -> list[Chunk]:
    sentences = split_sentences(passage.text)
    if len(sentences) <= 1:
        return [_unit_chunk(passage, 0, len(passage.text), passage.text,
                            "semantic")]
    texts = [s for _, _, s in sentences]
    vecs = embed_fn(texts) if embed_fn else None
    gaps = _sentence_gaps(vecs) if vecs is not None else None

    chunks: list[Chunk] = []
    cur: list[tuple] = []
    cur_tokens = 0
    for i, (start, end, sent) in enumerate(sentences):
        cur.append((start, end, sent))
        cur_tokens += approx_tokens(sent)
        will_break = (gaps is not None and i < len(gaps)
                      and gaps[i] > cfg.semantic_gap_threshold
                      and cur_tokens >= cfg.min_chunk_tokens)
        if (cur_tokens >= cfg.max_chunk_tokens) or will_break:
            if i == len(sentences) - 1 or will_break:
                chunks.append(_flush(passage, cur))
                cur, cur_tokens = [], 0
    if cur:
        chunks.append(_flush(passage, cur))
    if not chunks:
        chunks.append(_unit_chunk(passage, 0, len(passage.text), passage.text,
                                  "semantic"))
    return chunks


def _sentence_gaps(vecs: np.ndarray) -> list[float]:
    if vecs.shape[0] < 2:
        return []
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    u = vecs / n
    sims = (u[:-1] * u[1:]).sum(axis=1)
    return list(1.0 - np.clip(sims, -1, 1))


def _flush(passage, sents) -> Chunk:
    start = sents[0][0]
    end = sents[-1][1]
    text = passage.text[start:end].strip()
    return Chunk(
        id=f"{passage.pid.replace(':', '_')}-semantic-{start}-{end}",
        passage_id=passage.pid, text=text, start=start, end=end,
        tokens=approx_tokens(text), meta=_meta_for(passage), strategy="semantic")


# --- router + corpus --------------------------------------------------------

def chunk_passage(passage, cfg: ChunkConfig, embed_fn=None) -> list[Chunk]:
    """Length-aware router: short passages stay whole, long ones get the
    configured strategy."""
    if approx_tokens(passage.text) <= max(cfg.short_passage_tokens, 1):
        return [_unit_chunk(passage, 0, len(passage.text), passage.text, "whole")]
    if cfg.strategy == "fixed":
        return chunk_fixed(passage, cfg)
    if cfg.strategy == "semantic" and embed_fn is not None:
        return chunk_semantic(passage, cfg, embed_fn)
    return chunk_recursive(passage, cfg)


def chunk_corpus(passages, cfg: ChunkConfig, embed_fn=None) -> list[Chunk]:
    """Chunk all corpus passages, assigning globally unique chunk ids."""
    chunks: list[Chunk] = []
    for p in passages:
        for c in chunk_passage(p, cfg, embed_fn):
            chunks.append(c)
    ids = set()
    for c in chunks:
        base = c.id
        k = 1
        while c.id in ids:
            c.id = f"{base}#{k}"
            k += 1
        ids.add(c.id)
    return chunks