"""ChunkStore: the vector index (semantic) + lexical index (BM25) + fusion.

Design notes (why this and not a heavier vector DB):
- At demo corpus scales (~20-50k chunks x 384-dim float32) an exact cosine
  top-k is a single (N x D) matmul: microseconds-to-low-milliseconds, *and*
  exact rather than ANN-approximate. A separate DB server (FAISS/Qdrant/...) is
  pure latency and ops overhead here, so it is skipped by the lazy rule "reuse
  what's installed / correct" — the swap to HNSW is one method behind
  ``VectorBackend`` if the corpus ever outgrows a few hundred thousand chunks.
- BM25 is implemented self-contained (bigram vocab -> inverted lists as numpy
  arrays), no scipy needed, and gives the lexical half of hybrid retrieval for
  rare terms / Indic spellings that embeddings smooth over. Bigram indexing
  (``text.bigram_terms``) is deliberate: Hindi word boundaries are orthographic
  (``ताज महल`` vs ``ताजमहल``), and exact token matching drops the compound
  spelling entirely; bigrams keep both forms matchable.
- Fusion is reciprocal-rank fusion (RRF): stable, hyperparameter-light, and it
  produces one ``score`` that guardrails can threshold on.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunk import Chunk
from .text import bigram_terms, tokenize


@dataclass
class Hit:
    idx: int
    chunk: Chunk
    score: float          # fused (RRF) score, higher = better
    vector_sim: float
    bm25_score: float
    bm25_bigram: float = 0.0
    lexical_coverage: float = 0.0   # fraction of query tokens seen in this chunk


class _BM25:
    """Vocabulary-level inverted index over chunks with BM25 scoring.

    The index is parameterized by a term function: ``tokenize`` for exact
    tokens (precision) and ``text.bigram_terms`` for padded character bigrams
    (recall). Bigrams map Hindi word-boundary variants (``ताज महल`` and
    ``ताजमहल``) onto overlapping terms that exact tokens miss entirely; token
    bigrams alone are too noisy to lead with, so both indexes coexist and the
    rank lists are merged with reciprocal-rank fusion in ``ChunkStore.search``.
    """

    def __init__(self, doc_len: np.ndarray, avg_len: float,
                 term_fn=tokenize):
        self.doc_len = doc_len
        self.avg_len = avg_len
        self.term_fn = term_fn
        self.vocab: dict[str, int] = {}
        self.offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self.chunks: np.ndarray = np.zeros(0, dtype=np.int32)
        self.tfs: np.ndarray = np.zeros(0, dtype=np.float32)
        self.dfs: np.ndarray = np.zeros(0, dtype=np.int32)

    @classmethod
    def build(cls, texts: list[str], term_fn=tokenize) -> "_BM25":
        lens = np.array([len(term_fn(t)) for t in texts],
                        dtype=np.float32)
        lens[lens == 0] = 1.0
        avg_len = float(lens.mean()) if len(lens) else 1.0
        bm = cls(lens, avg_len, term_fn)
        term_chunks: dict[str, list[tuple[int, int]]] = {}
        for ci, doc in enumerate((term_fn(t) for t in texts)):
            seen: dict[str, int] = {}
            for w in doc:
                seen[w] = seen.get(w, 0) + 1
            for w, tf in seen.items():
                term_chunks.setdefault(w, []).append((ci, tf))
        bm.vocab = {w: i for i, w in enumerate(term_chunks)}
        n = len(term_chunks)
        caps = np.array([len(v) for v in term_chunks.values()], dtype=np.int64)
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(caps)
        chunks = np.empty(int(offsets[-1]), dtype=np.int32)
        tfs = np.empty(int(offsets[-1]), dtype=np.float32)
        dfs = np.zeros(n, dtype=np.int32)
        for w, pairs in term_chunks.items():
            i = bm.vocab[w]
            if not pairs:
                continue
            ci, tf = zip(*pairs)
            ci = np.asarray(ci, dtype=np.int32)
            tf = np.asarray(tf, dtype=np.float32)
            chunks[offsets[i]:offsets[i + 1]] = ci
            tfs[offsets[i]:offsets[i + 1]] = tf
            dfs[i] = len(ci)
        bm.offsets, bm.chunks, bm.tfs, bm.dfs = offsets, chunks, tfs, dfs
        return bm

    def score(self, query_text: str, k1: float = 1.5, b: float = 0.75,
              k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return (tops indices, tops scores) for BM25 on ``query_text``."""
        ndocs = len(self.doc_len)
        scores = np.zeros(ndocs, dtype=np.float32)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        qterms = self.term_fn(query_text)
        for w in qterms:
            i = self.vocab.get(w)
            if i is None:
                continue
            df = self.dfs[i]
            if df == 0:
                continue
            idf[i] = np.log(1 + (ndocs - df + 0.5) / (df + 0.5))
        for w in qterms:
            i = self.vocab.get(w)
            if i is None or self.dfs[i] == 0:
                continue
            s, e = self.offsets[i], self.offsets[i + 1]
            if s >= e:
                continue
            cids = self.chunks[s:e]
            tf = self.tfs[s:e]
            dl = self.doc_len[cids]
            denom = tf + k1 * (1 - b + b * dl / self.avg_len)
            term_scores = idf[i] * tf * (k1 + 1) / denom
            np.add.at(scores, cids, term_scores)
        nz = np.count_nonzero(scores)
        if nz == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        kk = min(k, nz)
        idx = np.argpartition(scores, -kk)[-kk:]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]


class VectorBackend:
    """Exact-cosine semantic top-k over a normalized (N x D) float32 matrix."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors  # normalized, float32

    def topk(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.vectors.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        sims = self.vectors @ q
        kk = min(k, sims.shape[0])
        idx = np.argpartition(-sims, kk - 1)[:kk]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]


class ChunkStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray,
                 bm25: _BM25 | None = None,
                 bm25_bigram: _BM25 | None = None):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = bm25 or _BM25(np.ones(len(chunks), dtype=np.float32), 1.0)
        self.bm25_bigram = bm25_bigram
        self.vector = VectorBackend(vectors)

    # --- persistence --------------------------------------------------------

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({
                    "id": c.id, "passage_id": c.passage_id, "text": c.text,
                    "start": c.start, "end": c.end, "tokens": c.tokens,
                    "meta": c.meta, "strategy": c.strategy},
                    ensure_ascii=False) + "\n")
        extra: dict[str, object] = {}
        if self.bm25_bigram is not None:
            extra.update(
                bg_chunks=self.bm25_bigram.chunks, bg_tfs=self.bm25_bigram.tfs,
                bg_dfs=self.bm25_bigram.dfs, bg_offsets=self.bm25_bigram.offsets,
                bg_doc_len=self.bm25_bigram.doc_len,
                bg_avg_len=np.float32(self.bm25_bigram.avg_len))
        np.savez_compressed(directory / "lex.npz",
                            chunks=self.bm25.chunks, tfs=self.bm25.tfs,
                            dfs=self.bm25.dfs, offsets=self.bm25.offsets,
                            doc_len=self.bm25.doc_len,
                            avg_len=np.float32(self.bm25.avg_len), **extra)
        with (directory / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(self.bm25.vocab, f, ensure_ascii=False)
        if self.bm25_bigram is not None:
            with (directory / "vocab_bigram.json").open("w", encoding="utf-8") as f:
                json.dump(self.bm25_bigram.vocab, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path) -> "ChunkStore":
        vectors = np.load(directory / "vectors.npy")
        chunks: list[Chunk] = []
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                chunks.append(Chunk(d["id"], d["passage_id"], d["text"],
                                    d["start"], d["end"], d["tokens"],
                                    d.get("meta", {}), d.get("strategy", "")))
        z = np.load(directory / "lex.npz")
        bm25 = _BM25(z["doc_len"], float(z["avg_len"]))
        bm25.offsets, bm25.chunks, bm25.tfs, bm25.dfs = (
            z["offsets"], z["chunks"], z["tfs"], z["dfs"])
        with (directory / "vocab.json").open(encoding="utf-8") as f:
            bm25.vocab = json.load(f)
        bm25_bigram = None
        if "bg_chunks" in z:
            from .text import bigram_terms
            bm25_bigram = _BM25(z["bg_doc_len"], float(z["bg_avg_len"]),
                                bigram_terms)
            bm25_bigram.offsets, bm25_bigram.chunks, bm25_bigram.tfs = (
                z["bg_offsets"], z["bg_chunks"], z["bg_tfs"])
            bm25_bigram.dfs = z["bg_dfs"]
            with (directory / "vocab_bigram.json").open(encoding="utf-8") as f:
                bm25_bigram.vocab = json.load(f)
        return cls(chunks, vectors, bm25, bm25_bigram)

    # --- search -------------------------------------------------------------

    def search(self, q_vec: np.ndarray, query_text: str, k: int = 6,
               top_sem: int = 50, top_lex: int = 50,
               rrf_k: float = 60.0,
               mode: str = "semantic") -> list[Hit]:
        """Retrieve and fuse the available rankers.

        ``mode="semantic"`` (default) is exact-cosine top-k only — the measured
        best on this corpus. ``mode="hybrid"`` additionally fuses the token
        BM25 list via reciprocal-rank fusion. The bigram index, when built, is
        used *only* as an optional recall rescue (``mode="hybrid_bigram"``):
        its top answers are too noisy to lead, so it just contributes rank
        credit and never displaces a higher-ranked semantic/token candidate's
        own evidence.
        """
        v_idx, v_scores = self.vector.topk(q_vec, top_sem)
        rrf: dict[int, float] = {}
        vinfo: dict[int, float] = {}
        linfo: dict[int, float] = {}
        bgmap: dict[int, float] = {}
        rank = 0
        for i, s in zip(v_idx.tolist(), v_scores.tolist()):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
            vinfo[i] = float(np.clip(s, -1, 1))
            rank += 1
        if mode != "semantic":
            l_idx, l_scores = self.bm25.score(query_text, k=top_lex)
            rank = 0
            for i, s in zip(l_idx.tolist(), l_scores.tolist()):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
                linfo[i] = float(s)
                rank += 1
            if mode == "hybrid_bigram" and self.bm25_bigram is not None:
                bg_idx, bg_scores = self.bm25_bigram.score(query_text,
                                                           k=top_lex)
                rank = 0
                for i, s in zip(bg_idx.tolist(), bg_scores.tolist()):
                    rrf[i] = rrf.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
                    bgmap[int(i)] = float(s)
                    if i not in linfo:
                        linfo[i] = float(s)
                    rank += 1
        if not rrf:
            return []
        qterms = set(tokenize(query_text))
        hits = []
        for i, score in sorted(rrf.items(), key=lambda kv: -kv[1])[:k]:
            hits.append(Hit(
                idx=int(i), chunk=self.chunks[int(i)], score=float(score),
                vector_sim=vinfo.get(int(i), 0.0),
                bm25_score=linfo.get(int(i), 0.0),
                bm25_bigram=bgmap.get(int(i), 0.0),
                lexical_coverage=self._coverage(i, qterms)))
        return hits

    def _coverage(self, idx: int, qterms: set[str]) -> float:
        if not qterms:
            return 1.0
        c = set(tokenize(self.chunks[idx].text))
        return len(qterms & c) / len(qterms)

    def max_similarity(self, q_vec: np.ndarray) -> float:
        """Best cosine similarity over the whole corpus (off-topic gate)."""
        if self.vectors.shape[0] == 0:
            return 0.0
        return float(np.max(self.vectors @ q_vec))


def build_store(chunks: list[Chunk], vectors: np.ndarray,
                use_bigram: bool = False) -> ChunkStore:
    if len(chunks) != vectors.shape[0]:
        raise ValueError("chunks/vectors length mismatch")
    texts = [c.text for c in chunks]
    bm25 = _BM25.build(texts)
    bm25_bigram = _BM25.build(texts, bigram_terms) if use_bigram else None
    return ChunkStore(chunks, vectors, bm25, bm25_bigram)