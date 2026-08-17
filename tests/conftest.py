"""Shared test fixtures: fake embedder + fake store + tiny real index, so the
harness/guardrail paths run without the 80k corpus or any network.
"""

import numpy as np
import pytest

from voice_rag.chunk import Chunk
from voice_rag.config import Settings
from voice_rag.data import Passage
from voice_rag.embed import Embedder
from voice_rag.harness import RAGPipeline
from voice_rag.index import ChunkStore, Hit, build_store
from voice_rag.stt import NoopSTT


class FakeEmbedder:
    """Deterministic bag-of-words -> normalized one-hot-ish vectors."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.dim = 0

    def _vec(self, text: str) -> np.ndarray:
        toks = text.split()
        for w in toks:
            if w not in self.vocab:
                self.vocab[w] = self.dim
                self.dim += 1
        v = np.zeros(self.dim, dtype=np.float32)
        for w in toks:
            v[self.vocab[w]] = 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def make_chunks():
    return [
        Chunk("taj-whole", "0:0",
              "The Taj Mahal is a white marble mausoleum in Agra, India. "
              "It was built by Shah Jahan in memory of his wife Mumtaz Mahal.",
              0, 100, 20, {"passage_id": "0:0"}, "whole"),
        Chunk("ganga-whole", "0:1",
              "The Ganges is a trans-boundary river of Asia that flows "
              "through India and Bangladesh. It is the most sacred river "
              "to Hindus.", 0, 110, 20, {"passage_id": "0:1"}, "whole"),
        Chunk("delhi-whole", "1:0",
              "New Delhi is the capital of India and part of the National "
              "Capital Territory of Delhi.", 0, 90, 14,
              {"passage_id": "1:0"}, "whole"),
    ]


@pytest.fixture
def tiny_store():
    """A real ChunkStore (semantic + BM25) over 3 synthetic passages."""
    from voice_rag.index import build_store
    chunks = make_chunks()
    # one-hot semantic vectors via FakeEmbedder
    emb = FakeEmbedder()
    vecs = emb.encode([c.text for c in chunks])
    return build_store(chunks, vecs)


def fake_hits(chunk_texts, sims, scores=None, coverages=None):
    """Build Hit objects for the harness without a real store."""
    chunks = []
    for i, t in enumerate(chunk_texts):
        chunks.append(Chunk(f"c{i}", f"p{i}", t, 0, len(t), len(t.split()),
                            {"passage_id": f"p{i}"}, "test"))
    scores = scores or [0.02] * len(chunk_texts)
    coverages = coverages or [1.0] * len(chunk_texts)
    return [Hit(i, c, float(scores[i]), float(sims[i]), 0.0,
                float(coverages[i])) for i, c in enumerate(chunks)]


class FakeStore:
    """Programmable store whose search() returns the configured hits."""

    def __init__(self, hits):
        self._hits = hits

    def search(self, qvec, query_text, k=6, rrf_k=60.0, mode="semantic"):
        return list(self._hits)


@pytest.fixture
def pipeline(fake_embedder, settings):
    hits = fake_hits(
        ["The Taj Mahal is a white marble mausoleum in Agra, India.",
         "New Delhi is the capital of India."],
        sims=[0.90, 0.80])
    return RAGPipeline(settings, FakeStore(hits), fake_embedder,
                       stt=NoopSTT("where is the taj mahal"))


@pytest.fixture
def sample_passages():
    return [
        Passage("0:0",
                "The Taj Mahal is a white marble mausoleum in Agra, India. "
                "It was built by Shah Jahan in memory of his wife Mumtaz "
                "Mahal. It is one of the seven wonders of the world. "
                "Construction began around 1632 and took over twenty years.",
                "", [1], ["The Taj Mahal is in Agra."], ["DESCRIPTION"]),
        Passage("0:1",
                "The Ganges is a trans-boundary river of Asia. It flows "
                "through India and Bangladesh.",
                "", [2], [], []),
    ]