"""Index tests: exact semantic top-k, BM25, hybrid fusion, persistence."""

import numpy as np
import pytest

from voice_rag.index import ChunkStore, _BM25, build_store
from voice_rag.chunk import Chunk

TEXTS = [
    "the taj mahal is in agra india",          # ch0
    "new delhi is the capital of india",       # ch1
    "ganges river flows through india",        # ch2
]


def _store():
    chunks = [Chunk(f"c{i}", f"p{i}", t, 0, len(t), len(t.split()), {}, "x")
              for i, t in enumerate(TEXTS)]
    # one-hot vectors (deterministic)
    vecs = np.eye(3, dtype=np.float32)  # orthogonal
    return build_store(chunks, vecs)


def test_semantic_topk_returns_best():
    store = _store()
    idx, scores = store.vector.topk(np.array([1.0, 0, 0], dtype=np.float32), 1)
    assert store.chunks[idx[0]].id == "c0"
    assert scores[0] == pytest.approx(1.0)


def test_bm25_scores_and_idf():
    store = _store()
    idx, scores = store.bm25.score("taj agra", k=2)
    assert len(idx) == 1  # only ch0 contains both query terms
    assert store.chunks[idx[0]].id == "c0"
    # unknown term -> no hits
    idx2, _ = store.bm25.score("zzzzqqq", k=2)
    assert len(idx2) == 0


def test_bigram_bm25_tolerates_compound_spelling():
    # Hindi word boundaries are orthographic: ``ताज महल`` and ``ताजमहल`` are
    # the same concept, and an exact token index would miss the compound form.
    texts = [
        "ताज महल आगरा में है और शाहजहाँ ने बनवाया",  # spaced (c0)
        "ताजमहल आगरा में स्थित है",                  # compound (c1)
        "गंगा नदी हिमालय से बहती है",                # unrelated (c2)
    ]
    chunks = [Chunk(f"c{i}", f"p{i}", t, 0, len(t), len(t.split()), {}, "x")
              for i, t in enumerate(texts)]
    store = build_store(chunks, np.eye(3, dtype=np.float32), use_bigram=True)
    # token BM25 alone drops the compound doc entirely...
    t_idx, _ = store.bm25.score("ताज महल क्या है", k=3)
    assert 1 not in t_idx.tolist()
    # ...while the bigram index links both spellings to the top.
    bg_idx, bg_scores = store.bm25_bigram.score("ताज महल क्या है", k=3)
    assert sorted(bg_idx[:2].tolist()) == [0, 1]
    assert bg_scores[1] > 0
    assert bg_idx[2] == 2 and bg_scores[1] > bg_scores[2]


def test_bigram_index_off_by_default_and_rescue_mode():
    texts = ["ताज महल आगरा में है", "ताजमहल आगरा में स्थित है"]
    chunks = [Chunk(f"c{i}", f"p{i}", t, 0, len(t), len(t.split()), {}, "x")
              for i, t in enumerate(texts)]
    store = build_store(chunks, np.eye(2, dtype=np.float32))  # bigram off
    assert store.bm25_bigram is None
    # hybrid without the bigram layer must not crash
    hits = store.search(np.array([1.0, 0], dtype=np.float32), "ताज महल", k=2,
                        mode="hybrid")
    assert len(hits) == 2


def test_hybrid_bigram_rescue_surfaces_compound_doc():
    texts = [
        "ताज महल आगरा में है और शाहजहाँ ने बनवाया",  # spaced (c0)
        "ताजमहल आगरा में स्थित है",                  # compound (c1)
        "गंगा नदी हिमालय से बहती है",                # unrelated (c2)
    ]
    chunks = [Chunk(f"c{i}", f"p{i}", t, 0, len(t), len(t.split()), {}, "x")
              for i, t in enumerate(texts)]
    store = build_store(chunks, np.eye(3, dtype=np.float32), use_bigram=True)
    q = np.array([0.7, 0.7, 0.2], dtype=np.float32)
    hits = store.search(q, "ताज महल क्या है", k=3, mode="hybrid_bigram")
    top_ids = {h.chunk.id for h in hits}
    assert "c1" in top_ids  # compound-spelled doc reaches the fused top-k
    assert hits[0].chunk.id in ("c0", "c1")


def test_hybrid_search_fused_and_sorted():
    store = _store()
    q = np.array([1.0, 0, 0], dtype=np.float32)
    hits = store.search(q, "taj agra", k=3, rrf_k=60)
    assert hits and hits[0].chunk.id == "c0"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].vector_sim == pytest.approx(1.0)
    assert 0.0 <= hits[0].lexical_coverage <= 1.0


def test_store_save_load_roundtrip(tmp_path):
    store = _store()
    store.save(tmp_path)
    loaded = ChunkStore.load(tmp_path)
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.vectors.shape == store.vectors.shape
    idx, _ = loaded.vector.topk(np.array([0, 1.0, 0], dtype=np.float32), 1)
    assert loaded.chunks[idx[0]].id == "c1"
    l_idx, l_sc = loaded.bm25.score("capital delhi", k=1)
    assert loaded.chunks[l_idx[0]].id == "c1"


def test_bm25_empty_corpus():
    bm = _BM25.build([])
    idx, sc = bm.score("anything", k=5)
    assert len(idx) == 0 and len(sc) == 0