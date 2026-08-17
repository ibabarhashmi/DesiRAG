"""Chunking unit tests: sentence splitting, strategies, overlap, metadata,
length-aware router."""

from voice_rag.chunk import (
    ChunkConfig,
    approx_tokens,
    chunk_corpus,
    chunk_fixed,
    chunk_passage,
    chunk_recursive,
    chunk_semantic,
    split_sentences,
)
from voice_rag.data import Passage


def test_approx_tokens():
    assert approx_tokens("") == 0
    assert approx_tokens("a b c") == 3
    assert approx_tokens("यह एक हिंदी वाक्य है") == 5


def test_split_sentences_hindi_and_english():
    text = "पहला वाक्य। दूसरा वाक्य! तीसरा वाक्य?"
    sents = split_sentences(text)
    assert [s for _, _, s in sents] == ["पहला वाक्य।", "दूसरा वाक्य!",
                                        "तीसरा वाक्य?"]
    offsets = [(a, b) for a, b, _ in sents]
    assert offsets[0][1] <= offsets[1][0]  # contiguous, ordered


def test_short_passage_indexed_whole():
    p = Passage("0:0", "Short passage that fits in one chunk.", "")
    cfg = ChunkConfig(strategy="fixed", short_passage_tokens=80)
    chunks = chunk_passage(p, cfg)
    assert len(chunks) == 1 and chunks[0].strategy == "whole"
    assert chunks[0].text == p.text


def test_fixed_chunks_carry_overlap():
    p = Passage("0:0", "one two three four five six seven eight nine ten " * 6,
                "")
    cfg = ChunkConfig(strategy="fixed", max_chunk_tokens=10, overlap_tokens=3)
    chunks = chunk_fixed(p, cfg)
    assert len(chunks) >= 3
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert all(c.tokens <= 10 for c in chunks)


def test_recursive_chunks_metadata_preserved():
    p = Passage("7:2",
                "Paragraph one sentence here. Paragraph one second sentence. "
                "\nParagraph two has its own topic. And continues with more.",
                "", [42], ["gold answer"], ["DESCRIPTION"])
    cfg = ChunkConfig(strategy="recursive", max_chunk_tokens=6,
                      overlap_tokens=2)
    chunks = chunk_recursive(p, cfg)
    assert chunks
    for c in chunks:
        assert c.passage_id == "7:2"
        assert c.meta["query_ids"] == [42]
        assert c.meta["answers"] == ["gold answer"]
        assert c.meta["query_types"] == ["DESCRIPTION"]


def test_semantic_chunking_with_embeddings():
    p = Passage("0:0",
                "Cricket is a bat and ball sport. It is popular in India. "
                "Quantum chromodynamics describes the strong force. "
                "Gluons mediate it at short ranges.", "")
    cfg = ChunkConfig(strategy="semantic", max_chunk_tokens=40,
                      min_chunk_tokens=3, semantic_gap_threshold=0.5)

    class FakeEmb:
        def __call__(self, texts):
            import numpy as np
            v = np.zeros((len(texts), 2), dtype=np.float32)
            for i, t in enumerate(texts):
                tl = t.lower()
                if any(w in tl for w in
                       ("cricket", "sport", "india", "bat", "ball")):
                    v[i, 0] = 1.0
                if any(w in tl for w in ("quantum", "gluon", "strong", "force")):
                    v[i, 1] = 1.0
                n = np.linalg.norm(v[i])
                if n:
                    v[i] = v[i] / n
            return v

    chunks = chunk_semantic(p, cfg, FakeEmb())
    # two distinct topics must surface as >= 2 chunks (sentence 3 is a
    # discontinuous jump from the cricket topic)
    assert len(chunks) >= 2
    texts = [c.text.lower() for c in chunks]
    assert any("quantum" in t for t in texts)
    assert any("cricket" in t for t in texts)


def test_corpus_ids_unique():
    ps = [
        Passage("0:0", "same sentence here. " * 4, ""),
        Passage("0:1", "different text entirely. " * 4, ""),
    ]
    cfg = ChunkConfig(strategy="fixed", max_chunk_tokens=8, overlap_tokens=2)
    chunks = chunk_corpus(ps, cfg)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))