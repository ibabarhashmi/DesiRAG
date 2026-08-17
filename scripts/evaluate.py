"""Chunking-strategy + retrieval evaluation (the "real thought" proof).

Builds three comparable mini-stores (same corpus, same held-out queries; only
the chunking strategy differs) and reports, per strategy:

- recall@5   — fraction of queries whose gold passage surfaces in top-5
- answer-F1  — extractive answer token-F1 against the gold answer
- p50 latency — retrieve+extract, in ms

Run:
    python scripts/evaluate.py [--n-queries 30] [--mini 2000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice_rag.chunk import ChunkConfig, chunk_corpus
from voice_rag.config import get_settings
from voice_rag.data import read_corpus, read_eval
from voice_rag.embed import Embedder
from voice_rag.generate import extractive_answer
from voice_rag.guardrails import ijac
from voice_rag.index import build_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=30)
    ap.add_argument("--mini", type=int, default=2000)
    args = ap.parse_args()

    s = get_settings(refresh=True)
    passages = read_corpus(s.corpus_jsonl)
    evals = read_eval(s.eval_jsonl)[:args.n_queries]

    gold_texts = {e.qid: e.gold_text for e in evals}
    gold_pids = {e.qid: e.gold_pid for e in evals}
    # mini-corpus: gold passages of the eval queries + a sample of the rest
    gold_by_text = {e.gold_text for e in evals if e.gold_text}
    gold_seen = [p for p in passages if p.text in gold_by_text]
    filler = [p for p in passages if p.text not in gold_by_text][:args.mini]
    mini = gold_seen + filler
    print(f"mini-corpus: {len(mini)} passages  | eval queries: {len(evals)}")

    embedder = Embedder(s.embed_model)
    qvecs = embedder.encode([e.query for e in evals])
    qtexts = [e.query for e in evals]

    strategies = ["fixed", "recursive", "semantic"]
    print("\nstrategy   chunks  recall@5  answerF1  retr_ms  gen_ms")
    best = None
    for st in strategies:
        cfg = ChunkConfig(strategy=st, max_chunk_tokens=s.max_chunk_tokens,
                          short_passage_tokens=s.short_passage_words,
                          overlap_tokens=s.overlap_words)
        chunks = chunk_corpus(mini, cfg,
                              embed_fn=embedder.encode if st == "semantic" else None)
        vecs = embedder.encode([c.text for c in chunks])
        store = build_store(chunks, vecs)
        # map chunk -> gold pid reachable
        hits = recall = f1 = 0
        retr_ms = gen_ms = 0.0
        for qi, (qv, qt) in enumerate(zip(qvecs, qtexts)):
            t0 = time.perf_counter()
            res = store.search(qv, qt, k=5, rrf_k=s.rrf_k)
            retr_ms += (time.perf_counter() - t0) * 1000
            got_pid = {h.chunk.passage_id for h in res}
            if gold_pids[evals[qi].qid] in got_pid:
                recall += 1
            t0 = time.perf_counter()
            ex = extractive_answer(qt, res, min_overlap=0.0)
            gen_ms += (time.perf_counter() - t0) * 1000
            if ex:
                f1 += ijac(ex.text, evals[qi].answer)
        n = len(evals)
        rec = recall / n
        af1 = f1 / n
        print(f"{st:<10} {len(chunks):<6} {rec:9.3f} {af1:9.3f} "
              f"{retr_ms/n:8.1f} {gen_ms/n:7.1f}")
        if best is None or rec > best[1] or (rec == best[1] and af1 > best[2]):
            best = (st, rec, af1)
    print(f"\n-> best strategy: {best[0]} (recall@5={best[1]:.3f}, "
          f"answerF1={best[2]:.3f})")


if __name__ == "__main__":
    main()