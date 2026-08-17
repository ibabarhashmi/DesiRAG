"""Build the index end-to-end: raw parquet -> corpus/eval slices -> chunks
(winner strategy) -> embeddings -> hybrid ChunkStore persisted to data/index/.

Usage:
    python scripts/build_index.py [--strategy semantic|recursive|fixed]
                                  [--n-corpus 40000] [--n-eval 200]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_rag.chunk import ChunkConfig, chunk_corpus
from voice_rag.config import get_settings
from voice_rag.data import build_slices, read_corpus
from voice_rag.embed import Embedder
from voice_rag.index import build_store


def ensure_raw(settings) -> Path:
    if settings.raw_parquet.exists():
        return settings.raw_parquet
    hits = list(settings.raw_parquet.parent.rglob("*.parquet"))
    if hits:
        return hits[0]
    print(f"raw file missing -> downloading to {settings.raw_parquet.parent} ...")
    from huggingface_hub import hf_hub_download
    settings.raw_parquet.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download("ai4bharat/MSMARCO-XI",
                    f"validation/{settings.lang}inval.parquet",
                    repo_type="dataset", local_dir=str(settings.raw_parquet.parent))
    return ensure_raw(settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["semantic", "recursive", "fixed"])
    ap.add_argument("--n-corpus", type=int)
    ap.add_argument("--n-eval", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    s = get_settings(refresh=True)
    if args.strategy:
        s.chunk_strategy = args.strategy
    if args.n_corpus:
        s.n_corpus_examples = args.n_corpus
    if args.n_eval:
        s.n_eval_queries = args.n_eval

    s.raw_parquet = ensure_raw(s)
    corpus_path, eval_path = build_slices(s, force=args.force)
    passages = read_corpus(corpus_path)
    print(f"corpus passages: {len(passages)}  | eval queries: "
          f"{len(eval_path.read_text(encoding='utf-8').splitlines())}")

    embedder = Embedder(s.embed_model)
    cfg = ChunkConfig(
        strategy=s.chunk_strategy, max_chunk_tokens=s.max_chunk_tokens,
        short_passage_tokens=s.short_passage_words,
        overlap_tokens=s.overlap_words)
    print(f"chunking strategy: {s.chunk_strategy}")
    chunks = chunk_corpus(passages, cfg,
                          embed_fn=embedder.encode if s.chunk_strategy == "semantic" else None)
    print(f"chunks: {len(chunks)}")

    vecs = embedder.encode([c.text for c in chunks])
    store = build_store(chunks, vecs)
    store.save(s.index_dir)
    meta = {**s.index_meta(), "n_passages": len(passages), "n_chunks": len(chunks),
            "embed_dim": int(vecs.shape[1])}
    (s.index_dir / "meta.json").write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"index saved -> {s.index_dir}  ({meta['n_chunks']} chunks x "
          f"{meta['embed_dim']} dims)")


if __name__ == "__main__":
    main()