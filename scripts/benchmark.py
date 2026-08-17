"""Latency analytics: P50 / P70 / P100 for the full text->answer RAG loop.

Measures the pipeline *after* speech-to-text (STT is an external network call;
its own latency is reported separately and excluded from the budget, matching
how the 200ms target is scoped in the README). Warmup runs are discarded so the
percentiles reflect steady-state, not model-load cold start.

Run:
    python scripts/benchmark.py [--queries 100] [--report data/benchmark.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice_rag.config import get_settings
from voice_rag.data import read_eval
from voice_rag.harness import pipeline_from_artifacts


def pctile(name, values):
    p = {f"p{p}": round(float(np.percentile(values, p)), 2)
         for p in (50, 70, 100)}
    print(f"  {name:<22} n={len(values):<4} "
          + " ".join(f"{k}={v}ms" for k, v in p.items()))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--report", type=str, default="data/benchmark.json")
    args = ap.parse_args()

    s = get_settings(refresh=True)
    evals = read_eval(s.eval_jsonl)[: args.queries]
    pipe = pipeline_from_artifacts(s.index_dir, s)
    if not evals:
        print("no eval queries — run scripts/build_index.py first")
        return

    print(f"warming up ({len(evals) and 'first query'})...")
    pipe.run_from_text(evals[0].query)   # model load + caches

    totals, extracts, retrieves, encodes = [], [], [], []
    answered = refused = 0
    for e in evals:
        t0 = time.perf_counter()
        res = pipe.run_from_text(e.query)
        totals.append((time.perf_counter() - t0) * 1000)
        stage = res.stage_ms
        extracts.append(stage.get("generate_extractive", 0.0))
        retrieves.append(stage.get("retrieve", 0.0))
        encodes.append(stage.get("encode_query", 0.0))
        if res.status.value == "answered":
            answered += 1
        else:
            refused += 1

    out = {}
    print(f"\nqueries measured: {len(evals)} (warm) | answered={answered} "
          f"refused={refused}")
    print("steady-state text->answer RAG loop (excl. STT):")
    out["total_ms"] = pctile("total", totals)
    print("\nstage breakdown:")
    out["encode_query_ms"] = pctile("encode", encodes)
    out["retrieve_ms"] = pctile("retrieve", retrieves)
    out["generate_extractive_ms"] = pctile("extract", extracts)
    out["n_queries"] = len(evals)
    out["n_answered"] = answered
    out["n_refused"] = refused
    out["scope"] = "text->answer; excludes external STT network round-trip"

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()