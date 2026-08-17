# Voice RAG — Hindi, sub-200ms, hauled from the ground up

A voice-enabled Retrieval-Augmented Generation system over the
**ai4bharat/MSMARCO-XI** Hindi corpus.

- **Inputs**: typed text **or** recorded/uploaded audio (Sarvam AI STT).
- **Retrieval**: dense + lexical hybrid over an 80,000-passage index built
  entirely in-repo (no vector-DB server needed at demo scale — a single NumPy
  matmul does exact-cosine top-k in ~1.4 ms).
- **Generation**: grounded extractive answers by default; optional fluent-LLM
  rung that is *refused unless every sentence is contained in the retrieved
  context*.
- **Guardrails**: five independent gates that make the system honest about
  when it does **not** know — 25 % of the 100 eval queries are refused
  (off-topic) rather than hallucinated.
- **Latency**: end-to-end text→answer **p50 = 11.5 ms, p100 = 23.9 ms**
  (well under the 200 ms budget), measured over 100 warm queries.

---

## Quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# build the 80k-passage index (downloads validation/hinval.parquet, ~460 MB)
python scripts/build_index.py --n-corpus 80000 --n-eval 200

# run the 39 tests
pytest -q

# latency percentiles, guardrail stats
python scripts/benchmark.py
python scripts/evaluate.py

# launch the Streamlit demo (mic + audio-upload + text tabs)
streamlit run app.py
```

Optional audio pipeline: set `SARVAM_API_KEY` in `.env` (it is read at runtime;
the STT path retries transient network errors but refuses on auth errors).

Optional fluent answers: the LLM rung reuses an OpenAI-compatible proxy via the
same env vars as any sibling project (`LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`,
falling back to `ANTHROPIC_*`).

---

## Architecture

```
                 ┌──────────────┐  typed query
  mic ──▶ Sarvam ──▶ transcript ─▶│ check_input  │
  upload ───────▶ WS? no          └──────┬───────┘
                                        guardrails (never crash, always
                                        return a typed RAGResult)
        ┌───────────────────────────────▼──────────────────────────┐
        │ 1. safety       2. encode query (e5-small, "query:" pref.) │
        │ 3. retrieve     4. off-topic   5. confidence gate          │
        │ 6. extractive   7. (optional) grounded LLM                 │
        └───────────────────────────────────────────────────────────┘
```

All retrieval + generation stages are timed independently and returned on
`RAGResult.stage_ms`; `total_ms` is their sum. Every path — including crashes —
returns a structured `RAGResult` (`Status.ANSWERED | OFF_TOPIC |
LOW_CONFIDENCE | BLOCKED | UNGROUNDED | ERROR`).

### Retrieval backends (all self-contained, pure NumPy)

| Module | What it is | When it wins |
|---|---|---|
| `VectorBackend` | exact cosine top-k over an (N×384) float32 matrix | **the default** — measured best |
| `_BM25` (token) | vocab → inverted lists, BM25 scoring, no scipy | rare/exact lexical terms |
| `_BM25` (bigram) | same, over padded char bigrams | Hindi compound spellings (`ताज महल` / `ताजमहल`) |
| RRF fusion | reciprocal-rank fusion of whatever lists are enabled | blended ranking |

Fusion mode and the experimental bigram layer are config flags, because they
were **measured**, not assumed (see below). The swap to an ANN index (HNSW) if
the corpus ever outgrows a few hundred thousand chunks is one method inside
`VectorBackend`.

### Chunking (multi-strategy, router-augmented)

`chunk.py` implements three strategies — fixed-size, recursive (sentential,
with overlap carry-over), and semantic (sentence-embedding discontinuity) —
plus a length-aware router that keeps short passages whole. The build default
is semantic; the honest evaluation follows.

---

## The "real thought" — measurements, not assumptions

### 1. Chunking strategy does not matter at this corpus size

30 held-out eval queries over a comparably-built 2,018-passage mini corpus:

| strategy | chunks | recall@5 | answer-F1 |
|---|---|---|---|
| fixed | 2033 | 0.367 | 0.111 |
| recursive | 2085 | 0.367 | 0.111 |
| semantic | 2018 | 0.367 | 0.111 |

Every passage in this corpus is short enough that the router keeps it whole, so
no strategy splits anything — they tie. The multi-strategy chunker exists for
**longer documents** (the task's "more than one strategy", exercised and
documented), not because it helps these 80k terse passages.

### 2. Lexical fusion was tested and *dropped* because it hurt recall

measured on the full 80k index, 100 eval queries:

| retrieval mode | recall@5 | recall@10 |
|---|---|---|
| semantic only | **0.330** | **0.390** |
| + token-BM25 (RRF) | 0.250 | — |
| + bigram-BM25 (RRF) | 0.220 | — |

The dense matcher already captures Hindi semantics; the lexical lists add
mostly noise here (the bigram vocabulary is only 5,809 types for the whole
corpus, so common bigrams drag unrelated passages up). The token and bigram
BM25 indexes remain available (`VRAG_FUSION=hybrid`, `VRAG_BIGRAM=1`) as the
compound-spelling rescue path, and the bigram index is unit-tested to link
`ताज महल`↔`ताजमहल`. But the shipped default is semantic-only, because that is
what the data supports.

> Honest limits: recall@5 ≈ 0.33 reflects noisy machine-translated eval
> queries and an 80k-passage slice; single missing/ambiguity cases are
> *refused*, not guessed (e.g. `ताज महल क्या है` is answered only if the
> monument passage actually surfaces; otherwise `off_topic`).

### 3. Guardrails were calibrated against the actual failure modes

On 100 eval queries the system answered 73 and refused 27 (25 `off_topic` +
2 `low_confidence`) — the refusals are the point: it knows when it doesn't
know.

Two calibration findings that shaped the gates:

- **Similarity alone cannot separate on- from off-topic.** On-topic top-1
  cosine sits at p10=0.873 / p50=0.902, but genuinely foreign queries still
  max 0.876 — an 80k-passage corpus always has *something* nearby.
- **Lexical coverage of the top-5 does separate them**: on-topic p10=1.0 vs
  off-topic max=0.5. Hence `off_topic = top_sim≥0.24 **and** coverage≥0.6`.

The five gates in order: `check_input` → `safety` (English **and** Hindi
abuse / PII-harvesting / prompt-injection) → `off_topic` → `confidence` →
`grounded` (for LLM answers). `safety_block` uses curated phrase lists,
dependency-free by design — a real deployment swaps in a dedicated classifier.

### 4. Every failure mode degrades gracefully, never crashes

- STT: 2 attempts, backoff for `network`/`rate_limit`, immediate refusal for
  `auth`.
- Generation: LLM → extractive → `LOW_CONFIDENCE`. An LLM answer that fails
  the grounding check falls through to extractive instead of hallucinating.
- Everything is wrapped: `Status.ERROR` returns with a sanitised reason.

---

## Why no separate vector database?

At 80k chunks × 384 float32 dims, an exact cosine search is one `(N×D) @ D`
matmul on CPU: p50 = **1.41 ms**. A FAISS/Qdrant server would add
infrastructure, ops surface, and network latency for negative value at this
scale. `VectorBackend` isolates the swap point (HNSW) for when the corpus
grows — the "reuse what's installed, choose correctness first" rule applied
deliberately.

## Repository layout

```
voice_rag/
  voice_rag/            core: config data chunk embed text index generate
                         guardrails stt harness app(Streamlit)
  scripts/              build_index, evaluate, benchmark
  tests/                39 tests across chunk/index/guardrails/harness
  requirements.txt      single file, dependency-free guardrails
```

## Streamlit cloud deploy

The demo runs on a free Streamlit Community Cloud app linked to this repo.
`data/` is git-ignored; the persisted index (`vectors.npy` + token-BM25
`lex.npz` + `chunks.jsonl`) is ~150 MB and must be shipped via the cloud app's
secrets/files or regenerated on launch with `scripts/build_index.py`.