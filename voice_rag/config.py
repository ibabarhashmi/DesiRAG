"""Configuration: everything is overridable via environment variables.

The LLM settings deliberately reuse the same OpenAI-compatible proxy variables
the sibling `assessoraudit` project uses, so no new secrets are needed for the
optional fluent-answer mode.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


@dataclass
class Settings:
    # --- corpus / language -------------------------------------------------
    lang: str = field(default_factory=lambda: os.getenv("VRAG_LANG", "hi"))
    raw_parquet: Path = field(default_factory=lambda: Path(
        os.getenv("VRAG_RAW", ROOT / "data" / "raw" / "hinval.parquet")))
    corpus_jsonl: Path = field(default_factory=lambda: Path(
        os.getenv("VRAG_CORPUS", ROOT / "data" / "corpus" / "passages.jsonl")))
    eval_jsonl: Path = field(default_factory=lambda: Path(
        os.getenv("VRAG_EVAL", ROOT / "data" / "eval" / "queries.jsonl")))
    index_dir: Path = field(default_factory=lambda: Path(
        os.getenv("VRAG_INDEX", ROOT / "data" / "index")))
    n_corpus_examples: int = field(default_factory=lambda: int(
        os.getenv("VRAG_N_CORPUS", "40000")))
    n_eval_queries: int = field(default_factory=lambda: int(
        os.getenv("VRAG_N_EVAL", "200")))
    max_corpus_passages: int = field(default_factory=lambda: int(
        os.getenv("VRAG_MAX_PASSAGES", "80000")))

    # --- embeddings --------------------------------------------------------
    # e5-family models need "query:"/"passage:" prefixes; Embedder auto-detects
    # and applies them (overridable via VRAG_EMBED_QUERY_PREFIX / ..._PASSAGE).
    embed_model: str = field(default_factory=lambda: os.getenv(
        "VRAG_EMBED_MODEL", "intfloat/multilingual-e5-small"))
    embed_query_prefix: str = field(default_factory=lambda: os.getenv(
        "VRAG_EMBED_QUERY_PREFIX", ""))
    embed_passage_prefix: str = field(default_factory=lambda: os.getenv(
        "VRAG_EMBED_PASSAGE_PREFIX", ""))

    # --- chunking ----------------------------------------------------------
    chunk_strategy: str = field(default_factory=lambda: os.getenv(
        "VRAG_CHUNK_STRATEGY", "semantic"))
    max_chunk_tokens: int = field(default_factory=lambda: int(
        os.getenv("VRAG_MAX_CHUNK", "220")))
    short_passage_words: int = field(default_factory=lambda: int(
        os.getenv("VRAG_SHORT_PASSAGE", "80")))   # shorter => indexed whole
    overlap_words: int = field(default_factory=lambda: int(
        os.getenv("VRAG_OVERLAP", "20")))

    # --- retrieval ---------------------------------------------------------
    top_k: int = field(default_factory=lambda: int(os.getenv("VRAG_TOP_K", "6")))
    bm25_k1: float = field(default_factory=lambda: float(os.getenv("VRAG_K1", "1.5")))
    bm25_b: float = field(default_factory=lambda: float(os.getenv("VRAG_B", "0.75")))
    rrf_k: float = field(default_factory=lambda: float(os.getenv("VRAG_RRF_K", "60")))
    # "semantic" (exact-cosine top-k, the measured best on this corpus) or
    # "hybrid" (semantic + token-BM25, reciprocal-rank fused). The bigram
    # lexical layer is experimental: it maps Hindi compound spellings but was
    # measured to *drop* fused recall@5 (0.22 vs 0.33 semantic-only), so it is
    # off by default and only built when VRAG_BIGRAM=1.
    fusion_mode: str = field(default_factory=lambda: os.getenv(
        "VRAG_FUSION", "semantic"))
    use_bigram_index: bool = field(default_factory=lambda: os.getenv(
        "VRAG_BIGRAM", "0") == "1")

    # --- guardrail thresholds ---------------------------------------------
    # Calibrated on the built 80k Hindi index (see README "Guardrails"):
    # e5 cosine top-1 for on-topic queries sits at ~0.85-0.93 (min 0.846) and
    # is ~0.79-0.88 for genuinely foreign queries, so *similarity alone cannot
    # separate them* — the lexical-coverage gate does (on-topic p10=1.0,
    # off-topic max=0.5). Thresholds below are tuned to that data.
    off_topic_min_sim: float = field(default_factory=lambda: float(
        os.getenv("VRAG_OFF_TOPIC_MIN_SIM", "0.24")))
    off_topic_min_coverage: float = field(default_factory=lambda: float(
        os.getenv("VRAG_OFF_TOPIC_COVERAGE", "0.6")))
    confidence_min_sim: float = field(default_factory=lambda: float(
        os.getenv("VRAG_MIN_SIM", "0.80")))
    confidence_min_margin: float = field(default_factory=lambda: float(
        os.getenv("VRAG_MIN_MARGIN", "0.02")))
    extract_min_overlap: float = field(default_factory=lambda: float(
        os.getenv("VRAG_MIN_OVERLAP", "0.18")))
    max_query_chars: int = field(default_factory=lambda: int(
        os.getenv("VRAG_MAX_QUERY", "500")))

    # --- STT / LLM providers ----------------------------------------------
    stt_provider: str = field(default_factory=lambda: os.getenv(
        "VRAG_STT", "sarvam"))
    sarvam_api_key: str | None = field(default_factory=lambda: os.getenv(
        "SARVAM_API_KEY"))
    sarvam_base_url: str = field(default_factory=lambda: os.getenv(
        "SARVAM_BASE_URL", "https://api.sarvam.ai"))
    sarvam_stt_model: str = field(default_factory=lambda: os.getenv(
        "SARVAM_STT_MODEL", "saarika:v2.5"))
    stt_language: str = field(default_factory=lambda: os.getenv(
        "VRAG_STT_LANG", "hi-IN"))

    llm_base_url: str | None = field(default_factory=lambda: _env(
        "LLM_BASE_URL", "ANTHROPIC_BASE_URL"))
    llm_api_key: str | None = field(default_factory=lambda: _env(
        "LLM_API_KEY", "ANTHROPIC_API_KEY"))
    llm_model: str = field(default_factory=lambda: os.getenv(
        "LLM_MODEL", os.getenv("ANTHROPIC_MODEL", "")))

    # --- latency -----------------------------------------------------------
    bench_queries: int = field(default_factory=lambda: int(
        os.getenv("VRAG_BENCH_QUERIES", "100")))

    def index_meta(self) -> dict:
        return {
            "lang": self.lang,
            "embed_model": self.embed_model,
            "chunk_strategy": self.chunk_strategy,
            "max_chunk_tokens": self.max_chunk_tokens,
            "short_passage_words": self.short_passage_words,
            "overlap_words": self.overlap_words,
            "top_k": self.top_k,
            "embed_dim": None,
        }


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings