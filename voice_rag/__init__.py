"""Voice-enabled RAG over MS MARCO-XI.

Pipeline: voice -> speech-to-text (Sarvam) -> hybrid retrieval over a
semantic + lexical chunk index -> grounded answer generation, all wrapped in a
typed harness with guardrails and latency analytics.

Run end to end:
    python scripts/build_index.py     # data -> chunks -> embeddings -> index
    python scripts/benchmark.py       # P50/P70/P100 over ~100 val queries
    streamlit run app.py              # voice + text demo (the "live link")
"""

from .config import Settings, get_settings
from .index import ChunkStore, build_store
from .chunk import chunk_corpus, ChunkConfig
from .embed import Embedder
from .guardrails import (
    check_input,
    off_topic_result,
    grounded_result,
    confidence_gate_result,
)
from .harness import RAGPipeline, RAGResult, Citation, Status, pipeline_from_artifacts

__all__ = [
    "Settings",
    "get_settings",
    "ChunkStore",
    "build_store",
    "chunk_corpus",
    "ChunkConfig",
    "Embedder",
    "check_input",
    "off_topic_result",
    "grounded_result",
    "confidence_gate_result",
    "RAGPipeline",
    "RAGResult",
    "Citation",
    "Status",
    "pipeline_from_artifacts",
]