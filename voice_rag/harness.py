"""The harness: a typed orchestrator around the RAG core.

This is the "structured orchestration" the brief asks for, not a single
prompt-in/text-out call:

- typed stages, each timed and recorded on the result
- retries with backoff on *retryable* failures (network / rate-limit for STT,
  network for the optional LLM) but never on bad input or auth errors
- a fallback ladder for generation (LLM -> extractive -> refused), each rung
  itself guarded
- every exit path returns a structured ``RAGResult`` (never a raw crash),
  including an ERROR status with a sanitised reason when something unexpected
  escapes the guardrail gates
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from . import guardrails as gr
from .config import Settings, get_settings
from .embed import Embedder, default_embedder
from .generate import LLMGenerator, Extracted, extractive_answer
from .index import ChunkStore
from .errors import STTError
from .stt import STTProvider, make_stt_provider
from .text import tokenize


def _lexical_coverage(query: str, texts: list[str]) -> float:
    """Fraction of query content-tokens found in the union of the top hits."""
    qt = set(tokenize(query))
    if not qt:
        return 1.0
    have = set()
    for tx in texts:
        have |= set(tokenize(tx))
    return len(qt & have) / len(qt)

RETRYABLE_STT = {"network", "rate_limit"}


class Status(str, Enum):
    ANSWERED = "answered"
    BLOCKED = "blocked"
    OFF_TOPIC = "off_topic"
    LOW_CONFIDENCE = "low_confidence"
    NO_ANSWER = "no_answer"
    ERROR = "error"


@dataclass
class Citation:
    chunk_id: str
    passage_id: str
    text: str
    vector_sim: float
    bm25_score: float
    rank: int


@dataclass
class RAGResult:
    status: Status
    answer: str | None = None
    transcript: str | None = None
    query: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    intent: str = "question"
    blocked_reason: str | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)
    generator: str = ""
    language: str = "hi"
    meta: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stage_ms.values()), 2)


def _classify_intent(q: str) -> str:
    q = q.strip()
    if q.endswith("?"):
        return "question"
    if q.lower().startswith(("what", "who", "where", "when", "why", "how",
                             "क्या", "कौन", "कहाँ", "कब", "क्यों", "कैसे")):
        return "question"
    if q.lower().startswith(("tell", "explain", "define", "बताओ", "समझाओ",
                             "बतायें", "बताइए")):
        return "imperative"
    return "statement"


class RAGPipeline:
    def __init__(self, settings: Settings, store: ChunkStore,
                 embedder: Embedder, stt: STTProvider | None = None,
                 llm: LLMGenerator | None = None,
                 stt_attempts: int = 2, llm_attempts: int = 2):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.stt = stt
        self.llm = llm
        self.stt_attempts = stt_attempts
        self.llm_attempts = llm_attempts

    # --- public entry points ------------------------------------------------

    def run_from_audio(self, audio: bytes, content_type: str) -> RAGResult:
        t0 = time.perf_counter()
        try:
            transcript = self._transcribe_with_retry(audio, content_type)
        except STTError as e:
            meta = {"stt_provider": getattr(self.stt, "last_provider", None)}
            if getattr(self.stt, "attempts", None):
                meta["stt_errors"] = list(self.stt.attempts)
            return RAGResult(Status.ERROR, blocked_reason=e.message,
                             language=self.settings.lang,
                             stage_ms={"stt_ms": round(
                                 (time.perf_counter() - t0) * 1000, 2)},
                             meta=meta)
        meta = {"stt_provider": getattr(self.stt, "last_provider", None)}
        if getattr(self.stt, "attempts", None):
            meta["stt_errors"] = list(self.stt.attempts)
        res = self.run_from_text(transcript)
        res.meta = {**res.meta, **meta}
        return self._finish(res,
                            {"stt_ms": (time.perf_counter() - t0) * 1000},
                            transcript)

    def run_from_text(self, query_text: str) -> RAGResult:
        """Full RAG loop over text. Every path returns a RAGResult."""
        self.stage_ms = {}
        try:
            q = (query_text or "").strip()
            res = self._guarded(q)
        except Exception as e:  # noqa: BLE001 — safety net, never crash
            res = RAGResult(Status.ERROR, query=q, language=self.settings.lang,
                            blocked_reason=f"internal error: {type(e).__name__}")
        res.stage_ms = dict(self.stage_ms)
        res.language = self.settings.lang
        return res

    # --- internals ----------------------------------------------------------

    def _guarded(self, q: str) -> RAGResult:
        t = self._t

        g = gr.check_input(q, self.settings.max_query_chars)
        if not g.ok:
            return RAGResult(Status.BLOCKED, query=q, blocked_reason=g.reason)
        g = gr.safety_block(q)
        if g:
            return RAGResult(Status.BLOCKED, query=q, blocked_reason=g.reason,
                             meta=g.meta)

        qvec = t("encode_query", lambda: self.embedder.encode_query(q))

        hits = t("retrieve", lambda: self.store.search(
            qvec, q, k=self.settings.top_k, rrf_k=self.settings.rrf_k,
            mode=self.settings.fusion_mode))
        if not hits:
            return RAGResult(Status.LOW_CONFIDENCE, query=q,
                             blocked_reason="no evidence retrieved")
        top_sim = hits[0].vector_sim
        coverage = _lexical_coverage(q, [h.chunk.text for h in hits])
        g = gr.off_topic_result(top_sim, coverage,
                                self.settings.off_topic_min_sim,
                                self.settings.off_topic_min_coverage)
        if g:
            return RAGResult(Status.OFF_TOPIC, query=q,
                             blocked_reason=g.reason, meta=g.meta)

        g = gr.confidence_gate_result(hits,
                                      self.settings.confidence_min_sim,
                                      self.settings.confidence_min_margin)
        if g:
            return RAGResult(Status.LOW_CONFIDENCE, query=q,
                             blocked_reason=g.reason, meta=g.meta)

        extracted = t("generate_extractive", lambda: extractive_answer(
            q, hits, min_overlap=self.settings.extract_min_overlap))
        if self.llm and self.llm.available:
            llm_out = self._llm_with_retry(q, hits)
            if llm_out:
                text, cit = llm_out
                g = gr.grounded_result(text, [h.chunk.text for h in hits])
                if g.ok:
                    return self._answered(q, hits, text, cit,
                                          generator="llm",
                                          confidence=max(
                                              hits[0].vector_sim,
                                              extracted.confidence if extracted else 0))
                # ungrounded LLM output -> fall through to extractive
        g = gr.extractive_ok(extracted, self.settings.extract_min_overlap)
        if g:
            return RAGResult(Status.LOW_CONFIDENCE, query=q,
                             blocked_reason=g.reason, meta=g.meta)
        return self._answered(q, hits, extracted.text,
                              self._cite(hits, extracted),
                              generator="extractive",
                              confidence=extracted.confidence)

    def _answered(self, q, hits, text, citations, generator, confidence):
        return RAGResult(
            Status.ANSWERED, answer=text, query=q,
            citations=[Citation(c["chunk_id"], c["passage_id"], c["text"],
                                c["vector_sim"], c["bm25_score"], c["rank"])
                       for c in citations],
            confidence=round(float(confidence), 4),
            intent=_classify_intent(q), generator=generator,
            meta={"top_sim": round(hits[0].vector_sim, 4),
                  "n_citations": len(citations)})

    def _cite(self, hits, extracted: Extracted) -> list[dict]:
        seen = set()
        out = []
        for h in hits:
            if h.chunk.id in seen:
                continue
            seen.add(h.chunk.id)
            out.append({"chunk_id": h.chunk.id, "passage_id": h.chunk.passage_id,
                        "text": h.chunk.text[:400],
                        "vector_sim": round(h.vector_sim, 4),
                        "bm25_score": round(h.bm25_score, 4),
                        "rank": len(out)})
        return out

    def _llm_with_retry(self, q, hits):
        for attempt in range(self.llm_attempts):
            try:
                return self._t("generate_llm", lambda: self.llm.generate(q, hits))
            except Exception:
                if attempt + 1 >= self.llm_attempts:
                    return None
                time.sleep(0.4 * (attempt + 1))
        return None

    def _transcribe_with_retry(self, audio, content_type):
        if not self.stt:
            raise STTError("auth", "no STT provider configured")
        for attempt in range(self.stt_attempts):
            try:
                return self.stt.transcribe(audio, content_type)
            except STTError as e:
                if e.kind in RETRYABLE_STT and attempt + 1 < self.stt_attempts:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise
        raise STTError("network", "STT retries exhausted")

    def _finish(self, res: RAGResult, extra_stages: dict, transcript: str) -> RAGResult:
        res.transcript = transcript
        res.stage_ms = {**extra_stages, **res.stage_ms}
        return res

    def _t(self, name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        self.stage_ms[name] = round((time.perf_counter() - t0) * 1000, 2)
        return out


def _make_stt(settings: Settings) -> STTProvider | None:
    return make_stt_provider(settings)


def pipeline_from_artifacts(index_dir: Path | str,
                            settings: Settings | None = None,
                            use_llm: bool | None = None) -> RAGPipeline:
    """Build a ready pipeline from a built index directory.

    ``use_llm`` defaults to the VRAG_LLM env flag; when unset, the fluent
    LLM rung stays off so the default path is the fast, grounded extractive
    one (see README latency budget).
    """
    s = settings or get_settings()
    store = ChunkStore.load(Path(index_dir))
    embed = default_embedder(s)
    llm = None
    if use_llm is None:
        use_llm = os.getenv("VRAG_LLM", "0") == "1"
    if use_llm:
        llm = LLMGenerator(s)
    return RAGPipeline(s, store, embed, stt=_make_stt(s), llm=llm)


def embed_matrix(embedder: Embedder, texts: list[str]) -> np.ndarray:
    return embedder.encode(texts)