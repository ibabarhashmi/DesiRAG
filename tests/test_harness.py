"""Harness tests: the full guarded loop, retries, fallback ladder, error
recovery, and structured results."""

import numpy as np
import pytest

from voice_rag.harness import RAGPipeline, Status
from voice_rag.stt import NoopSTT, STTError
from tests.conftest import fake_hits


class RetryingSTT:
    """Fails twice with a retryable error, then succeeds."""

    name = "retry"
    calls = 0

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, content_type, language_code=None):
        self.calls += 1
        if self.calls < 3:
            raise STTError("network", "flaky network")
        return "where is the taj mahal"


class AuthSTT:
    name = "auth"

    def transcribe(self, audio, content_type, language_code=None):
        raise STTError("auth", "SARVAM_API_KEY is not set")


def test_answered_path(pipeline):
    res = pipeline.run_from_text("where is the taj mahal")
    assert res.status is Status.ANSWERED
    assert res.answer and "Taj Mahal" in res.answer
    assert res.citations and res.citations[0].passage_id == "p0"
    assert res.generator == "extractive"
    assert res.confidence > 0
    assert set(res.stage_ms) >= {"encode_query", "retrieve",
                                 "generate_extractive"}
    assert res.intent == "question"
    assert res.total_ms > 0


def test_blocked_then_no_llm_touch(pipeline):
    res = pipeline.run_from_text("fuck you and your mother")
    assert res.status is Status.BLOCKED
    res2 = pipeline.run_from_text("")
    assert res2.status is Status.BLOCKED


def test_off_topic_via_coverage(fake_embedder, settings):
    hits = fake_hits(["zzz qqq unrelated text here"], sims=[0.90],
                     coverages=[0.0])
    pipe = RAGPipeline(settings, type("S", (), {"search": lambda *a, **k: hits})(),
                       fake_embedder)
    res = pipe.run_from_text("taj mahal kahan hai")
    assert res.status is Status.OFF_TOPIC


def test_low_confidence(fake_embedder, settings):
    # covers the query lexically (passes the off-topic gate) but scores below
    # the confidence threshold -> LOW_CONFIDENCE
    hits = fake_hits(["taj mahal unrelated passage text"], sims=[0.45],
                     coverages=[1.0])
    pipe = RAGPipeline(settings, type("S", (), {"search": lambda *a, **k: hits})(),
                       fake_embedder)
    res = pipe.run_from_text("where is the taj mahal")
    assert res.status is Status.LOW_CONFIDENCE


def test_audio_path_transcribes_and_answers(settings, fake_embedder):
    hits = fake_hits(
        ["The Taj Mahal is a white marble mausoleum in Agra, India.",
         "New Delhi is the capital of India."], sims=[0.90, 0.80])
    pipe = RAGPipeline(settings,
                       type("S", (), {"search": lambda *a, **k: hits})(),
                       fake_embedder, stt=NoopSTT("where is the taj mahal"))
    res = pipe.run_from_audio(b"RIFF....", "audio/wav")
    assert res.status is Status.ANSWERED
    assert res.transcript == "where is the taj mahal"
    assert "stt_ms" in res.stage_ms


def test_stt_retry_then_success(settings, fake_embedder):
    hits = fake_hits(
        ["The Taj Mahal is a white marble mausoleum in Agra, India."],
        sims=[0.90])
    stt = RetryingSTT()
    pipe = RAGPipeline(settings,
                       type("S", (), {"search": lambda *a, **k: hits})(),
                       fake_embedder, stt=stt, stt_attempts=3)
    res = pipe.run_from_audio(b"audio", "audio/wav")
    assert res.status is Status.ANSWERED
    assert stt.calls == 3  # two failures absorbed by retry


def test_stt_auth_error_is_not_retried(settings, fake_embedder):
    pipe = RAGPipeline(settings,
                       type("S", (), {"search": lambda *a, **k: []})(),
                       fake_embedder, stt=AuthSTT(), stt_attempts=3)
    res = pipe.run_from_audio(b"audio", "audio/wav")
    assert res.status is Status.ERROR
    assert "SARVAM_API_KEY" in res.blocked_reason


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.available = True

    def generate(self, query, hits):
        return self.text, [{"chunk_id": "c0", "passage_id": "p0",
                            "text": hits[0].chunk.text, "vector_sim": 0.9,
                            "bm25_score": 0.0, "rank": 0}]


def test_llm_grounded_answer_used(settings, fake_embedder):
    hits = fake_hits(
        ["The Taj Mahal is a white marble mausoleum in Agra, India."],
        sims=[0.90])
    pipe = RAGPipeline(settings,
                       type("S", (), {"search": lambda *a, **k: hits})(),
                       fake_embedder, llm=FakeLLM(
                           "The Taj Mahal is in Agra, India."))
    res = pipe.run_from_text("where is the taj mahal")
    assert res.status is Status.ANSWERED
    assert res.generator == "llm"


def test_llm_ungrounded_falls_back_to_extractive(settings, fake_embedder):
    hits = fake_hits(
        ["The Taj Mahal is a white marble mausoleum in Agra, India."],
        sims=[0.90])
    pipe = RAGPipeline(settings,
                       type("S", (), {"search": lambda *a, **k: hits})(),
                       fake_embedder, llm=FakeLLM(
                           "The moon is made entirely of green cheese."))
    res = pipe.run_from_text("where is the taj mahal")
    assert res.status is Status.ANSWERED
    assert res.generator == "extractive"


def test_internal_error_becomes_error_result(fake_embedder, settings):
    class BoomStore:
        def search(self, *a, **k):
            raise RuntimeError("index corrupted")

    pipe = RAGPipeline(settings, BoomStore(), fake_embedder)
    res = pipe.run_from_text("where is the taj mahal")
    assert res.status is Status.ERROR
    assert "RuntimeError" in res.blocked_reason


def test_llm_failure_falls_back(settings, fake_embedder):
    hits = fake_hits(
        ["The Taj Mahal is a white marble mausoleum in Agra, India."],
        sims=[0.90])

    class BrokenLLM:
        available = True

        def generate(self, q, hits):
            raise RuntimeError("llm down")

    pipe = RAGPipeline(settings,
                       type("S", (), {"search": lambda *a, **k: hits})(),
                       fake_embedder, llm=BrokenLLM(), llm_attempts=2)
    res = pipe.run_from_text("where is the taj mahal")
    assert res.status is Status.ANSWERED
    assert res.generator == "extractive"


def test_intent_classification():
    from voice_rag.harness import _classify_intent
    assert _classify_intent("क्या है ताज महल?") == "question"
    assert _classify_intent("what is a monsoon") == "question"
    assert _classify_intent("tell me about the ganges") == "imperative"
    assert _classify_intent("the taj mahal is a monument") == "statement"