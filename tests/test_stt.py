"""STT chain + audio normalization tests (no network, no model downloads)."""

import io
import struct
import wave

import numpy as np
import pytest

from voice_rag.audio import to_pcm16k_mono, to_float_16k_mono
from voice_rag.config import Settings
from voice_rag.errors import STTError
from voice_rag.harness import RAGPipeline, Status
from voice_rag.stt import (STTProvider, ChainSTT, FasterWhisperSTT,
                           SarvamSTT, VoskSTT, make_stt_provider)
from tests.conftest import FakeStore, FakeEmbedder, fake_hits


def _wav(sample_rate=44100, channels=2, seconds=0.3, tone=440.0):
    """In-memory int16 PCM WAV (a nonzero auditible tone)."""
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = 0.4 * np.sin(2 * np.pi * tone * t)
    data = np.repeat(mono[:, None], channels, axis=1) \
        if channels > 1 else mono.reshape(-1, 1)
    pcm = (data * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class OkSTT(STTProvider):
    name = "ok"

    def transcribe(self, audio, content_type, language_code=None):
        return "where is the taj mahal"


class FailingSTT(STTProvider):
    name = "failing"

    def __init__(self, kinds=("auth",), msgs=("boom",)):
        self._kinds, self._msgs = list(kinds), list(msgs)
        self.calls = 0

    def transcribe(self, audio, content_type, language_code=None):
        i = min(self.calls, len(self._kinds) - 1)
        self.calls += 1
        raise STTError(self._kinds[i], self._msgs[i])


def test_pcm16k_mono_output_shape_and_bounds():
    pcm = to_pcm16k_mono(_wav(44100, 2, 0.3), "audio/wav")
    assert len(pcm) == 16000 * 0.3 * 2          # 16 kHz, mono, int16
    arr = np.frombuffer(pcm, dtype="<i2")
    assert abs(arr).max() <= 32767


def test_pcm16k_resamples_from_8k():
    pcm = to_pcm16k_mono(_wav(8000, 1, 0.2), "audio/wav")
    assert len(pcm) == 16000 * 0.2 * 2


def test_unsupported_format_without_ffmpeg(monkeypatch):
    monkeypatch.setattr("voice_rag.audio.shutil.which", lambda _: None)
    with pytest.raises(STTError) as ei:
        to_float_16k_mono(b"\x00" * 100, "audio/mpeg")
    assert ei.value.kind == "bad_audio"


def test_empty_audio_rejected():
    with pytest.raises(STTError) as ei:
        to_pcm16k_mono(b"", "audio/wav")
    assert ei.value.kind == "bad_audio"


def test_make_chain_ordering():
    s = Settings()
    chain = make_stt_provider(s)
    assert isinstance(chain, ChainSTT)
    assert [p.name for p in chain.providers] == ["sarvam", "faster-whisper"]
    s2 = Settings()
    s2.stt_provider = "none"
    assert make_stt_provider(s2) is None


def test_chain_falls_back_on_primary_error():
    chain = ChainSTT([FailingSTT(("auth",), ("no key",)), OkSTT()])
    out = chain.transcribe(b"audio", "audio/wav")
    assert out == "where is the taj mahal"
    assert chain.last_provider == "ok"
    assert chain.attempts == [("failing", "auth", "no key")]


def test_chain_uses_primary_when_it_works():
    chain = ChainSTT([OkSTT(), FailingSTT()])
    assert chain.transcribe(b"a", "audio/wav") == "where is the taj mahal"
    assert chain.last_provider == "ok"
    assert chain.attempts == []


def test_chain_all_failed_raises_last_error():
    chain = ChainSTT([FailingSTT(("network",), ("net down",)),
                      FailingSTT(("bad_audio",), ("no speech",))])
    with pytest.raises(STTError) as ei:
        chain.transcribe(b"a", "audio/wav")
    assert ei.value.kind == "bad_audio"          # closest error surfaces
    assert len(chain.attempts) == 2


def test_sarvam_needs_key():
    s = Settings()
    s.sarvam_api_key = None
    with pytest.raises(STTError) as ei:
        SarvamSTT(s).transcribe(b"a", "audio/wav")
    assert ei.value.kind == "auth"


def test_vosk_unavailable_degrades_gracefully(monkeypatch):
    def boom():
        raise STTError("unsupported", "vosk package not installed")

    monkeypatch.setattr("voice_rag.stt._get_vosk", boom)
    with pytest.raises(STTError) as ei:
        VoskSTT(Settings()).transcribe(_wav(), "audio/wav")
    assert ei.value.kind == "unsupported"


def test_faster_whisper_load_failure_is_graceful(monkeypatch):
    def raise_unsupported(self):
        raise STTError("unsupported", "faster-whisper not installed")

    monkeypatch.setattr(FasterWhisperSTT, "_load", raise_unsupported)
    with pytest.raises(STTError) as ei:
        FasterWhisperSTT(Settings()).transcribe(_wav(), "audio/wav")
    assert ei.value.kind == "unsupported"


def test_harness_audio_reports_provider_and_errors(fake_embedder, settings):
    hits = fake_hits(["The Taj Mahal is in Agra, India."], sims=[0.9])
    stt = ChainSTT([FailingSTT(("auth",), ("no key",)), OkSTT()])
    pipe = RAGPipeline(settings, FakeStore(hits), fake_embedder, stt=stt)
    res = pipe.run_from_audio(_wav(), "audio/wav")
    assert res.status is Status.ANSWERED
    assert res.transcript == "where is the taj mahal"
    assert res.meta["stt_provider"] == "ok"
    assert res.meta["stt_errors"] == [("failing", "auth", "no key")]
    assert "stt_ms" in res.stage_ms