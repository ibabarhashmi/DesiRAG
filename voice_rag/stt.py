"""Speech-to-text providers.

Default provider is **Sarvam** — the free-tier choice that best fits an Indic
corpus (MS MARCO-XI is Hindi/regional), ₹100 free credits ≈ 3h+ of audio.
``STTProvider`` is a two-method protocol, so swapping in ElevenLabs is a
one-class change (see ``VoiceRAG._make_stt``).
"""

from dataclasses import dataclass

import httpx

from .config import Settings


@dataclass
class STTError(Exception):
    kind: str   # auth | rate_limit | network | bad_audio | unsupported
    message: str


class STTProvider:
    name = "base"

    def transcribe(self, audio: bytes, content_type: str,
                   language_code: str | None = None) -> str:
        raise NotImplementedError


class SarvamSTT(STTProvider):
    name = "sarvam"

    def __init__(self, settings: Settings):
        self.key = settings.sarvam_api_key
        self.base = settings.sarvam_base_url
        self.model = settings.sarvam_stt_model
        self.language = settings.stt_language

    def transcribe(self, audio: bytes, content_type: str,
                   language_code: str | None = None) -> str:
        if not self.key:
            raise STTError("auth", "SARVAM_API_KEY is not set")
        if not audio:
            raise STTError("bad_audio", "empty audio payload")
        url = f"{self.base.rstrip('/')}/speech-to-text"
        try:
            r = httpx.post(
                url, headers={"api-subscription-key": self.key},
                files={"file": ("audio", audio,
                                content_type or "audio/wav")},
                data={"model": self.model,
                      "language_code": language_code or self.language},
                timeout=30.0)
        except httpx.HTTPError as e:
            raise STTError("network", f"STT request failed: {e}") from e
        if r.status_code == 401:
            raise STTError("auth", "invalid Sarvam api-subscription-key")
        if r.status_code == 429:
            raise STTError("rate_limit", "Sarvam rate limit exceeded")
        if r.status_code >= 500:
            raise STTError("network", f"Sarvam 5xx: {r.status_code}")
        if r.status_code >= 400:
            raise STTError("bad_audio",
                           f"Sarvam rejected audio ({r.status_code}): "
                           f"{r.text[:200]}")
        data = r.json()
        transcript = (data.get("transcript") or "").strip()
        if not transcript:
            raise STTError("bad_audio", "no speech recognised in audio")
        return transcript


class NoopSTT(STTProvider):
    """Test double: returns the recorded text verbatim."""

    name = "noop"

    def __init__(self, text: str = "no transcript"):
        self.text = text

    def transcribe(self, audio, content_type, language_code=None) -> str:
        return self.text