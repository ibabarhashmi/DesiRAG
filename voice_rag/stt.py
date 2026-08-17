"""Speech-to-text providers with automatic fallback.

Primary provider is **Sarvam** — the free-tier choice that best fits an Indic
corpus (₹100 free credits ≈ 3h+ of audio). Because the demo must keep working
with no key and no budget, ``ChainSTT`` (via ``make_stt_provider``) wraps
Sarvam in a fallback chain of *free* providers the user can enable:

- **vosk** (default fallback) — offline, Kaldi-based, ~40 MB Hindi model,
  no key, no network, CPU-friendly; accuracy is lower but it always answers.
- **faster-whisper** (opt-in) — offline, CTranslate2 Whisper, better accuracy
  (``VRAG_STT_FALLBACKS=vosk,faster-whisper``).

Providers are a two-method protocol, so adding another one is one class.
"""

import json
import threading
import urllib.request
import zipfile
from pathlib import Path

from .audio import to_float_16k_mono, to_pcm16k_mono
from .config import Settings
from .errors import STTError

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/{model_id}.zip"

_vosk_lock = threading.Lock()
_vosk = None                       # lazily imported vosk module
_vosk_models: dict[str, object] = {}


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
        import httpx
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


class VoskSTT(STTProvider):
    """Offline, keyless fallback: Vosk over the small Hindi model."""

    name = "vosk"

    def __init__(self, settings: Settings):
        self.model_dir = Path(settings.vosk_model_dir)
        self.model_id = settings.vosk_model_id
        self.language = settings.stt_language.split("-")[0] or "hi"

    def transcribe(self, audio: bytes, content_type: str,
                   language_code: str | None = None) -> str:
        vosk = _get_vosk()
        model = _load_model(vosk, self.model_dir, self.model_id)
        pcm = to_pcm16k_mono(audio, content_type)
        rec = vosk.KaldiRecognizer(model, 16000)
        rec.AcceptWaveform(pcm)              # one-shot feed is fine
        text = json.loads(rec.FinalResult()).get("text", "").strip()
        if not text:
            raise STTError("bad_audio", "no speech recognised")
        return text


class FasterWhisperSTT(STTProvider):
    """Opt-in offline fallback with better accuracy (CTranslate2)."""

    name = "faster-whisper"

    def __init__(self, settings: Settings):
        self.size = settings.fw_model
        self.language = settings.stt_language.split("-")[0] or "hi"
        self._model = None

    def transcribe(self, audio: bytes, content_type: str,
                   language_code: str | None = None) -> str:
        model = self._model or self._load()
        arr = to_float_16k_mono(audio, content_type)
        segments, _ = model.transcribe(
            arr, language=language_code or self.language, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            raise STTError("bad_audio", "no speech recognised")
        return text

    def _load(self):
        try:
            from faster_whisper import WhisperModel
        except Exception as e:  # noqa: BLE001
            raise STTError("unsupported",
                           f"faster-whisper not installed: {e}") from e
        self._model = WhisperModel(self.size, device="cpu", compute_type="int8")
        return self._model


class ChainSTT(STTProvider):
    """Tries providers in order; the first transcript wins.

    Exposes ``last_provider`` and ``attempts`` so the harness can surface which
    engine served the clip and what the earlier ones said."""

    name = "chain"

    def __init__(self, providers: list[STTProvider]):
        self.providers = [p for p in providers if p]
        self.last_provider: str | None = None
        self.attempts: list[tuple[str, str, str]] = []
        self.name = "+".join(p.name for p in self.providers)

    def transcribe(self, audio: bytes, content_type: str,
                   language_code: str | None = None) -> str:
        self.attempts = []
        last: STTError | None = None
        for p in self.providers:
            try:
                out = p.transcribe(audio, content_type, language_code)
                self.last_provider = p.name
                return out
            except STTError as e:
                last = e
                self.attempts.append((p.name, e.kind, e.message))
        if last:
            raise last
        raise STTError("auth", "no STT provider configured")


def make_stt_provider(settings: Settings) -> STTProvider | None:
    """Build the provider chain from settings (None disables audio)."""
    if settings.stt_provider == "none":
        return None
    chain: list[STTProvider] = []
    if settings.stt_provider == "sarvam":
        chain.append(SarvamSTT(settings))
    for fb in settings.stt_fallbacks:
        if fb == "vosk":
            chain.append(VoskSTT(settings))
        elif fb == "faster-whisper":
            chain.append(FasterWhisperSTT(settings))
    if not chain:
        return None
    return ChainSTT(chain)


# --- vosk helpers ------------------------------------------------------------

def _get_vosk():
    global _vosk
    if _vosk is None:
        try:
            import vosk
        except Exception as e:  # noqa: BLE001
            raise STTError("unsupported",
                           f"vosk package not installed: {e}") from e
        vosk.SetLogLevel(-1)
        _vosk = vosk
    return _vosk


def _load_model(vosk, model_dir: Path, model_id: str):
    path = _ensure_model(vosk, model_dir, model_id)
    key = str(path)
    model = _vosk_models.get(key)
    if model is None:
        with _vosk_lock:
            model = _vosk_models.get(key)
            if model is None:
                model = vosk.Model(key)
                _vosk_models[key] = model
    return model


def _ensure_model(vosk, model_dir: Path, model_id: str) -> Path:
    """Auto-download + extract the Vosk model once (lazy, idempotent)."""
    target = model_dir / model_id
    if target.is_dir():
        return target
    with _vosk_lock:
        if target.is_dir():
            return target
        model_dir.mkdir(parents=True, exist_ok=True)
        url = VOSK_MODEL_URL.format(model_id=model_id)
        tmp = model_dir / f"{model_id}.zip"
        print(f"[stt] downloading Vosk model {model_id} (~40 MB) ...")
        try:
            urllib.request.urlretrieve(url, tmp)
            with zipfile.ZipFile(tmp) as zf:
                zf.extractall(model_dir)
        except Exception as e:  # noqa: BLE001 — network / corrupt zip
            tmp.unlink(missing_ok=True)
            raise STTError("network", f"Vosk model download failed: {e}") from e
        finally:
            tmp.unlink(missing_ok=True)
        if not target.is_dir():
            raise STTError("network", "Vosk model download/extract failed")
    return target


class NoopSTT(STTProvider):
    """Test double: returns the recorded text verbatim."""

    name = "noop"

    def __init__(self, text: str = "no transcript"):
        self.text = text

    def transcribe(self, audio, content_type, language_code=None) -> str:
        return self.text