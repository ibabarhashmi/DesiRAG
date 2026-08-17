"""Audio normalization for the STT providers.

Decodes an uploaded/recorded clip into 16 kHz mono content. WAV/PCM is handled
with the stdlib `wave` module + NumPy (the browser mic ships WAV, so the common
path has zero extra deps); other containers (mp3/m4a/ogg/…) go through ffmpeg
when it is available, and produce a clear error otherwise.
"""

import io
import shutil
import subprocess
import wave

import numpy as np

from .errors import STTError

_TARGET_RATE = 16000


def _decode_wav(audio: bytes) -> tuple[int, np.ndarray]:
    w = wave.open(io.BytesIO(audio), "rb")
    try:
        nch, sw, rate, n = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        raw = w.readframes(n)
    finally:
        w.close()
    if n <= 0:
        return rate, np.zeros(0, dtype=np.float32)
    if sw == 1:                                    # 8-bit unsigned
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sw == 2:                                  # 16-bit signed
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:                                  # 32-bit signed
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise STTError("bad_audio",
                       f"unsupported wav bit depth ({sw * 8}-bit)")
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)
    return rate, arr


def _resample(arr: np.ndarray, from_rate: int) -> np.ndarray:
    if from_rate == _TARGET_RATE or arr.size < 2:
        return arr
    n_out = int(round(arr.size * _TARGET_RATE / from_rate))
    x = np.linspace(0.0, 1.0, arr.size, endpoint=False)
    xi = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(xi, x, arr)


def _decode(audio: bytes, content_type: str) -> np.ndarray:
    ctype = (content_type or "").split(";")[0].strip().lower()
    if not audio:
        raise STTError("bad_audio", "empty audio payload")
    if ctype in ("audio/wav", "audio/x-wav", "audio/wave", "audio/ogg") or not ctype:
        rate, arr = _decode_wav(audio)
        return _resample(arr, rate)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise STTError("bad_audio",
                       f"unsupported audio format '{ctype}' — feed a WAV file "
                       "or install ffmpeg for mp3/m4a/ogg")
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", "-", "-f", "s16le", "-ac", "1",
         "-ar", str(_TARGET_RATE), "-"],
        input=audio, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if proc.returncode != 0 or not proc.stdout:
        raise STTError("bad_audio", f"ffmpeg could not decode '{ctype}' audio")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def to_float_16k_mono(audio: bytes, content_type: str) -> np.ndarray:
    """Contents as float32 mono @ 16 kHz in [-1, 1]."""
    arr = _decode(audio, content_type)
    if arr.size < 400:                             # < ~12.5 ms of speech
        raise STTError("bad_audio", "no audio frames decoded")
    return np.clip(arr, -1.0, 1.0)


def to_pcm16k_mono(audio: bytes, content_type: str) -> bytes:
    """Contents as raw int16 little-endian mono @ 16 kHz (Vosk input)."""
    return (to_float_16k_mono(audio, content_type) * 32767.0).astype("<i2").tobytes()