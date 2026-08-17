"""Shared structured errors (no cross-module import cycles)."""

from dataclasses import dataclass


@dataclass
class STTError(Exception):
    kind: str   # auth | rate_limit | network | bad_audio | unsupported
    message: str