"""Multilingual embedding for retrieval + semantic chunking.

Uses sentence-transformers with Apple's Metal (MPS) when available. The same
encoder produces corpus vectors, query vectors, and (in semantic chunking)
sentence gaps, so there is exactly one embedding distribution in play.

e5-family models (the default) require instruction prefixes — ``query:`` for
queries, ``passage:`` for documents — which this wrapper applies
automatically.
"""

from functools import lru_cache

import numpy as np

from .config import Settings, get_settings

_DEFAULT_QUERY = "query: "
_DEFAULT_PASSAGE = "passage: "


class Embedder:
    def __init__(self, model_name: str, device: str | None = None,
                 batch_size: int = 256, query_prefix: str = "",
                 passage_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=(
            device or ("mps" if self._mps() else "cpu")))
        get_dim = getattr(self.model, "get_embedding_dimension",
                          self.model.get_sentence_embedding_dimension)
        self.dim = get_dim()
        self.batch_size = batch_size
        e5 = "e5" in model_name.lower()
        self.query_prefix = (query_prefix or (_DEFAULT_QUERY if e5 else "")).rstrip()
        self.passage_prefix = (passage_prefix or (_DEFAULT_PASSAGE if e5 else "")).rstrip()

    @staticmethod
    def _mps() -> bool:
        try:
            import torch
            return bool(torch.backends.mps.is_available())
        except Exception:
            return False

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [f"{prefix} {t}" if prefix else t for t in texts]
        out = self.model.encode(
            prefixed, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Document/sentence embeddings (passage-prefixed)."""
        return self._encode(list(texts), self.passage_prefix)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], self.query_prefix)[0]


@lru_cache(maxsize=4)
def _embedder_for(model_name: str, query_prefix: str,
                  passage_prefix: str) -> Embedder:
    return Embedder(model_name, query_prefix=query_prefix,
                    passage_prefix=passage_prefix)


def default_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    return _embedder_for(s.embed_model, s.embed_query_prefix,
                         s.embed_passage_prefix)