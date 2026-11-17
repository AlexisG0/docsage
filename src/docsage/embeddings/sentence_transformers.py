"""sentence-transformers backend. Install with ``pip install docsage[local]``."""

from __future__ import annotations

from typing import Any


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed; run `pip install docsage[local]`"
            ) from exc
        self._model: Any = SentenceTransformer(model_name)
        self._batch_size = batch_size
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"cannot determine embedding size of {model_name!r}")
        self._dimension = int(dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in out]
