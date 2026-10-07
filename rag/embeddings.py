from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from inference.device import get_device
from rag.chunker import Chunk


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk together with its embedding vector."""

    chunk: Chunk
    embedding: np.ndarray


class EmbeddingModel:
    """Generate semantic embeddings for document chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.model_name = model_name

        device = get_device()

        self.model = SentenceTransformer(
            model_name,
            device=str(device),
        )

    def encode(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:
        """Generate embeddings for a list of chunks."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding.astype(np.float32),
            )
            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            )
        ]