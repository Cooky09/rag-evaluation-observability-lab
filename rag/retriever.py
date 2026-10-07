from dataclasses import dataclass

import numpy as np

from rag.chunker import Chunk
from rag.embeddings import EmbeddedChunk, EmbeddingModel


@dataclass(frozen=True)
class RetrievalResult:
    """A chunk returned by semantic retrieval."""

    chunk: Chunk
    score: float


class SemanticRetriever:
    """Retrieve document chunks using cosine similarity."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        self.embedding_model = embedding_model
        self.embedded_chunks = embedded_chunks

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        """Return the most similar chunks for a query."""

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not self.embedded_chunks:
            return []

        query_embedding = self.embedding_model.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores: list[RetrievalResult] = []

        for embedded_chunk in self.embedded_chunks:
            score = float(
                np.dot(
                    query_embedding,
                    embedded_chunk.embedding,
                )
            )

            scores.append(
                RetrievalResult(
                    chunk=embedded_chunk.chunk,
                    score=score,
                )
            )

        scores.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return scores[:top_k]