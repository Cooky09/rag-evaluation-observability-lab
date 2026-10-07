import numpy as np

from rag.chunker import Chunk
from rag.embeddings import EmbeddedChunk
from rag.retriever import SemanticRetriever


class FakeEmbeddingModel:
    """Small deterministic embedding model for unit tests."""

    class FakeModel:
        def encode(
            self,
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ):
            if "python" in text.lower():
                return np.array(
                    [1.0, 0.0, 0.0],
                    dtype=np.float32,
                )

            return np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            )

    def __init__(self) -> None:
        self.model = self.FakeModel()


def test_retriever_returns_most_similar_chunk() -> None:
    python_chunk = Chunk(
        chunk_id="python-001",
        document_id="python",
        source="python.md",
        text="Python virtual environments isolate dependencies.",
    )

    docker_chunk = Chunk(
        chunk_id="docker-001",
        document_id="docker",
        source="docker.md",
        text="Docker containers package applications.",
    )

    embedded_chunks = [
        EmbeddedChunk(
            chunk=python_chunk,
            embedding=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
        EmbeddedChunk(
            chunk=docker_chunk,
            embedding=np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
        ),
    ]

    retriever = SemanticRetriever(
        embedding_model=FakeEmbeddingModel(),
        embedded_chunks=embedded_chunks,
    )

    results = retriever.search(
        "Python dependencies",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == "python"
    assert results[0].score == 1.0


def test_results_are_sorted_by_similarity() -> None:
    first_chunk = Chunk(
        chunk_id="first",
        document_id="first",
        source="first.md",
        text="First document",
    )

    second_chunk = Chunk(
        chunk_id="second",
        document_id="second",
        source="second.md",
        text="Second document",
    )

    embedded_chunks = [
        EmbeddedChunk(
            chunk=first_chunk,
            embedding=np.array(
                [0.5, 0.8660254, 0.0],
                dtype=np.float32,
            ),
        ),
        EmbeddedChunk(
            chunk=second_chunk,
            embedding=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
    ]

    retriever = SemanticRetriever(
        embedding_model=FakeEmbeddingModel(),
        embedded_chunks=embedded_chunks,
    )

    results = retriever.search(
        "Python",
        top_k=2,
    )

    assert results[0].chunk.document_id == "second"
    assert results[0].score > results[1].score


def test_empty_query_raises_error() -> None:
    retriever = SemanticRetriever(
        embedding_model=FakeEmbeddingModel(),
        embedded_chunks=[],
    )

    try:
        retriever.search("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "query cannot be empty."


def test_invalid_top_k_raises_error() -> None:
    retriever = SemanticRetriever(
        embedding_model=FakeEmbeddingModel(),
        embedded_chunks=[],
    )

    try:
        retriever.search("Python", top_k=0)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "top_k must be greater than zero."