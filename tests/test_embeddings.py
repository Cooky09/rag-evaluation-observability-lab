import numpy as np

from rag.chunker import Chunk
from rag.embeddings import EmbeddedChunk


def test_embedded_chunk_stores_chunk_and_vector() -> None:
    chunk = Chunk(
        chunk_id="python-chunk-000",
        document_id="python",
        source="python.md",
        text="Virtual environments isolate dependencies.",
    )

    embedding = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float32,
    )

    embedded_chunk = EmbeddedChunk(
        chunk=chunk,
        embedding=embedding,
    )

    assert embedded_chunk.chunk == chunk
    assert embedded_chunk.embedding.dtype == np.float32
    assert embedded_chunk.embedding.shape == (3,)