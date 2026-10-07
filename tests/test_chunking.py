from rag.chunker import Chunk, chunk_document
from rag.document_loader import Document


def test_document_is_split_into_chunks() -> None:
    document = Document(
        document_id="example",
        source="example.md",
        text="abcdefghijklmnopqrstuvwxyz",
    )

    chunks = chunk_document(
        document,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(chunks) == 3

    assert chunks[0] == Chunk(
        chunk_id="example-chunk-000",
        document_id="example",
        source="example.md",
        text="abcdefghij",
    )

    assert chunks[1] == Chunk(
        chunk_id="example-chunk-001",
        document_id="example",
        source="example.md",
        text="ijklmnopqr",
    )

    assert chunks[2] == Chunk(
        chunk_id="example-chunk-002",
        document_id="example",
        source="example.md",
        text="qrstuvwxyz",
    )


def test_empty_document_returns_no_chunks() -> None:
    document = Document(
        document_id="empty",
        source="empty.md",
        text="   ",
    )

    assert chunk_document(document) == []


def test_invalid_chunk_size_raises_error() -> None:
    document = Document(
        document_id="example",
        source="example.md",
        text="hello world",
    )

    try:
        chunk_document(document, chunk_size=0)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "chunk_size must be greater than zero."


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    document = Document(
        document_id="example",
        source="example.md",
        text="hello world",
    )

    try:
        chunk_document(
            document,
            chunk_size=100,
            chunk_overlap=100,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert (
            str(exc)
            == "chunk_overlap must be smaller than chunk_size."
        )