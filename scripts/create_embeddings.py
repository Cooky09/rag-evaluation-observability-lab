from pathlib import Path

from rag.chunker import chunk_document
from rag.document_loader import load_documents
from rag.embeddings import EmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main() -> None:
    documents = load_documents(DOCUMENTS_DIR)

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")

    print()
    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    embedded_chunks = embedding_model.encode(chunks)

    print()
    print(f"Embedded chunks: {len(embedded_chunks)}")

    if embedded_chunks:
        first_embedding = embedded_chunks[0].embedding

        print(f"Vector shape:    {first_embedding.shape}")
        print(f"Vector dtype:    {first_embedding.dtype}")
        print(
            f"Vector norm:     "
            f"{float((first_embedding ** 2).sum() ** 0.5):.4f}"
        )


if __name__ == "__main__":
    main()