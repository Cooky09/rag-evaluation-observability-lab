from pathlib import Path

from rag.chunker import chunk_document
from rag.document_loader import load_documents
from rag.embeddings import EmbeddingModel
from rag.retriever import SemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main() -> None:
    documents = load_documents(DOCUMENTS_DIR)

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created:   {len(chunks)}")

    embedding_model = EmbeddingModel()

    embedded_chunks = embedding_model.encode(chunks)

    retriever = SemanticRetriever(
        embedding_model=embedding_model,
        embedded_chunks=embedded_chunks,
    )

    queries = [
        "How can I isolate Python dependencies?",
        "How do I start multiple Docker services?",
        "How can an API validate incoming data?",
        "How do I use an NVIDIA GPU with PyTorch?",
        "How can I improve GPU inference throughput?",
    ]

    for query in queries:
        print()
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.search(
            query,
            top_k=3,
        )

        for rank, result in enumerate(results, start=1):
            print()
            print(f"Rank:   {rank}")
            print(f"Score:  {result.score:.4f}")
            print(f"Source: {result.chunk.source}")
            print(f"ID:     {result.chunk.chunk_id}")
            print("-" * 70)
            print(result.chunk.text[:300])


if __name__ == "__main__":
    main()