import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunker import chunk_document
from rag.document_loader import load_documents
from rag.embeddings import EmbeddingModel
from rag.generator import LocalLLMGenerator
from rag.pipeline import RAGPipeline
from rag.retriever import SemanticRetriever


DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def build_retriever() -> SemanticRetriever:
    documents = load_documents(DOCUMENTS_DIR)

    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=800,
                chunk_overlap=100,
            )
        )

    embedding_model = EmbeddingModel()
    embedded_chunks = embedding_model.encode(chunks)

    return SemanticRetriever(
        embedding_model,
        embedded_chunks,
   )


def main() -> None:
    print("Loading documents...")
    retriever = build_retriever()

    print("Initializing local LLM...")
    generator = LocalLLMGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    question = input("\nQuestion: ").strip()

    if not question:
        print("No question provided.")
        return

    response = pipeline.answer(
        question,
        top_k=3,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(response.answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for index, source in enumerate(
        response.sources,
        start=1,
    ):
        print(
            f"{index}. "
            f"{source.chunk.source} "
            f"(score={source.score:.4f})"
        )


if __name__ == "__main__":
    main()