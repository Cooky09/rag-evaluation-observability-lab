from rag.chunker import chunk_document
from rag.document_loader import load_documents

DOCUMENTS_DIR = "data/documents"


def main() -> None:
    documents = load_documents(DOCUMENTS_DIR)

    for document in documents:
        chunks = chunk_document(
            document,
            chunk_size=300,
            chunk_overlap=0,
        )

        print(f"\nDocument: {document.source}")
        print(f"Characters: {len(document.text)}")
        print(f"Chunks: {len(chunks)}")

        for chunk in chunks:
            print(
                f"  {chunk.chunk_id}: "
                f"{len(chunk.text)} characters"
            )
            print(f"    {chunk.text}")


if __name__ == "__main__":
    main()