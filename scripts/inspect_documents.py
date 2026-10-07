
from pathlib import Path

from rag.document_loader import load_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main() -> None:
    documents = load_documents(DOCUMENTS_DIR)

    print(f"Documents loaded: {len(documents)}")
    print("-" * 50)

    for document in documents:
        print(f"ID: {document.document_id}")
        print(f"Source: {document.source}")
        print(f"Characters: {len(document.text)}")
        print(f"Preview: {document.text[:120]!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()