
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A source document loaded from the knowledge base."""

    document_id: str
    source: str
    text: str


def load_documents(directory: str | Path) -> list[Document]:
    """Load Markdown documents from a directory."""
    documents_path = Path(directory)

    if not documents_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {documents_path}"
        )

    if not documents_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {documents_path}"
        )

    documents: list[Document] = []

    for file_path in sorted(documents_path.glob("*.md")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            Document(
                document_id=file_path.stem,
                source=file_path.name,
                text=text,
            )
        )

    return documents