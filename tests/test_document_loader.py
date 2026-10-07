
from pathlib import Path

import pytest

from rag.document_loader import Document, load_documents


def test_loads_markdown_documents(tmp_path: Path) -> None:
    """Markdown files should load with their text and metadata."""
    (tmp_path / "python.md").write_text(
        "# Python\nVirtual environments isolate dependencies.",
        encoding="utf-8",
    )
    (tmp_path / "docker.md").write_text(
        "# Docker\nContainers package applications.",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text(
        "This file should not be loaded.",
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert documents[0] == Document(
        document_id="docker",
        source="docker.md",
        text="# Docker\nContainers package applications.",
    )
    assert documents[1].document_id == "python"
    assert "isolate dependencies" in documents[1].text


def test_skips_empty_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")

    assert load_documents(tmp_path) == []


def test_missing_directory_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        load_documents(missing_path)


def test_file_instead_of_directory_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_documents(file_path)