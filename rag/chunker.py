from dataclasses import dataclass
import re

from rag.document_loader import Document


@dataclass(frozen=True)
class Chunk:
    """A searchable chunk derived from a source document."""

    chunk_id: str
    document_id: str
    source: str
    text: str


def _split_into_sections(text: str) -> list[str]:
    """Split Markdown into natural sections."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_oversized_section(
    section: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split an oversized section while preserving overlap."""
    if len(section) <= chunk_size:
        return [section]

    parts: list[str] = []
    start = 0

    while start < len(section):
        end = min(start + chunk_size, len(section))
        parts.append(section[start:end])

        if end >= len(section):
            break

        start = end - chunk_overlap

    return parts


def chunk_document(
    document: Document,
    chunk_size: int = 300,
    chunk_overlap: int = 0,
) -> list[Chunk]:
    """Split a document into paragraph-aware chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = document.text.strip()

    if not text:
        return []

    sections = _split_into_sections(text)

    chunks: list[Chunk] = []
    current_sections: list[str] = []
    current_length = 0
    chunk_number = 0

    for section in sections:
        # Normal section: keep it as a semantic unit.
        if len(section) <= chunk_size:
            section_length = len(section)

            if (
                current_sections
                and current_length + section_length + 2 > chunk_size
            ):
                chunk_text = "\n\n".join(current_sections).strip()

                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{document.document_id}"
                            f"-chunk-{chunk_number:03d}"
                        ),
                        document_id=document.document_id,
                        source=document.source,
                        text=chunk_text,
                    )
                )

                chunk_number += 1

                # Apply overlap only to natural sections.
                overlap_sections: list[str] = []
                overlap_length = 0

                for previous in reversed(current_sections):
                    previous_length = len(previous)

                    if (
                        overlap_length
                        + previous_length
                        + 2
                        > chunk_overlap
                    ):
                        break

                    overlap_sections.insert(0, previous)
                    overlap_length += previous_length + 2

                current_sections = overlap_sections
                current_length = overlap_length

            current_sections.append(section)
            current_length += section_length + 2
            continue

        # Oversized section: split it directly.
        # These generated pieces should not receive additional
        # overlap because that would duplicate artificial splits.
        if current_sections:
            chunk_text = "\n\n".join(current_sections).strip()

            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.document_id}"
                        f"-chunk-{chunk_number:03d}"
                    ),
                    document_id=document.document_id,
                    source=document.source,
                    text=chunk_text,
                )
            )

            chunk_number += 1
            current_sections = []
            current_length = 0

        oversized_parts = _split_oversized_section(
        section,
            chunk_size,
            chunk_overlap,
        )

        for part in oversized_parts:
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.document_id}"
                        f"-chunk-{chunk_number:03d}"
                    ),
                    document_id=document.document_id,
                    source=document.source,
                    text=part,
                )
            )
            chunk_number += 1

    # Flush remaining natural sections.
    if current_sections:
        chunk_text = "\n\n".join(current_sections).strip()

        chunks.append(
            Chunk(
                chunk_id=(
                    f"{document.document_id}"
                    f"-chunk-{chunk_number:03d}"
                ),
                document_id=document.document_id,
                source=document.source,
                text=chunk_text,
            )
        )

    return chunks