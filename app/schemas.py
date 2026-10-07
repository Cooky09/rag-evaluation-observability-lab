from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for a RAG query."""

    question: str = Field(
        min_length=1,
        description="Question to ask the RAG system.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve.",
    )


class SourceResponse(BaseModel):
    """Retrieved source information."""

    chunk_id: str
    document_id: str
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    """RAG query response."""

    question: str
    answer: str
    sources: list[SourceResponse]