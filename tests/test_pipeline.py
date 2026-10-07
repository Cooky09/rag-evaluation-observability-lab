from dataclasses import dataclass

from rag.pipeline import RAGPipeline


@dataclass
class FakeChunk:
    source: str
    text: str


@dataclass
class FakeResult:
    chunk: FakeChunk
    score: float


class FakeRetriever:
    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[FakeResult]:
        return [
            FakeResult(
                chunk=FakeChunk(
                    source="python.md",
                    text="Use python -m venv .venv.",
                ),
                score=0.9,
            )
        ]


class FakeGenerator:
    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        assert question == "How do I create a virtual environment?"
        assert "python -m venv .venv" in context

        return "Use python -m venv .venv."


def test_rag_pipeline() -> None:
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    response = pipeline.answer(
        "How do I create a virtual environment?"
    )

    assert response.answer == "Use python -m venv .venv."
    assert len(response.sources) == 1
    assert response.sources[0].chunk.source == "python.md"