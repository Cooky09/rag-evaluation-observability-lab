import time
from dataclasses import dataclass

from observability.logging import get_logger
from observability.metrics import Metrics
from rag.generator import LocalLLMGenerator
from rag.retriever import RetrievalResult, SemanticRetriever


logger = get_logger(__name__)


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievalResult]


class RAGPipeline:
    def __init__(
        self,
        retriever: SemanticRetriever,
        generator: LocalLLMGenerator,
        metrics: Metrics | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.metrics = metrics or Metrics()

    def answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> RAGResponse:
        retrieval_start = time.perf_counter()

        results = self.retriever.search(
            question,
            top_k=top_k,
        )

        retrieval_duration_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        self.metrics.record_retrieval(
            retrieval_duration_ms
        )

        logger.info(
            "Retrieval completed | duration_ms=%.2f | results=%d",
            retrieval_duration_ms,
            len(results),
        )

        context = self._build_context(results)

        generation_start = time.perf_counter()

        answer = self.generator.generate(
            question=question,
            context=context,
        )

        generation_duration_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        self.metrics.record_generation(
            generation_duration_ms
        )

        logger.info(
            "Generation completed | duration_ms=%.2f",
            generation_duration_ms,
        )

        total_duration_ms = (
            retrieval_duration_ms
            + generation_duration_ms
        )

        self.metrics.record_query(
            total_duration_ms
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=results,
        )

    @staticmethod
    def _build_context(
        results: list[RetrievalResult],
    ) -> str:
        sections: list[str] = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            sections.append(
                (
                    f"[Source {index}: "
                    f"{result.chunk.source}]\n"
                    f"{result.chunk.text}"
                )
            )

        return "\n\n".join(sections)
