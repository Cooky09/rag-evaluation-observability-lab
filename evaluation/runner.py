from evaluation.generation import evaluate_answer
from evaluation.metrics import EvaluationResult
from evaluation.retrieval import evaluate_retrieval
from rag.pipeline import RAGPipeline


def evaluate_question(
    pipeline: RAGPipeline,
    question_id: str,
    question: str,
    expected_sources: list[str],
    reference_answer: str,
    expected_facts: list[str],
    top_k: int = 3,
) -> EvaluationResult:
    """Run RAG and evaluate the resulting answer."""

    response = pipeline.answer(
        question=question,
        top_k=top_k,
    )

    retrieved_sources = [
        result.chunk.source
        for result in response.sources
    ]

    context = "\n\n".join(
        result.chunk.text
        for result in response.sources
    )

    retrieval = evaluate_retrieval(
        expected_sources=expected_sources,
        retrieved_sources=retrieved_sources,
    )

    answer = evaluate_answer(
        generated_answer=response.answer,
        reference_answer=reference_answer,
        context=context,
        expected_facts=expected_facts,
    )

    return EvaluationResult(
        question_id=question_id,
        question=question,
        retrieval=retrieval,
        answer=answer,
    )