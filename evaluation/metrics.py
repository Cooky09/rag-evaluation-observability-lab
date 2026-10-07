from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Evaluation of the documents retrieved for a question."""

    expected_sources: list[str]
    retrieved_sources: list[str]
    matched_sources: list[str]
    precision: float
    recall: float


@dataclass(frozen=True)
class AnswerEvaluation:
    """Evaluation of a generated answer against a reference answer."""

    reference_answer: str
    generated_answer: str
    similarity: float
    grounded: bool
    facts_supported: int
    facts_total: int
    fact_coverage: float


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation result for one RAG question."""

    question_id: str
    question: str
    retrieval: RetrievalEvaluation
    answer: AnswerEvaluation