import re

from evaluation.metrics import AnswerEvaluation


def _tokenize(text: str) -> set[str]:
    """Convert text into a normalized set of tokens."""

    return {
        token
        for token in re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
        if len(token) > 2
    }


def answer_similarity(
    generated_answer: str,
    reference_answer: str,
) -> float:
    """Calculate token-overlap similarity between two answers."""

    generated_tokens = _tokenize(generated_answer)
    reference_tokens = _tokenize(reference_answer)

    if not reference_tokens:
        return 0.0

    overlap = generated_tokens & reference_tokens

    return len(overlap) / len(reference_tokens)


def evaluate_facts(
    generated_answer: str,
    expected_facts: list[str],
) -> tuple[int, int, float]:
    """Evaluate how many expected facts appear in the answer."""

    if not expected_facts:
        return 0, 0, 0.0

    generated_tokens = _tokenize(generated_answer)

    supported = 0

    for fact in expected_facts:
        fact_tokens = _tokenize(fact)

        if not fact_tokens:
            continue

        overlap = generated_tokens & fact_tokens

        coverage = len(overlap) / len(fact_tokens)

        if coverage >= 0.40:
            supported += 1

    total = len(expected_facts)

    return (
        supported,
        total,
        supported / total,
    )


def evaluate_answer(
    generated_answer: str,
    reference_answer: str,
    context: str,
    expected_facts: list[str] | None = None,
) -> AnswerEvaluation:
    """Evaluate a generated answer."""

    similarity = answer_similarity(
        generated_answer=generated_answer,
        reference_answer=reference_answer,
    )

    grounded = _is_grounded(
        generated_answer=generated_answer,
        context=context,
    )

    facts_supported = 0
    facts_total = 0
    fact_coverage = 0.0

    if expected_facts is not None:
        (
            facts_supported,
            facts_total,
            fact_coverage,
        ) = evaluate_facts(
            generated_answer=generated_answer,
            expected_facts=expected_facts,
        )

    return AnswerEvaluation(
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        similarity=similarity,
        grounded=grounded,
        facts_supported=facts_supported,
        facts_total=facts_total,
        fact_coverage=fact_coverage,
    )


def _is_grounded(
    generated_answer: str,
    context: str,
) -> bool:
    """Check whether important answer tokens appear in the context."""

    answer_tokens = _tokenize(generated_answer)
    context_tokens = _tokenize(context)

    if not answer_tokens:
        return False

    overlap = answer_tokens & context_tokens

    return len(overlap) / len(answer_tokens) >= 0.30