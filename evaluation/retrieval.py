from evaluation.metrics import RetrievalEvaluation


def evaluate_retrieval(
    expected_sources: list[str],
    retrieved_sources: list[str],
) -> RetrievalEvaluation:
    """Evaluate whether expected documents were retrieved."""

    expected = set(expected_sources)
    retrieved = set(retrieved_sources)

    matched = expected & retrieved

    if retrieved:
        precision = len(matched) / len(retrieved)
    else:
        precision = 0.0

    if expected:
        recall = len(matched) / len(expected)
    else:
        recall = 0.0

    return RetrievalEvaluation(
        expected_sources=sorted(expected),
        retrieved_sources=retrieved_sources,
        matched_sources=sorted(matched),
        precision=precision,
        recall=recall,
    )