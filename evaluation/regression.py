from dataclasses import dataclass


@dataclass(frozen=True)
class RegressionThresholds:
    """Minimum acceptable evaluation metrics."""

    minimum_retrieval_recall: float = 0.90
    minimum_answer_similarity: float = 0.25
    minimum_fact_coverage: float = 0.25
    minimum_grounded_rate: float = 0.30


@dataclass(frozen=True)
class RegressionResult:
    """Result of comparing evaluation metrics against thresholds."""

    passed: bool
    retrieval_recall: float
    answer_similarity: float
    fact_coverage: float
    grounded_rate: float
    failures: list[str]


def check_regression(
    *,
    retrieval_recall: float,
    answer_similarity: float,
    fact_coverage: float,
    grounded_rate: float,
    thresholds: RegressionThresholds | None = None,
) -> RegressionResult:
    """Check evaluation metrics against minimum quality thresholds."""

    thresholds = thresholds or RegressionThresholds()

    failures: list[str] = []

    if retrieval_recall < thresholds.minimum_retrieval_recall:
        failures.append(
            "Retrieval recall is below the minimum threshold."
        )

    if answer_similarity < thresholds.minimum_answer_similarity:
        failures.append(
            "Answer similarity is below the minimum threshold."
        )

    if fact_coverage < thresholds.minimum_fact_coverage:
        failures.append(
            "Fact coverage is below the minimum threshold."
        )

    if grounded_rate < thresholds.minimum_grounded_rate:
        failures.append(
            "Grounded rate is below the minimum threshold."
        )

    return RegressionResult(
        passed=not failures,
        retrieval_recall=retrieval_recall,
        answer_similarity=answer_similarity,
        fact_coverage=fact_coverage,
        grounded_rate=grounded_rate,
        failures=failures,
    )