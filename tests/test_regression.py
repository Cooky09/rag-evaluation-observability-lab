from evaluation.regression import (
    RegressionThresholds,
    check_regression,
)


def test_regression_passes_when_all_metrics_meet_thresholds() -> None:
    result = check_regression(
        retrieval_recall=1.0,
        answer_similarity=0.272,
        fact_coverage=0.3,
        grounded_rate=0.4,
    )

    assert result.passed is True
    assert result.failures == []


def test_regression_fails_when_answer_similarity_is_too_low() -> None:
    result = check_regression(
        retrieval_recall=1.0,
        answer_similarity=0.10,
        fact_coverage=0.3,
        grounded_rate=0.4,
    )

    assert result.passed is False
    assert "Answer similarity is below the minimum threshold." in result.failures


def test_regression_fails_when_multiple_metrics_are_too_low() -> None:
    result = check_regression(
        retrieval_recall=0.5,
        answer_similarity=0.10,
        fact_coverage=0.1,
        grounded_rate=0.1,
    )

    assert result.passed is False
    assert len(result.failures) == 4


def test_custom_thresholds_are_supported() -> None:
    thresholds = RegressionThresholds(
        minimum_retrieval_recall=0.95,
        minimum_answer_similarity=0.30,
        minimum_fact_coverage=0.40,
        minimum_grounded_rate=0.50,
    )

    result = check_regression(
        retrieval_recall=1.0,
        answer_similarity=0.35,
        fact_coverage=0.5,
        grounded_rate=0.6,
        thresholds=thresholds,
    )

    assert result.passed is True