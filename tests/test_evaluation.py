from evaluation.generation import answer_similarity, evaluate_answer
from evaluation.retrieval import evaluate_retrieval


def test_retrieval_evaluation() -> None:
    result = evaluate_retrieval(
        expected_sources=["python.md"],
        retrieved_sources=[
            "python.md",
            "docker.md",
            "fastapi.md",
        ],
    )

    assert result.matched_sources == ["python.md"]
    assert result.recall == 1.0
    assert round(result.precision, 4) == round(1 / 3, 4)


def test_retrieval_misses_expected_source() -> None:
    result = evaluate_retrieval(
        expected_sources=["python.md"],
        retrieved_sources=["docker.md"],
    )

    assert result.matched_sources == []
    assert result.recall == 0.0


def test_answer_similarity() -> None:
    score = answer_similarity(
        generated_answer=(
            "Create a virtual environment using python -m venv."
        ),
        reference_answer=(
            "Use python -m venv to create an isolated environment."
        ),
    )

    assert score > 0.4


def test_answer_with_context_is_grounded() -> None:
    result = evaluate_answer(
        generated_answer=(
            "Use python -m venv to create an isolated environment."
        ),
        reference_answer=(
            "Use python -m venv to create an isolated environment."
        ),
        context=(
            "Python virtual environments can be created with "
            "python -m venv."
        ),
    )

    assert result.grounded is True
    assert result.similarity > 0.4