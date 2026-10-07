from rag.generator import LocalLLMGenerator


def test_prompt_contains_question_and_context() -> None:
    prompt = LocalLLMGenerator._build_prompt(
        question="How do I create a virtual environment?",
        context="Use python -m venv .venv.",
    )

    assert "How do I create a virtual environment?" in prompt
    assert "Use python -m venv .venv." in prompt
    assert "context below" in prompt
