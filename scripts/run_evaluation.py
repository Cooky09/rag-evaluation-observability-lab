import json
import subprocess
import sys
from pathlib import Path

from evaluation.regression import check_regression
from evaluation.runner import evaluate_question
from rag.chunker import chunk_document
from rag.document_loader import load_documents
from rag.embeddings import EmbeddingModel
from rag.generator import LocalLLMGenerator
from rag.pipeline import RAGPipeline
from rag.retriever import SemanticRetriever


EVALUATION_PATH = Path("data/evaluation.json")
OUTPUT_PATH = Path("results/evaluation_report.json")


def build_pipeline() -> RAGPipeline:
    documents = load_documents("data/documents")

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    embedding_model = EmbeddingModel()

    embedded_chunks = embedding_model.encode(chunks)

    retriever = SemanticRetriever(
        embedding_model=embedding_model,
        embedded_chunks=embedded_chunks,
    )

    generator = LocalLLMGenerator()

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
    )


def main() -> None:
    evaluation_cases = json.loads(
        EVALUATION_PATH.read_text(
            encoding="utf-8",
        )
    )

    pipeline = build_pipeline()

    results = []

    for case in evaluation_cases:
        result = evaluate_question(
            pipeline=pipeline,
            question_id=case["id"],
            question=case["question"],
            expected_sources=(
                [case["expected_document"]]
                if "expected_document" in case
                else []
            ),
            reference_answer=case["reference_answer"],
            expected_facts=case["expected_facts"],
        )

        results.append(result)

        print(
            f"{case['id']} | "
            f"retrieval_precision="
            f"{result.retrieval.precision:.2f} | "
            f"retrieval_recall="
            f"{result.retrieval.recall:.2f} | "
            f"answer_similarity="
            f"{result.answer.similarity:.2f} | "
            f"fact_coverage="
            f"{result.answer.fact_coverage:.2f} | "
            f"grounded="
            f"{result.answer.grounded}"
        )

    average_precision = sum(
        result.retrieval.precision
        for result in results
    ) / len(results)

    average_recall = sum(
        result.retrieval.recall
        for result in results
    ) / len(results)

    average_similarity = sum(
        result.answer.similarity
        for result in results
    ) / len(results)

    average_fact_coverage = sum(
        result.answer.fact_coverage
        for result in results
    ) / len(results)

    grounded_answers = sum(
        result.answer.grounded
        for result in results
    )

    grounded_rate = grounded_answers / len(results)

    regression = check_regression(
        retrieval_recall=average_recall,
        answer_similarity=average_similarity,
        fact_coverage=average_fact_coverage,
        grounded_rate=grounded_rate,
    )

    print()

    if regression.passed:
        print("Regression check: PASS")
    else:
        print("Regression check: FAIL")

        for failure in regression.failures:
            print(f"  - {failure}")

    report = {
        "questions_evaluated": len(results),
        "retrieval": {
            "average_precision": average_precision,
            "average_recall": average_recall,
        },
        "answer": {
            "average_similarity": average_similarity,
            "average_fact_coverage": average_fact_coverage,
            "grounded_answers": grounded_answers,
            "grounded_rate": grounded_rate,
        },
        "regression": {
            "passed": regression.passed,
            "failures": regression.failures,
        },
        "questions": [
            {
                "question_id": result.question_id,
                "question": result.question,
                "retrieval": {
                    "expected_sources": (
                        result.retrieval.expected_sources
                    ),
                    "retrieved_sources": (
                        result.retrieval.retrieved_sources
                    ),
                    "matched_sources": (
                        result.retrieval.matched_sources
                    ),
                    "precision": result.retrieval.precision,
                    "recall": result.retrieval.recall,
                },
                "answer": {
                    "generated_answer": (
                        result.answer.generated_answer
                    ),
                    "reference_answer": (
                        result.answer.reference_answer
                    ),
                    "similarity": result.answer.similarity,
                    "facts_supported": (
                        result.answer.facts_supported
                    ),
                    "facts_total": result.answer.facts_total,
                    "fact_coverage": (
                        result.answer.fact_coverage
                    ),
                    "grounded": result.answer.grounded,
                },
            }
            for result in results
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_report",
        ],
        check=True,
    )

    print()
    print("Full evaluation workflow completed successfully.")
    print("Generated:")
    print("  results/evaluation_report.json")
    print("  results/quality_performance_report.json")
    print()
    print("Evaluation completed successfully.")
    print(f"Questions evaluated: {len(results)}")
    print(
        f"Average retrieval recall: "
        f"{average_recall:.3f}"
    )
    print(
        f"Average answer similarity: "
        f"{average_similarity:.3f}"
    )
    print(
        f"Average fact coverage: "
        f"{average_fact_coverage:.3f}"
    )
    print(
        f"Grounded rate: "
        f"{grounded_rate:.3f}"
    )
    print(f"Saved report to: {OUTPUT_PATH}")

    if not regression.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()