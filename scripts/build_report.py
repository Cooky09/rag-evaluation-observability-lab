import json
from pathlib import Path


EVALUATION_PATH = Path("results/evaluation_report.json")
BENCHMARK_PATH = Path("results/benchmark_matrix.json")
OUTPUT_PATH = Path("results/quality_performance_report.json")


def load_json(path: Path) -> dict | list:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    evaluation = load_json(EVALUATION_PATH)
    benchmarks = load_json(BENCHMARK_PATH)

    if not isinstance(evaluation, dict):
        raise ValueError("Evaluation report must contain a JSON object.")

    if not isinstance(benchmarks, list):
        raise ValueError("Benchmark matrix must contain a JSON list.")

    retrieval = evaluation["retrieval"]
    answer = evaluation["answer"]
    regression = evaluation["regression"]

    benchmark_summary = []

    for result in benchmarks:
        benchmark_summary.append(
            {
                "device": result["device"],
                "precision": result["precision"],
                "batch_size": result["batch_size"],
                "average_latency_ms": result["average_latency_ms"],
                "throughput": result["throughput"],
                "throughput_speedup": result["throughput_speedup"],
                "peak_memory_mb": result["peak_memory_mb"],
            }
        )

    best_throughput = max(
        benchmark_summary,
        key=lambda result: result["throughput"],
    )

    lowest_latency = min(
        benchmark_summary,
        key=lambda result: result["average_latency_ms"],
    )

    return {
        "evaluation_summary": {
            "questions_evaluated": evaluation["questions_evaluated"],
            "retrieval": {
                "average_precision": retrieval["average_precision"],
                "average_recall": retrieval["average_recall"],
            },
            "answer": {
                "average_similarity": answer["average_similarity"],
                "average_fact_coverage": answer["average_fact_coverage"],
                "grounded_answers": answer["grounded_answers"],
                "grounded_rate": answer["grounded_rate"],
            },
            "regression": {
                "passed": regression["passed"],
                "failures": regression["failures"],
            },
        },
        "performance_summary": {
            "benchmark_results": benchmark_summary,
            "best_throughput": best_throughput,
            "lowest_latency": lowest_latency,
        },
        "interpretation": {
            "retrieval_quality": (
                "Retrieval achieved perfect average recall in the "
                "current evaluation set."
            ),
            "generation_quality": (
                "Generated-answer quality is substantially lower "
                "than retrieval quality in the current baseline."
            ),
            "batching": (
                "Increasing batch size improves throughput on the "
                "current CPU benchmark, with batch size 16 producing "
                "the highest measured throughput."
            ),
            "gpu_limitations": (
                "CUDA and FP16 results are unavailable on the current "
                "CPU-only machine."
            ),
        },
    }


def main() -> None:
    report = build_report()

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

    print("Quality vs performance report created.")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print(
        "Best throughput: "
        f"{report['performance_summary']['best_throughput']['throughput']:.2f} "
        "samples/sec"
    )
    print(
        "Lowest latency: "
        f"{report['performance_summary']['lowest_latency']['average_latency_ms']:.3f} ms"
    )
    print(
        "Average retrieval recall: "
        f"{report['evaluation_summary']['retrieval']['average_recall']:.2f}"
    )
    print(
        "Average answer similarity: "
        f"{report['evaluation_summary']['answer']['average_similarity']:.3f}"
    )
    print(
        "Regression status: "
        f"{'PASS' if report['evaluation_summary']['regression']['passed'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
