import json
from pathlib import Path

import torch

from inference.benchmark import benchmark_model
from inference.config import InferenceConfig
from inference.model import create_benchmark_model
from inference.precision import validate_precision


BATCH_SIZES = [1, 2, 4, 8, 16]
PRECISIONS = ["fp32", "fp16"]
ITERATIONS = 20


def run_benchmark(
    device: torch.device,
    precision: str,
    batch_size: int,
) -> dict[str, float | int | str]:
    config = InferenceConfig(
        batch_size=batch_size,
        precision=precision,
    )

    validate_precision(
        device=device,
        precision=precision,
    )

    model = create_benchmark_model(device)

    inputs = torch.randn(
        batch_size,
        1024,
        device=device,
    )

    result = benchmark_model(
        model=model,
        inputs=inputs,
        device=device,
        config=config,
        iterations=ITERATIONS,
    )

    return {
        "device": result.device,
        "precision": result.precision,
        "batch_size": result.batch_size,
        "iterations": result.iterations,
        "total_time_ms": result.total_time_ms,
        "average_latency_ms": result.average_latency_ms,
        "throughput": result.throughput,
        "memory_before_mb": result.memory_before_mb,
        "memory_after_mb": result.memory_after_mb,
        "memory_delta_mb": result.memory_delta_mb,
        "peak_memory_mb": result.peak_memory_mb,
    }


def main() -> None:
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    results: list[dict[str, float | int | str]] = []

    for device in devices:
        for precision in PRECISIONS:
            baseline_throughput: float | None = None

            for batch_size in BATCH_SIZES:
                try:
                    result = run_benchmark(
                        device=device,
                        precision=precision,
                        batch_size=batch_size,
                    )
                except ValueError as exc:
                    print(
                        f"Skipped | Device: {device} | "
                        f"Precision: {precision} | "
                        f"Batch: {batch_size} | "
                        f"Reason: {exc}"
                    )
                    continue

                if baseline_throughput is None:
                    baseline_throughput = float(
                        result["throughput"]
                    )

                throughput_speedup = (
                    float(result["throughput"])
                    / baseline_throughput
                )

                result["throughput_speedup"] = throughput_speedup

                results.append(result)

                print(
                    f"Device: {result['device']:>4} | "
                    f"Precision: {result['precision']:>4} | "
                    f"Batch: {result['batch_size']:>2} | "
                    f"Latency: "
                    f"{result['average_latency_ms']:>8.3f} ms | "
                    f"Throughput: "
                    f"{result['throughput']:>10.2f} samples/sec | "
                    f"Speedup: "
                    f"{throughput_speedup:>6.2f}x | "
                    f"Peak memory: "
                    f"{result['peak_memory_mb']:>8.2f} MB"
                )

    output_path = Path("results/benchmark_matrix.json")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved {len(results)} benchmark results to:"
    )
    print(output_path)


if __name__ == "__main__":
    main()