import json
from pathlib import Path

import torch

from inference.benchmark import benchmark_model
from inference.config import InferenceConfig
from inference.model import create_benchmark_model


BATCH_SIZES = [1, 2, 4, 8, 16]
ITERATIONS = 20


def main() -> None:
    device = torch.device("cpu")

    model = create_benchmark_model(device)

    results = []

    for batch_size in BATCH_SIZES:
        config = InferenceConfig(
            batch_size=batch_size,
            precision="fp32",
        )

        inputs = torch.randn(
            batch_size,
            1024,
        )

        result = benchmark_model(
            model=model,
            inputs=inputs,
            device=device,
            config=config,
            iterations=ITERATIONS,
        )

        result_data = {
            "device": result.device,
            "precision": result.precision,
            "batch_size": result.batch_size,
            "iterations": result.iterations,
            "total_time_ms": result.total_time_ms,
            "average_latency_ms": result.average_latency_ms,
            "throughput": result.throughput,
        }

        results.append(result_data)

        print(
            f"Batch {batch_size:>2} | "
            f"Latency: {result.average_latency_ms:>8.3f} ms | "
            f"Throughput: {result.throughput:>10.2f} samples/sec"
        )

    output_path = Path(
        "results/cpu_batch_benchmark.json"
    )

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
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()