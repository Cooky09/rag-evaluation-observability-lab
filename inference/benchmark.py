from dataclasses import dataclass
import time

import torch

from inference.config import InferenceConfig
from inference.gpu_metrics import measure_gpu_memory_delta
from inference.precision import prepare_model, validate_precision


@dataclass(frozen=True)
class BenchmarkResult:
    """Performance measurements from an inference benchmark."""

    device: str
    precision: str
    batch_size: int
    iterations: int
    total_time_ms: float
    average_latency_ms: float
    throughput: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    peak_memory_mb: float


def _synchronize_if_needed(
    device: torch.device,
) -> None:
    """Synchronize CUDA before measuring GPU work."""

    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_callable(
    inference_fn,
    device: torch.device,
    config: InferenceConfig,
    iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark a callable inference operation."""

    if iterations <= 0:
        raise ValueError(
            "iterations must be greater than zero."
        )

    validate_precision(
        device=device,
        precision=config.precision,
    )

    # Warm-up execution.
    inference_fn()

    _synchronize_if_needed(device)

    start_time = time.perf_counter()

    for _ in range(iterations):
        inference_fn()

    _synchronize_if_needed(device)

    total_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    average_latency_ms = (
        total_time_ms / iterations
    )

    total_samples = (
        config.batch_size * iterations
    )

    total_time_seconds = (
        total_time_ms / 1000
    )

    throughput = (
        total_samples / total_time_seconds
    )

    return BenchmarkResult(
        device=device.type,
        precision=config.precision,
        batch_size=config.batch_size,
        iterations=iterations,
        total_time_ms=total_time_ms,
        average_latency_ms=average_latency_ms,
        throughput=throughput,
        memory_before_mb=0.0,
        memory_after_mb=0.0,
        memory_delta_mb=0.0,
        peak_memory_mb=0.0,
    )


def benchmark_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    config: InferenceConfig,
    iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark a PyTorch model forward pass."""

    model = prepare_model(
        model=model,
        device=device,
        precision=config.precision,
    )

    inputs = inputs.to(device)

    if config.precision == "fp16":
        inputs = inputs.half()
    else:
        inputs = inputs.float()

    def inference() -> None:
        with torch.inference_mode():
            model(inputs)

    memory = measure_gpu_memory_delta(
        inference_fn=inference,
    )

    performance = benchmark_callable(
        inference_fn=inference,
        device=device,
        config=config,
        iterations=iterations,
    )

    return BenchmarkResult(
        device=performance.device,
        precision=performance.precision,
        batch_size=performance.batch_size,
        iterations=performance.iterations,
        total_time_ms=performance.total_time_ms,
        average_latency_ms=performance.average_latency_ms,
        throughput=performance.throughput,
        memory_before_mb=memory["before_mb"],
        memory_after_mb=memory["after_mb"],
        memory_delta_mb=memory["delta_mb"],
        peak_memory_mb=memory["peak_mb"],
    )