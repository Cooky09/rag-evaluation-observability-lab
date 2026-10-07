import pytest
import torch

from inference.benchmark import (
    BenchmarkResult,
    benchmark_callable,
    benchmark_model,
)
from inference.config import InferenceConfig
from inference.model import create_benchmark_model


def test_cpu_benchmark() -> None:
    config = InferenceConfig(
        batch_size=2,
        precision="fp32",
    )

    counter = {"calls": 0}

    def inference() -> None:
        counter["calls"] += 1

    result = benchmark_callable(
        inference_fn=inference,
        device=torch.device("cpu"),
        config=config,
        iterations=5,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.device == "cpu"
    assert result.precision == "fp32"
    assert result.batch_size == 2
    assert result.iterations == 5
    assert result.total_time_ms >= 0
    assert result.average_latency_ms >= 0
    assert result.throughput > 0

    # One warm-up + five measured iterations.
    assert counter["calls"] == 6


def test_real_model_benchmark() -> None:
    model = create_benchmark_model(
        device=torch.device("cpu"),
    )

    inputs = torch.randn(
        2,
        1024,
    )

    config = InferenceConfig(
        batch_size=2,
        precision="fp32",
    )

    result = benchmark_model(
        model=model,
        inputs=inputs,
        device=torch.device("cpu"),
        config=config,
        iterations=3,
    )

    assert result.device == "cpu"
    assert result.precision == "fp32"
    assert result.batch_size == 2
    assert result.iterations == 3
    assert result.average_latency_ms > 0
    assert result.throughput > 0


def test_real_model_fp16_is_rejected_on_cpu() -> None:
    model = create_benchmark_model(
        device=torch.device("cpu"),
    )

    inputs = torch.randn(
        2,
        1024,
    )

    config = InferenceConfig(
        batch_size=2,
        precision="fp16",
    )

    with pytest.raises(ValueError):
        benchmark_model(
            model=model,
            inputs=inputs,
            device=torch.device("cpu"),
            config=config,
            iterations=3,
        )


def test_invalid_iterations() -> None:
    config = InferenceConfig()

    with pytest.raises(ValueError):
        benchmark_callable(
            inference_fn=lambda: None,
            device=torch.device("cpu"),
            config=config,
            iterations=0,
        )


def test_cpu_fp16_is_rejected() -> None:
    config = InferenceConfig(
        precision="fp16",
    )

    with pytest.raises(ValueError):
        benchmark_callable(
            inference_fn=lambda: None,
            device=torch.device("cpu"),
            config=config,
            iterations=1,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this machine.",
)
def test_cuda_benchmark() -> None:
    config = InferenceConfig(
        batch_size=2,
        precision="fp16",
    )

    result = benchmark_callable(
        inference_fn=lambda: torch.zeros(
            (2, 1024),
            device="cuda",
        ),
        device=torch.device("cuda"),
        config=config,
        iterations=3,
    )

    assert result.device == "cuda"
    assert result.precision == "fp16"
    assert result.throughput > 0