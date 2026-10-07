import pytest
import torch

from inference.model import BenchmarkModel
from inference.precision import (
    prepare_model,
    validate_precision,
)


def test_fp32_is_valid_on_cpu() -> None:
    validate_precision(
        device=torch.device("cpu"),
        precision="fp32",
    )


def test_fp16_is_rejected_on_cpu() -> None:
    with pytest.raises(ValueError):
        validate_precision(
            device=torch.device("cpu"),
            precision="fp16",
        )


def test_invalid_precision_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_precision(
            device=torch.device("cpu"),
            precision="int8",
        )


def test_prepare_model_fp32_on_cpu() -> None:
    model = BenchmarkModel()

    prepared = prepare_model(
        model=model,
        device=torch.device("cpu"),
        precision="fp32",
    )

    parameter = next(prepared.parameters())

    assert parameter.device.type == "cpu"
    assert parameter.dtype == torch.float32
    assert prepared.training is False


def test_prepare_model_fp16_is_rejected_on_cpu() -> None:
    model = BenchmarkModel()

    with pytest.raises(ValueError):
        prepare_model(
            model=model,
            device=torch.device("cpu"),
            precision="fp16",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this machine.",
)
def test_prepare_model_fp16_on_cuda() -> None:
    model = BenchmarkModel()

    prepared = prepare_model(
        model=model,
        device=torch.device("cuda"),
        precision="fp16",
    )

    parameter = next(prepared.parameters())

    assert parameter.device.type == "cuda"
    assert parameter.dtype == torch.float16
    assert prepared.training is False