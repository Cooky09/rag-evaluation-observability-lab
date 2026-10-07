import torch

from inference.model import (
    BenchmarkModel,
    create_benchmark_model,
)


def test_model_output_shape() -> None:
    model = create_benchmark_model(
        device=torch.device("cpu"),
    )

    inputs = torch.randn(
        2,
        1024,
    )

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (2, 512)


def test_model_is_in_evaluation_mode() -> None:
    model = create_benchmark_model(
        device=torch.device("cpu"),
    )

    assert isinstance(model, BenchmarkModel)
    assert model.training is False


def test_model_is_on_cpu() -> None:
    model = create_benchmark_model(
        device=torch.device("cpu"),
    )

    first_parameter = next(model.parameters())

    assert first_parameter.device.type == "cpu"