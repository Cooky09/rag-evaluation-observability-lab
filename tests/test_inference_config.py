import pytest

from inference.config import InferenceConfig


def test_default_config() -> None:
    config = InferenceConfig()

    assert config.batch_size == 1
    assert config.precision == "fp32"


def test_custom_config() -> None:
    config = InferenceConfig(
        batch_size=8,
        precision="fp16",
    )

    assert config.batch_size == 8
    assert config.precision == "fp16"


def test_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        InferenceConfig(batch_size=0)


def test_invalid_precision() -> None:
    with pytest.raises(ValueError):
        InferenceConfig(precision="int8")