import torch

from inference.device import (
    cuda_available,
    get_device,
    get_device_info,
)


def test_device_is_valid() -> None:
    device = get_device()

    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_cuda_availability_matches_pytorch() -> None:
    assert cuda_available() == torch.cuda.is_available()


def test_device_info() -> None:
    info = get_device_info()

    assert "device" in info
    assert "cuda_available" in info

    if torch.cuda.is_available():
        assert info["device"] == "cuda"
        assert info["device_count"] >= 1
    else:
        assert info["device"] == "cpu"
        assert info["device_count"] == 0