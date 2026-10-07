import torch


def get_device() -> torch.device:
    """Return the best available inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def cuda_available() -> bool:
    """Return whether CUDA is available."""
    return torch.cuda.is_available()


def get_device_info() -> dict[str, str | bool | int]:
    """Return information about the available inference device."""
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "cuda_available": False,
            "device_count": 0,
        }

    return {
        "device": "cuda",
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
    }