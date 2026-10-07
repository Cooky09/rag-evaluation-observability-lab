import torch

from inference.device import get_device, get_device_info
from inference.gpu_metrics import (
    get_gpu_memory,
    get_gpu_name,
)
from inference.model import create_benchmark_model


def main() -> None:
    device = get_device()

    model = create_benchmark_model(device)

    inputs = torch.randn(
        4,
        1024,
        device=device,
    )

    print("Inference device information")
    print("----------------------------")

    info = get_device_info()

    for key, value in info.items():
        print(f"{key}: {value}")

    print()
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {inputs.device}")

    print()
    print("GPU information")
    print("----------------")

    print(f"GPU name: {get_gpu_name()}")

    memory = get_gpu_memory()

    for key, value in memory.items():
        print(f"{key}: {value:.2f} MB")


if __name__ == "__main__":
    main()