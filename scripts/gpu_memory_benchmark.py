import torch

from inference.config import InferenceConfig
from inference.gpu_metrics import (
    get_gpu_memory,
    get_gpu_name,
    measure_gpu_memory_delta,
)
from inference.model import create_benchmark_model
from inference.precision import prepare_model


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = InferenceConfig(
        batch_size=16,
        precision="fp32",
    )

    model = create_benchmark_model(device)

    model = prepare_model(
        model=model,
        device=device,
        precision=config.precision,
    )

    inputs = torch.randn(
        config.batch_size,
        1024,
        device=device,
    )

    def inference() -> None:
        with torch.inference_mode():
            model(inputs)

    memory = measure_gpu_memory_delta(
        inference_fn=inference,
    )

    print("GPU memory benchmark")
    print("--------------------")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU name: {get_gpu_name()}")

    print()
    print("Memory before inference:")
    print(f"{memory['before_mb']:.2f} MB")

    print("Memory after inference:")
    print(f"{memory['after_mb']:.2f} MB")

    print("Memory delta:")
    print(f"{memory['delta_mb']:.2f} MB")

    print("Peak allocated memory:")
    print(f"{memory['peak_mb']:.2f} MB")

    print()
    print("Current GPU memory:")
    current = get_gpu_memory()

    for key, value in current.items():
        print(f"{key}: {value:.2f} MB")


if __name__ == "__main__":
    main()