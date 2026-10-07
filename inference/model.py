import torch
from torch import nn


class BenchmarkModel(nn.Module):
    """Small neural network used for inference benchmarking."""

    def __init__(
        self,
        input_size: int = 1024,
        hidden_size: int = 2048,
        output_size: int = 512,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.network(inputs)


def create_benchmark_model(
    device: torch.device,
) -> BenchmarkModel:
    """Create the benchmark model on the requested device."""
    model = BenchmarkModel()
    model = model.to(device)
    model.eval()

    return model