from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for model inference."""

    batch_size: int = 1
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.precision not in {"fp32", "fp16"}:
            raise ValueError(
                "precision must be either 'fp32' or 'fp16'."
            )