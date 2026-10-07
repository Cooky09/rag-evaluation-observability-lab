import torch
from torch import nn


SUPPORTED_PRECISIONS = {"fp32", "fp16"}


def validate_precision(
    device: torch.device,
    precision: str,
) -> None:
    """Validate that the requested precision is supported."""

    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            "precision must be either 'fp32' or 'fp16'."
        )

    if precision == "fp16" and device.type == "cpu":
        raise ValueError(
            "FP16 inference is reserved for supported GPU devices."
        )


def prepare_model(
    model: nn.Module,
    device: torch.device,
    precision: str,
) -> nn.Module:
    """Move a model to a device and apply the requested precision."""

    validate_precision(
        device=device,
        precision=precision,
    )

    model = model.to(device)

    if precision == "fp16":
        model = model.half()
    else:
        model = model.float()

    model.eval()

    return model