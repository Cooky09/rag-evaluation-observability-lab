import torch


def get_gpu_memory() -> dict[str, float]:
    """Return GPU memory statistics in megabytes."""

    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "total_mb": 0.0,
        }

    device = torch.cuda.current_device()

    allocated_mb = (
        torch.cuda.memory_allocated(device) / (1024 ** 2)
    )

    reserved_mb = (
        torch.cuda.memory_reserved(device) / (1024 ** 2)
    )

    total_mb = (
        torch.cuda.get_device_properties(device).total_memory
        / (1024 ** 2)
    )

    return {
        "allocated_mb": allocated_mb,
        "reserved_mb": reserved_mb,
        "total_mb": total_mb,
    }


def get_gpu_name() -> str | None:
    """Return the current GPU name when CUDA is available."""

    if not torch.cuda.is_available():
        return None

    return torch.cuda.get_device_name(
        torch.cuda.current_device()
    )


def synchronize_cuda() -> None:
    """Synchronize the current CUDA device when available."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()

def measure_gpu_memory_delta(
    inference_fn,
) -> dict[str, float]:
    """Measure GPU memory change caused by an operation."""

    if not torch.cuda.is_available():
        return {
            "before_mb": 0.0,
            "after_mb": 0.0,
            "delta_mb": 0.0,
            "peak_mb": 0.0,
        }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    synchronize_cuda()

    before_mb = (
        torch.cuda.memory_allocated() / (1024 ** 2)
    )

    inference_fn()

    synchronize_cuda()

    after_mb = (
        torch.cuda.memory_allocated() / (1024 ** 2)
    )

    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )

    return {
        "before_mb": before_mb,
        "after_mb": after_mb,
        "delta_mb": after_mb - before_mb,
        "peak_mb": peak_mb,
    }