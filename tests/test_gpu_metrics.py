import torch

from inference.gpu_metrics import (
    get_gpu_memory,
    get_gpu_name,
    synchronize_cuda,
)

from inference.gpu_metrics import (
    get_gpu_memory,
    get_gpu_name,
    measure_gpu_memory_delta,
    synchronize_cuda,
)

def test_gpu_memory_is_safe_without_cuda() -> None:
    memory = get_gpu_memory()

    assert "allocated_mb" in memory
    assert "reserved_mb" in memory
    assert "total_mb" in memory

    if not torch.cuda.is_available():
        assert memory["allocated_mb"] == 0.0
        assert memory["reserved_mb"] == 0.0
        assert memory["total_mb"] == 0.0


def test_gpu_name() -> None:
    name = get_gpu_name()

    if torch.cuda.is_available():
        assert isinstance(name, str)
        assert name
    else:
        assert name is None


def test_cuda_synchronization_is_safe() -> None:
    synchronize_cuda()

def test_gpu_memory_delta_is_safe_without_cuda() -> None:
    memory = measure_gpu_memory_delta(
        inference_fn=lambda: None,
    )

    assert memory["before_mb"] == 0.0
    assert memory["after_mb"] == 0.0
    assert memory["delta_mb"] == 0.0
    assert memory["peak_mb"] == 0.0