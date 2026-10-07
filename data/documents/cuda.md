# CUDA GPU Inference

CUDA is a platform and programming model developed by NVIDIA for running computations on NVIDIA GPUs.

GPU acceleration can be useful for machine learning workloads because many operations can be executed in parallel.

## CPU and GPU

CPU inference executes model operations on the CPU.

GPU inference moves model computation to an NVIDIA GPU.

A PyTorch tensor can be moved to a CUDA device with:

```python
tensor = tensor.to("cuda")