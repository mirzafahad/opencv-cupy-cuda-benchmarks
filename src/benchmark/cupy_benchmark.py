"""
Benchmarking NumPy vs CuPy for image normalization.

This module compares CPU (NumPy) and GPU (CuPy) performance for batch
image normalization operations commonly used in deep learning preprocessing.
"""

import time

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from benchmark.utils.logger import Logger


logger = Logger()


# Normalization parameters (ImageNet stats)
MEAN = [123.675, 116.28, 103.53]  # RGB channel means
STD = [58.395, 57.12, 57.375]  # RGB channel standard deviations
# Pre-computed arrays for performance (avoid recreating on each function call)
MEAN_GPU = cp.array(MEAN, dtype=cp.float16)
STD_GPU = cp.array(STD, dtype=cp.float16)
MEAN_CPU = np.array(MEAN, dtype=np.float16)
STD_CPU = np.array(STD, dtype=np.float16)


def normalize_using_cupy(
    images: list[NDArray], return_gpu_arrays: bool = False
) -> list[NDArray] | list[cp.ndarray]:
    """
    GPU normalization using CuPy.

    Args:
        images: List of RGB images to normalize.
        return_gpu_arrays: If True, keep the end result on GPU array. If False, download to CPU memory.

    Returns:
        List of normalized arrays. Each array has shape (1, C, H, W).
        Returns CuPy arrays if return_gpu_arrays=True, NumPy arrays otherwise.
    """
    # Stack all images into single batch (N, H, W, C).
    images_batch = np.stack(images)
    # Upload to GPU.
    images_batch_gpu = cp.asarray(images_batch)

    # Normalize all images in parallel: (pixel - mean) / std
    normalized_batch_gpu = (images_batch_gpu - MEAN_GPU) / STD_GPU

    # Batch transpose: (N, H, W, C) -> (N, C, H, W)
    # PyTorch/deep learning frameworks expect channels-first format.
    transposed_batch_gpu = cp.transpose(normalized_batch_gpu, (0, 3, 1, 2))

    if return_gpu_arrays:
        # Keep on GPU - no download overhead!
        # Extract each image and add batch dimension: (C, H, W) -> (1, C, H, W)
        # Downstream inference models expect individual images with batch dimension.
        results_gpu = [
            cp.expand_dims(transposed_batch_gpu[i], axis=0)  # (1, 3, 640, 640) per image
            for i in range(len(images))
        ]

        return results_gpu
    else:
        # Download to CPU.
        batch_cpu = cp.asnumpy(transposed_batch_gpu)
        # Extract each image and add batch dimension: (C, H, W) -> (1, C, H, W)
        results_cpu = [
            np.expand_dims(batch_cpu[i], axis=0)  # (1, 3, 640, 640) per image.
            for i in range(len(images))
        ]

        return results_cpu


def normalize_using_numpy(images: list[NDArray]) -> list[NDArray]:
    """
    CPU normalization using numpy.

    Args:
        images: Images to normalize.

    Returns:
         List of normalized arrays. Each array has shape (1, C, H, W).
    """
    # Stack all images into single batch (N, H, W, C).
    images_batch = np.stack(images)

    # Normalize all images in parallel.
    normalized_batch = (images_batch - MEAN_CPU) / STD_CPU

    # Batch transpose: (N, H, W, C) -> (N, C, H, W)
    transposed_batch = np.transpose(normalized_batch, (0, 3, 1, 2))

    # Extract each image and add batch dimension: (C, H, W) -> (1, C, H, W)
    results = [
        np.expand_dims(transposed_batch[i], axis=0)  # (1, 3, 640, 640) per image
        for i in range(len(images))
    ]

    return results


def validate_cpu_and_gpu_results(images: list[NDArray]) -> None:
    """
    Validate that CPU and GPU normalization produce equivalent results.

    Compares the output of normalize_using_numpy() and normalize_using_cupy()
    to ensure both implementations produce the same numerical results within
    acceptable floating-point tolerance.

    Args:
        images: List of input images to validate normalization on.

    Raises:
        ValueError: If CPU and GPU results differ beyond tolerance thresholds
                   (rtol=1e-3, atol=1e-5).
    """
    logger.info("Validating CPU vs GPU results...")
    cpu_results = normalize_using_numpy(images)
    gpu_results = normalize_using_cupy(images, return_gpu_arrays=False)

    # Compare each result
    all_match = True
    for i, (cpu_arr, gpu_arr) in enumerate(zip(cpu_results, gpu_results)):
        if not np.allclose(cpu_arr, gpu_arr, rtol=1e-3, atol=1e-5):
            logger.info(f"Mismatch at image {i}: max diff = {np.abs(cpu_arr - gpu_arr).max()}")
            all_match = False

    if all_match:
        logger.info("✓ Validation passed: CPU and GPU results match")
    else:
        raise ValueError("✗ Validation failed: CPU and GPU results differ")


if __name__ == "__main__":
    # Benchmark configuration
    # How many frame we batch for the operation.
    NUM_IMAGES = 6
    # Let's assume our inference model takes 640-by-640 RGB images.
    IMAGE_SIZE = (640, 640, 3)
    # Number of iterations for averaging timing results.
    NUM_ITERATIONS = 20

    # Generate random dummy images for benchmarking.
    # Using uint8 [0, 255] to simulate real image data.
    dummy_images = []
    for _ in range(NUM_IMAGES):
        dummy_images.append(np.random.randint(0, 256, size=IMAGE_SIZE, dtype=np.uint8))

    logger.info(f"GPU Device: {cp.cuda.Device()}")
    logger.info(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")

    # Warm-up: First GPU operation initializes CUDA context and compiles kernels.
    # This can take 100-500ms and would skew benchmark results if not done separately.
    # Running multiple iterations ensures all kernels are fully compiled and cached.
    logger.info("Warming CUDA...")
    for _ in range(3):
        normalize_using_cupy(dummy_images)
    cp.cuda.Stream.null.synchronize()  # Ensure warm-up completes
    logger.info("Warm-up complete. Starting benchmark...")

    # Uncomment the following line if you need to validate GPU and CPU results.
    # validate_cpu_and_gpu_results(dummy_images)

    logger.info(f"Benchmark Configuration:")
    logger.info(f"  Number of images: {NUM_IMAGES}")
    logger.info(f"  Image size: {IMAGE_SIZE}")
    logger.info(f"  Iterations: {NUM_ITERATIONS}")

    # ===== Baseline: CPU-only benchmark =====
    # CPU Operation.
    cpu_time_start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        normalize_using_numpy(dummy_images)
    cpu_time = time.perf_counter() - cpu_time_start
    logger.info(f"CPU Normalize | Takes {cpu_time / NUM_ITERATIONS * 1000:.2f}ms")

    # ===== GPU benchmark with CPU memory output =====
    # Use case: GPU preprocessing followed by CPU-based inference
    # Measures GPU computation + GPU->CPU transfer overhead
    gpu_cpu_time_start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        normalize_using_cupy(dummy_images, return_gpu_arrays=False)
    # Block CPU until all GPU operations complete (needed for accurate timing)
    cp.cuda.Stream.null.synchronize()
    gpu_cpu_time = time.perf_counter() - gpu_cpu_time_start
    logger.info(
        f"GPU Normalize + Result in CPU memory | Takes {gpu_cpu_time / NUM_ITERATIONS * 1000:.2f}ms"
    )

    # ===== GPU benchmark with GPU memory output =====
    # Use case: GPU preprocessing followed by GPU-based inference (e.g., TensorRT)
    # Avoids CPU transfer overhead - fastest option for GPU inference pipelines
    gpu_gpu_time_start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        normalize_using_cupy(dummy_images, return_gpu_arrays=True)
    # Block CPU until all GPU operations complete (needed for accurate timing).
    cp.cuda.Stream.null.synchronize()
    gpu_gpu_time = time.perf_counter() - gpu_gpu_time_start
    logger.info(
        f"GPU Normalize + Result in GPU Array | Takes {gpu_gpu_time / NUM_ITERATIONS * 1000:.2f}ms"
    )
