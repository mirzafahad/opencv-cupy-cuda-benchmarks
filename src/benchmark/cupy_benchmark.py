"""
Benchmarking NumPy vs CuPy.
"""

import time

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from benchmark.utils.logger import Logger


logger = Logger()


# Normalization parameters (ImageNet stats)
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
MEAN_GPU = cp.array(MEAN, dtype=cp.float32)
STD_GPU = cp.array(STD, dtype=cp.float32)
MEAN_CPU = np.array(MEAN, dtype=np.float32)
STD_CPU = np.array(STD, dtype=np.float32)


def normalize_using_cupy(images: list[NDArray], return_gpu_arrays: bool = False):
    """
    GPU normalization using CuPy.

    Args:
        images: Images to normalize.
        return_gpu_arrays: If True, keep the end result on GPU array. If False, download to CPU memory.

    Returns:
        GPU or CPU array depending on return_gpu_arrays.
    """
    # Stack all images into single batch (N, H, W, C).
    images_batch = np.stack(images)
    # Upload to GPU.
    images_batch_gpu = cp.asarray(images_batch)

    # Normalize all images in parallel.
    normalized_batch_gpu = (images_batch_gpu - MEAN_GPU) / STD_GPU

    # Batch transpose: (N, H, W, C) -> (N, C, H, W)
    transposed_batch_gpu = cp.transpose(normalized_batch_gpu, (0, 3, 1, 2))

    if return_gpu_arrays:
        # Keep on GPU - no download overhead!
        # Extract each images and add batch dimension: (C, H, W) -> (1, C, H, W)
        results_gpu = [
            cp.expand_dims(transposed_batch_gpu[i], axis=0)  # (1, 3, 640, 640) per image
            for i in range(len(images))
        ]

        return results_gpu
    else:
        # Download to CPU.
        batch_cpu = cp.asnumpy(transposed_batch_gpu)
        # Extract each array and add batch dimension: (C, H, W) -> (1, C, H, W)
        results_cpu = [
            np.expand_dims(batch_cpu[i], axis=0)  # (1, 3, 640, 640) per image.
            for i in range(len(images))
        ]

        return results_cpu


def normalize_using_numpy(images: list[NDArray]):
    """
    CPU normalization using numpy.

    Args:
        images: Images to normalize.

    Returns:
        Normalized CPU arrays.
    """
    # Stack all images into single batch (N, H, W, C).
    images_batch = np.stack(images)

    # Normalize all images in parallel.
    normalized_batch = (images_batch - MEAN_CPU) / STD_CPU

    # Batch transpose: (N, H, W, C) -> (N, C, H, W)
    transposed_batch = np.transpose(normalized_batch, (0, 3, 1, 2))

    # Extract each array and add batch dimension: (C, H, W) -> (1, C, H, W)
    results = [
        np.expand_dims(transposed_batch[i], axis=0)  # (1, 3, 640, 640) per image
        for i in range(len(images))
    ]

    return results

def validate_cpu_and_gpu_results(images: list[NDArray]):
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
    NUM_IMAGES = 6
    # Let's assume our inference model takes 640-by-640 RGB images.
    IMAGE_SIZE = (640, 640, 3)
    NUM_ITERATIONS = 20

    dummy_images = []
    for _ in range(NUM_IMAGES):
        dummy_images.append(np.random.randint(0, 256, size=IMAGE_SIZE, dtype=np.uint8))



    # Note: First GPU operation initializes the CUDA context and CuPy compiles CUDA kernels on first use.
    # Due to that, I am running a warm-up iteration before the actual benchmark.
    logger.info("Warming CUDA...")
    normalize_using_cupy(dummy_images)
    logger.info("Warm-up complete. Starting actual benchmark...")

    # CPU Operation.
    cpu_time_start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        normalize_using_numpy(dummy_images)
    cpu_time = time.perf_counter() - cpu_time_start
    logger.info(f"CPU Normalize | Takes {cpu_time / NUM_ITERATIONS * 1000:.2f}ms")

    # GPU operation
    # Download the end result to CPU.
    gpu_cpu_time_start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        normalize_using_cupy(dummy_images, return_gpu_arrays=False)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU operations to complete.
    gpu_cpu_time = time.perf_counter() - gpu_cpu_time_start
    logger.info(f"GPU Normalize + Result in CPU memory | Takes {gpu_cpu_time / NUM_ITERATIONS * 1000:.2f}ms")

    # Keep the end result in GPU.
    gpu_gpu_time_start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        normalize_using_cupy(dummy_images, return_gpu_arrays=True)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU operations to complete.
    gpu_gpu_time = time.perf_counter() - gpu_gpu_time_start
    logger.info(f"GPU Normalize + Result in GPU Array | Takes {gpu_gpu_time / NUM_ITERATIONS * 1000:.2f}ms")

    # Uncomment the following line if you need to validate GPU and CPU results.
    # validate_cpu_and_gpu_results(dummy_images)