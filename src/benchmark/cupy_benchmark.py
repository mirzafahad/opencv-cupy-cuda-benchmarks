import time
import cv2
import cupy as cp
import numpy as np
from numpy.typing import NDArray
from benchmark.utils.logger import Logger


logger = Logger()


def normalize_using_cupy(images: list[NDArray], return_gpu_arrays: bool = False):
    """
    GPU normalization using CuPy.

    Args:
        images: Images to normalize.
        return_gpu_arrays: If True, keep end result on GPU array. If False, download to CPU memory.

    Returns:
        GPU or CPU array depending on return_gpu_arrays.
    """
    # Stack all images into single batch (N, H, W, C)
    images_batch = np.stack(images)
    images_batch_gpu = cp.asarray(images_batch)

    mean_gpu = cp.array([123.675, 116.28, 103.53], dtype=cp.float16)
    std_gpu = cp.array([58.395, 57.12, 57.375], dtype=cp.float16)

    # Normalize all images in parallel.
    normalized_batch_gpu = (images_batch_gpu - mean_gpu) / std_gpu

    # Batch transpose: (N, H, W, C) -> (N, C, H, W)
    transposed_batch_gpu = cp.transpose(normalized_batch_gpu, (0, 3, 1, 2))

    if return_gpu_arrays:
        # Keep on GPU - no download overhead!
        # Extract each images and add batch dimension: (C, H, W) -> (1, C, H, W)
        results_gpu = [
            cp.expand_dims(transposed_batch_gpu[i], axis=0)  # (1, 3, 640, 640) per camera
            for i in range(len(images))
        ]

        return results_gpu
    else:
        # Download to CPU.
        batch_cpu = cp.asnumpy(transposed_batch_gpu)
        # Extract each camera and add batch dimension: (C, H, W) -> (1, C, H, W)
        results = [
            np.expand_dims(batch_cpu[i], axis=0)  # (1, 3, 640, 640) per image
            for i in range(len(images))
        ]

        return results


def normalize_using_numpy(images: list):
    """
    CPU normalization using numpy.

    Args:
        images: Images to normalize.

    Returns:
        Normalized CPU arrays.
    """
    images_batch = np.stack(images)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float16)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float16)

    # Normalize all images in parallel.
    normalized_batch = (images_batch - mean) / std

    # Batch transpose: (N, H, W, C) -> (N, C, H, W)
    transposed_batch = np.transpose(normalized_batch, (0, 3, 1, 2))

    # Extract each camera and add batch dimension: (C, H, W) -> (1, C, H, W)
    results = [
        np.expand_dims(transposed_batch[i], axis=0)  # (1, 3, 640, 640) per image
        for i in range(len(images))
    ]
    
    return results


if __name__ == "__main__":
    # Let's assume our inference model takes 640-by-640 images.
    dummy_images = []
    for _ in range(6):
        dummy_images.append(np.random.randint(0, 256, size=(640, 640, 3), dtype=np.uint8))
    
    # Note: First GPU operation initializes the CUDA context and CuPy compiles CUDA kernels on first use.
    # Due to that, I am running a warm-up iteration before the actual benchmark.
    logger.info("Warming CUDA...")
    normalize_using_cupy(dummy_images)
    logger.info("Warm-up complete. Starting actual benchmark...")
    
    # CPU Operation.
    batch_start = time.perf_counter()
    for _ in range(20):
        normalize_using_numpy(dummy_images)
    
    total_time = time.perf_counter() - batch_start
    logger.info(
        f"CPU Normalize | " f"Takes {total_time / 20 * 1000:.2f}ms"
    )
    
    # GPU operation
    # Download the end result to CPU.
    batch_start = time.perf_counter()
    for _ in range(20):
        normalize_using_cupy(dummy_images, return_gpu_arrays=False)
    
    total_time = time.perf_counter() - batch_start
    logger.info(
        f"GPU Normalize + Result in CPU memory | "
        f"Takes {total_time / 20 * 1000:.2f}ms | "
    )
    
    # Keep the end result in GPU.
    batch_start = time.perf_counter()
    for _ in range(20):
        normalize_using_cupy(dummy_images, return_gpu_arrays=True)
        total_time = time.perf_counter() - batch_start
    logger.info(
        f"GPU Normalize + Result in GPU Array | "
        f"Takes {total_time / 20 * 1000:.2f}ms"
    )

