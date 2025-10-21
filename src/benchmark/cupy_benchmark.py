from benchmark.utils.logger import Logger
from numpy.typing import NDArray
import cv2
import numpuy as np
import cupy as cp

logger = Logger()


def preprocess(image: NDArray) -> list[NDArray]:
    """
    Preprocess the image for the model.
    """
    processed_images = []
    for _ in range(6):
        resized_image = cv2.resize(
            image,
            (640, 360),
            interpolation=cv2.INTER_LINEAR,
        )

        # Format: ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
        padding = ((0, 280), (0, 0), (0, 0))

        padded_image = np.pad(
            resized_image,
            padding,
            mode="constant",
            constant_values=114.0,
        )
        processed_images.append(padded_image)

    normalized_images = normalize_in_batch(processed_images)
    return normalized_images


def normalize_in_batch(images: list[NDArray], gpu: bool = False):
    if gpu:
        return normalize_batch_gpu(images, return_gpu_arrays)
    else:
        return normalize_batch_cpu(images)


def normalize_batch_gpu(images: list[NDArray], return_gpu_arrays: bool = False):
    """
    Batch GPU normalization for multiple images.

    Args:
        images: Images to normalize.
        return_gpu_arrays: If True, keep on GPU array. If False, download to CPU.

    Returns:
        GPU or CPU array depending on return_gpu_arrays.
    """
    batch_start = time.perf_counter()

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
        results = [
            cp.expand_dims(transposed_batch_gpu[i], axis=0)  # (1, 3, 640, 640) per camera
            for i in range(len(images))
        ]
        total_time = time.perf_counter() - batch_start
        logger.info(
            f"GPU Batch Normalize + GPU Array [{len(images)} images] | "
            f"Total: {total_time * 1000:.2f}ms"
        )
        return results
    else:
        # Download to CPU.
        batch_cpu = cp.asnumpy(transposed_batch_gpu)

        # Extract each camera and add batch dimension: (C, H, W) -> (1, C, H, W)
        results = [
            np.expand_dims(batch_cpu[i], axis=0)  # (1, 3, 640, 640) per image
            for i in range(num_cameras)
        ]
        total_time = time.perf_counter() - batch_start
        logger.info(
            f"GPU Batch Normalize + Download to CPU [{len(images)} cameras] | "
            f"Total: {total_time * 1000:.2f}ms"
        )

        return results


def normalize_batch_cpu(images: list):
    """
    CPU batch normalization.

    Args:
        images: Images to normalize.

    Returns:
        Normalized CPU arrays.
    """
    batch_start = time.perf_counter()

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
    total_time = time.perf_counter() - batch_start
    logger.info(
        f"CPU Batch Normalize [{len(images)} cameras] | " f"Total: {total_time * 1000:.2f}ms"
    )

    return results
