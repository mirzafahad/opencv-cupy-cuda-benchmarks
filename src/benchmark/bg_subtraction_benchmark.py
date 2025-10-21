"""
Background Subtraction Benchmark Module.

This module benchmarks CPU vs GPU performance for background subtraction using OpenCV's
MOG (Mixture of Gaussians) algorithm. It compares standard OpenCV (CPU) implementation
against OpenCV CUDA (GPU) implementation on both static images and video files.
"""

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from benchmark.utils.logger import Logger

if TYPE_CHECKING:
    from cv2.cuda import GpuMat, Stream
    from numpy.typing import NDArray


logger = Logger()

# Learning rate for background subtraction model updates.
# Lower values (0.0-1.0) make the model adapt more slowly to changes.
LEARNING_RATE = 0.1

# Number of warmup iterations to run before benchmarking GPU operations.
# GPU kernels are typically slower on the first run due to JIT compilation and initialization.
GPU_WARMUP_ITERATIONS = 10


def run_profiling(image_file: str, video_file: str, static_image_iteration: int) -> None:
    """
    Profiling CPU and GPU based background subtraction.

    Benchmarks background subtraction performance using both CPU (OpenCV) and GPU (CUDA)
    implementations. Tests are performed on both a static image (repeated iterations) and
    a video file. Results are logged showing timing comparisons and speedup metrics.

    Note: GPU timing includes data transfer overhead (upload to GPU and download from GPU).
    This provides a realistic measurement of end-to-end performance for applications that
    need to transfer results back to CPU memory. For pure algorithm comparison without I/O,
    the transfer operations would need to be measured separately.

    Args:
        image_file: Path to the static image file for testing.
        video_file: Path to the video file for testing.
        static_image_iteration: Number of iterations to run on the static image.

    Returns:
        None. Results are logged to the console via the logger.

    Raises:
        Early return if image/video files cannot be opened or CUDA is not available.
    """
    # Load a static image. Will use it for both CPU and GPU based background subtraction.
    static_image = cv2.imread(image_file)
    if static_image is None:
        logger.error(f"Can't open/read file: {image_file}. Check file path. Aborting...")
        return

    # Preload all video frames into memory to ensure a fair comparison between CPU and GPU.
    # This eliminates I/O overhead variability from the timing measurements.
    logger.info("Pre-loading video frames into memory...")
    video_frames = []
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_file}. Aborting...")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()

    logger.info(f"Loaded {len(video_frames)} frames from video.")

    #################### CPU ####################
    logger.info("Running CPU Background Subtraction...")
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    # Benchmark static image processing.
    cpu_image_start_time = time.perf_counter()
    for _ in range(static_image_iteration):
        bg_subtractor.apply(static_image, learningRate=LEARNING_RATE)
    cpu_image_time = time.perf_counter() - cpu_image_start_time

    # Create a fresh background model for video test (prevents static image from affecting results).
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    cpu_video_start_time = time.perf_counter()
    for frame in video_frames:
        bg_subtractor.apply(frame, learningRate=LEARNING_RATE)
    cpu_video_time = time.perf_counter() - cpu_video_start_time

    #################### GPU ####################
    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG()
        stream: Stream = cv2.cuda.Stream()
        gpu_frame: GpuMat = cv2.cuda.GpuMat()
        logger.debug(f"Using GPU OpenCV with {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)")
    else:
        logger.error("OpenCV-CUDA not available!")
        return

    logger.info("Running GPU Background Subtraction...")

    # Warmup: Run a few iterations to initialize GPU kernels and eliminate JIT compilation overhead.
    logger.debug(f"Warming up GPU with {GPU_WARMUP_ITERATIONS} iterations...")
    for _ in range(GPU_WARMUP_ITERATIONS):
        gpu_frame.upload(static_image, stream=stream)
        gpu_foreground_mask: GpuMat = bg_subtractor.apply(gpu_frame, learningRate=LEARNING_RATE, stream=stream)
        _ = gpu_foreground_mask.download(stream=stream)
    stream.waitForCompletion()

    gpu_image_start_time = time.perf_counter()
    # Benchmark static image processing with GPU.
    # Note: All operations use the same stream for asynchronous execution.
    for _ in range(static_image_iteration):
        gpu_frame.upload(static_image, stream=stream)
        gpu_foreground_mask: GpuMat = bg_subtractor.apply(gpu_frame, learningRate=LEARNING_RATE, stream=stream)
        # Download result to simulate real-world usage where CPU needs the output.
        _ = gpu_foreground_mask.download(stream=stream)
    # Synchronize stream to ensure all GPU operations complete before measuring time.
    # Without this, timing would only measure kernel launch overhead, not actual execution.
    stream.waitForCompletion()
    time_gpu_image = time.perf_counter() - gpu_image_start_time

    # Create a fresh background model for video test (prevents static image from affecting results).
    bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG()

    # Warmup for video benchmark with fresh background subtractor.
    logger.debug(f"Warming up GPU for video test with {GPU_WARMUP_ITERATIONS} iterations...")
    warmup_frames = video_frames[:GPU_WARMUP_ITERATIONS] if len(video_frames) >= GPU_WARMUP_ITERATIONS else video_frames
    for frame in warmup_frames:
        gpu_frame.upload(frame, stream=stream)
        gpu_foreground_mask: GpuMat = bg_subtractor.apply(gpu_frame, learningRate=LEARNING_RATE, stream=stream)
        _ = gpu_foreground_mask.download(stream=stream)
    stream.waitForCompletion()

    gpu_video_start_time = time.perf_counter()
    for frame in video_frames:
        gpu_frame.upload(frame, stream=stream)
        gpu_foreground_mask: GpuMat = bg_subtractor.apply(gpu_frame, learningRate=LEARNING_RATE, stream=stream)
        # Download result to simulate real-world usage where CPU needs the output.
        _ = gpu_foreground_mask.download(stream=stream)
    # Synchronize stream to ensure all GPU operations complete before measuring time.
    stream.waitForCompletion()
    gpu_video_time = time.perf_counter() - gpu_video_start_time

    # Explicitly release GPU resources
    gpu_frame.release()

    logger.info(f"CPU: Static image for {static_image_iteration} iterations - {cpu_image_time}")
    logger.info(f"GPU: Static image for {static_image_iteration} iterations - {time_gpu_image}")
    logger.info(
        f"OpenCV-CUDA was ~{round(cpu_image_time / time_gpu_image)} times faster i.e "
        f"~{round(((cpu_image_time - time_gpu_image) / cpu_image_time) * 100)}% reduction in time."
    )

    logger.info(f"CPU: Video - {cpu_video_time}")
    logger.info(f"GPU: Video - {gpu_video_time}")
    logger.info(
        f"OpenCV-CUDA was ~{round(cpu_video_time / gpu_video_time)} times faster i.e. "
        f"~{round(((cpu_video_time - gpu_video_time) / cpu_video_time) * 100)}% reduction in time."
    )


if __name__ == "__main__":
    image_file_path = "./resource/background.jpg"
    video_file_path = "./resource/demo_video.mp4"

    run_profiling(image_file_path, video_file_path, static_image_iteration=150)
