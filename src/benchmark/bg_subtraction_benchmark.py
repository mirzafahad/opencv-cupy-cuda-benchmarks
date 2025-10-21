"""
Background Subtraction Benchmark Module.

This module benchmarks CPU vs GPU performance for background subtraction using OpenCV's
MOG (Mixture of Gaussians) algorithm. It compares standard OpenCV (CPU) implementation
against OpenCV CUDA (GPU) implementation on both static images and video files.
"""

import time
from typing import TYPE_CHECKING

import cv2

from benchmark.utils.logger import Logger

if TYPE_CHECKING:
    from cv2.cuda import GpuMat, Stream


logger = Logger()

# Learning rate for background subtraction model updates.
# Lower values (0.0-1.0) make the model adapt more slowly to changes.
LEARNING_RATE = 0.1


def run_profiling(image_file: str, video_file: str, static_image_iteration: int) -> None:
    """
    Profiling CPU and GPU based background subtraction.

    Benchmarks background subtraction performance using both CPU (OpenCV) and GPU (CUDA)
    implementations. Tests are performed on both a static image (repeated iterations) and
    a video file. Results are logged showing timing comparisons and speedup metrics.

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
    image = cv2.imread(image_file)
    if image is None:
        logger.error(f"Can't open/read file: {image_file}. Check file path. Aborting...")
        return

    #################### CPU ####################
    logger.info("Running CPU Background Subtraction...")
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    # Benchmark static image processing.
    start_time_cpu_image = time.perf_counter()
    for _ in range(static_image_iteration):
        bg_subtractor.apply(image, learningRate=LEARNING_RATE)

    time_cpu_image = time.perf_counter() - start_time_cpu_image

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_file}. Aborting...")
        return

    try:
        # Create fresh background model for video test (prevents static image from affecting results).
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

        start_time_cpu_video = time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug("End of video.")
                break
            bg_subtractor.apply(frame, learningRate=LEARNING_RATE)

        time_cpu_video = time.perf_counter() - start_time_cpu_video
    finally:
        cap.release()

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
    start_time_gpu_image = time.perf_counter()
    # Benchmark static image processing with GPU.
    # Note: All operations use the same stream for asynchronous execution.
    for _ in range(static_image_iteration):
        gpu_frame.upload(image, stream=stream)
        gpu_foreground_mask = bg_subtractor.apply(gpu_frame, learningRate=LEARNING_RATE, stream=stream)
        # Download result to simulate real-world usage where CPU needs the output.
        cpu_fg_mask = gpu_foreground_mask.download(stream=stream)
    # Synchronize stream to ensure all GPU operations complete before measuring time.
    # Without this, timing would only measure kernel launch overhead, not actual execution.
    stream.waitForCompletion()
    time_gpu_image = time.perf_counter() - start_time_gpu_image

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_file}. Aborting...")
        return

    try:
        # Create fresh background model for video test (prevents static image from affecting results).
        bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG()

        start_time_gpu_video = time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug("End of video.")
                break
            gpu_frame.upload(frame, stream=stream)
            gpu_foreground_mask = bg_subtractor.apply(gpu_frame, learningRate=LEARNING_RATE, stream=stream)
            cpu_fg_mask = gpu_foreground_mask.download(stream=stream)
        # Synchronize stream to ensure all GPU operations complete before measuring time.
        stream.waitForCompletion()
        time_gpu_video = time.perf_counter() - start_time_gpu_video
    finally:
        cap.release()

    logger.debug(f"CPU: Static image for {static_image_iteration} iterations - {time_cpu_image}")
    logger.debug(f"GPU: Static image for {static_image_iteration} iterations - {time_gpu_image}")
    logger.info(
        f"OpenCV-CUDA was ~{round(time_cpu_image / time_gpu_image)} times faster i.e "
        f"~{round(((time_cpu_image - time_gpu_image) / time_cpu_image) * 100)}% reduction in time."
    )

    logger.debug(f"CPU: Video - {time_cpu_video}")
    logger.debug(f"GPU: Video - {time_gpu_video}")
    logger.info(
        f"OpenCV-CUDA was ~{round(time_cpu_video / time_gpu_video)} times faster i.e. "
        f"~{round(((time_cpu_video - time_gpu_video) / time_cpu_video) * 100)}% reduction in time."
    )


if __name__ == "__main__":
    image_file_path = "./resource/background.jpg"
    video_file_path = "./resource/demo_video.mp4"

    run_profiling(image_file_path, video_file_path, static_image_iteration=150)
