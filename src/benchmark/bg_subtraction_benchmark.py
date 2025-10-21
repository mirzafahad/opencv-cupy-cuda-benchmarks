import time
from typing import TYPE_CHECKING

import cv2

from benchmark.utils.logger import Logger

if TYPE_CHECKING:
    from cv2.cuda import GpuMat, Stream


logger = Logger()


def run_profiling(image_file: str, video_file: str, static_image_iteration: int) -> None:
    """
    Profiling CPU and GPU based background subtraction.
    """
    # Load a static image. Will use it for both CPU and GPU based background subtraction.
    image = cv2.imread(image_file)
    if image is None:
        logger.error(f"Can't open/read file: {image_file}. Check file path. Aborting...")
        return

    #################### CPU ####################
    logger.info("Running CPU Background Subtraction...")
    # Initialize the cpu based background subtractor.
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    # Now run the subtractor using the same image for multi iterations.
    start_time_cpu_image = time.perf_counter()
    for _ in range(static_image_iteration):
        bg_subtractor.apply(image, learningRate=0.1)

    time_cpu_image = time.perf_counter() - start_time_cpu_image

    # Read frames from a video file.
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_file}. Aborting...")
        return

    try:
        # Initialize the background subtractor again.
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

        start_time_cpu_video = time.perf_counter()
        while True:
            ret, frame = cap.read()

            # Check if frame reading was successful.
            if not ret:
                logger.debug("End of video.")
                break
            bg_subtractor.apply(frame, learningRate=0.1)

        time_cpu_video = time.perf_counter() - start_time_cpu_video
    finally:
        cap.release()

    #################### GPU ####################
    # First check if OpenCV-CUDA is available.
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
    # Now run the subtractor for 100 times.
    for _ in range(static_image_iteration):
        # Upload to GPU.
        gpu_frame.upload(image, stream=stream)
        # Get a mask from the subtraction.
        gpu_foreground_mask = bg_subtractor.apply(gpu_frame, learningRate=0.1, stream=stream)
        # Download the result from the GPU to CPU.
        cpu_fg_mask = gpu_foreground_mask.download(stream=stream)
    # Wait for all GPU operations to complete before stopping the timer.
    stream.waitForCompletion()
    time_gpu_image = time.perf_counter() - start_time_gpu_image

    # Now run the subtractor for the video.
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_file}. Aborting...")
        return

    try:
        # Create the background subtraction again.
        bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG()

        start_time_gpu_video = time.perf_counter()
        while True:
            ret, frame = cap.read()

            # Check if frame reading was successful.
            if not ret:
                logger.debug("End of video.")
                break
            gpu_frame.upload(frame, stream=stream)
            # Get a mask from the subtraction.
            gpu_foreground_mask = bg_subtractor.apply(gpu_frame, learningRate=0.1, stream=stream)
            # Download the result from the GPU to CPU.
            cpu_fg_mask = gpu_foreground_mask.download(stream=stream)
        # Wait for all GPU operations to complete before stopping the timer.
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
