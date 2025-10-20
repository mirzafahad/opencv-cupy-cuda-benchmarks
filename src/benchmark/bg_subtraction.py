import time
import cv2
from benchmark.utils.logger import Logger
from numpy.typing import NDArray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cv2.cuda import GpuMat, Stream


logger = Logger()


def run_profiling(image_file: str, video_file: str):
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

    # Now run the subtractor using the same image for 100 iterations.
    start_time_cpu_image = time.perf_counter()
    for _ in range(100):
        foreground_mask: NDArray = bg_subtractor.apply(image, learningRate=0.1)
    
    time_cpu_image = time.perf_counter() - start_time_cpu_image
       
    # Read frames from a video file.
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_file}. Aborting...")
        return
    
    # Initialize the background subtractor again.
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    start_time_cpu_video = time.perf_counter()
    while True:
        ret, frame = cap.read()
    
        # Check if frame reading was successful.
        if not ret:
            logger.debug("End of video.")
            break
        foreground_mask: NDArray = bg_subtractor.apply(frame, learningRate=0.1)
    
    time_cpu_video = time.perf_counter() - start_time_cpu_video
    

    #################### GPU ####################
    # First check if OpenCV-CUDA is available.
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG()
        stream: Stream = cv2.cuda_Stream()
        gpu_frame: GpuMat = cv2.cuda_GpuMat()
        logger.info(f"Using GPU OpenCV with {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)")
    else:
        logger.error("OpenCV-CUDA not available!")
        return
    
    logger.info("Running GPU Background Subtraction...")
    start_time_gpu_image = time.perf_counter()
    # Now run the subtractor for 100 times.
    for _ in range(100):
        # Upload to GPU.
        gpu_frame.upload(image)
        # Get a mask from the subtraction.
        gpu_foreground_mask = bg_subtractor.apply(gpu_frame, learningRate=0.1, stream=stream)
        # Download the result from the GPU to CPU.
        cpu_foreground_mask: NDArray = gpu_foreground_mask.download()
    time_gpu_image = time.perf_counter() - start_time_gpu_image
    
    
    # Now run the subtractor for the video.
    cap = cv2.VideoCapture(video_file)
    # Create the background subtraction again.
    bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG()
        
    start_time_gpu_video = time.perf_counter()
    while True:
        ret, frame = cap.read()
    
        # Check if frame reading was successful.
        if not ret:
            logger.debug("End of video.")
            break
        gpu_frame.upload(frame)
        # Get a mask from the subtraction
        gpu_foreground_mask = bg_subtractor.apply(gpu_frame, learningRate=0.1, stream=stream)
        # Download the result from the GPU to CPU.
        cpu_foreground_mask: NDArray = gpu_foreground_mask.download()
    time_gpu_video = time.perf_counter() - start_time_gpu_video
    

    logger.info(f"CPU: Static image for 100 iterations - {time_cpu_image}")
    logger.info(f"GPU: Static image for 100 iterations - {time_gpu_image}")
    logger.info(f"OpenCV-CUDA was {int(time_cpu_image / time_gpu_image)} times faster. That is {int((time_cpu_image - time_gpu_image) / time_cpu_image) * 100}% reduction.")
    
    logger.info(f"CPU: Video - {time_cpu_video}")
    logger.info(f"GPU: Video - {time_gpu_video}")
    logger.info(f"OpenCV-CUDA was {int(time_cpu_video / time_gpu_video)} times faster. That is {int((time_cpu_video - time_gpu_video) / time_cpu_video) * 100}% reduction.")


if __name__ == "__main__":
    image_file_path = "./background.jpg"
    video_file_path = "./demo_video.mp4"

    run_profiling(image_file_path, video_file_path)

