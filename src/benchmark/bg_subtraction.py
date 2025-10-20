import time
import cv2
from benchmark.utils.logger import Logger
from numpy.typing import NDArray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cv2.cuda import GpuMat, Stream


logger = Logger()


def run_gpu_bg_subtraction(image: NDArray):
    """
    Run opencv-cuda background subtraction.
    """
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG()
        stream: Stream = cv2.cuda_Stream()
        gpu_frame: GpuMat = cv2.cuda_GpuMat()
        logger.info(f"Using GPU OpenCV with {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)")
    else:
        logger.error("OpenCV-CUDA not available!")
        return
    
    # Now run the subtractor for 100 times.
    for _ in range(100):
        # Upload to GPU.
        gpu_frame.upload(image)
        # Get a mask from the subtraction
        gpu_foreground_mask = bg_subtractor.apply(gpu_frame, learningRate=0.1, stream=stream)
        # Download the result from the GPU to CPU.
        cpu_foreground_mask: NDArray = gpu_foreground_mask.download()


def main():
    """
    Profiling CPU based background subtraction.
    """
    
    logger.info("CPU Background Subtraction")

    # Load an image.
    image = cv2.imread("./background.jpg")
    if image is None:
        logger.error(f"Can't open/read file. Check file path. Aborting...")
        return
    
    cap = cv2.VideoCapture("./color.mp4")
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()
    


    # Initialize the cpu based background subtractor.
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    # Now run the subtractor for 100 times.
    start_time_cpu = time.perf_counter()
    for _ in range(100):
        foreground_mask: NDArray = bg_subtractor.apply(image, learningRate=0.1)
    logger.info(f"CPU: Time taken for 100 iterations - {time.perf_counter() - start_time_cpu}")
    
    
    start_time_cpu = time.perf_counter()
    while True:
        ret, frame = cap.read()
    
        # If frame reading was successful
        if not ret:
            print("End of video or error reading frame")
            break
        foreground_mask: NDArray = bg_subtractor.apply(frame, learningRate=0.1)
    
    logger.info(f"CPU: Time taken for the video - {time.perf_counter() - start_time_cpu}")

    start_time_gpu = time.perf_counter()
    run_gpu_bg_subtraction(image)
    logger.info(f"GPU: Time taken for 100 iterations - {time.perf_counter() - start_time_gpu}")
    
    
    cap = cv2.VideoCapture("./color.mp4")
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG()
        stream: Stream = cv2.cuda_Stream()
        gpu_frame: GpuMat = cv2.cuda_GpuMat()
        logger.info(f"Using GPU OpenCV with {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)")
    else:
        logger.error("OpenCV-CUDA not available!")
        return
        
    start_time_gpu = time.perf_counter()
    while True:
        ret, frame = cap.read()
    
        # If frame reading was successful
        if not ret:
            print("End of video or error reading frame")
            break
        gpu_frame.upload(frame)
        # Get a mask from the subtraction
        gpu_foreground_mask = bg_subtractor.apply(gpu_frame, learningRate=0.1, stream=stream)
        # Download the result from the GPU to CPU.
        cpu_foreground_mask: NDArray = gpu_foreground_mask.download()
    logger.info(f"GPU: Time taken for video - {time.perf_counter() - start_time_gpu}")


if __name__ == "__main__":
    main()

