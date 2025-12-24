# config.py

# Camera Settings
CAMERA_SOURCE = 0  # Use 0 for webcam, or a path to a video file: "videos/crowd_sample.mp4"
FRAME_WIDTH = 640
FRAME_HEIGHT = 640  # Square aspect ratio for YOLO optimization later
FPS_LIMIT = 30

# Privacy Filter Settings
ENABLE_BLUR = True
BLUR_INTENSITY = (99, 99)  # Kernel size for Gaussian Blur (Must be odd numbers)
DETECTION_CONFIDENCE = 0.5 # MediaPipe confidence threshold (0.0 to 1.0)