"""
Configuration settings for garage golf CV system
"""

# Camera settings
CAMERA_COUNT = 3
RECORDING_WIDTH = 1920
RECORDING_HEIGHT = 1080
RECORDING_FPS = 60

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
PREVIEW_FPS = 30

# Calibration settings
CHECKERBOARD_SIZE = (9, 6)  # Internal corners (width, height)
SQUARE_SIZE = 25  # mm

# Voice recognition settings
VOICE_COMMANDS = {
    'start': ['start swing', 'start', 'record', 'go'],
    'stop': ['stop swing', 'stop', 'end', 'done']
}

# Recording settings
PRE_SWING_BUFFER = 2.0  # seconds
POST_SWING_BUFFER = 3.0  # seconds
OUTPUT_FORMAT = 'mp4'

# File paths
CALIBRATION_DATA_PATH = 'calibration/camera_params'
CHECKERBOARD_IMAGES_PATH = 'calibration/checkerboard_images'
OUTPUT_VIDEOS_PATH = 'recording/output_videos'