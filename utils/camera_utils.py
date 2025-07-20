"""
Utility functions for camera operations
"""
import cv2
import numpy as np
import os
from config.settings import CAMERA_COUNT

def initialize_cameras(width=1920, height=1080, fps=60):
    """Initialize all three cameras with specified settings"""
    cameras = []
    
    for i in range(CAMERA_COUNT):
        # Use DirectShow backend for Windows
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Warning: Camera {i} could not be opened")
            continue
            
        # Set camera properties with optimizations
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Force MJPEG format
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # Minimize latency with buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify settings
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera {i}: {actual_width}x{actual_height} @ {actual_fps}fps")
        cameras.append(cap)
    
    return cameras

def release_cameras(cameras):
    """Safely release all camera resources"""
    for i, cap in enumerate(cameras):
        if cap is not None:
            cap.release()
            print(f"Camera {i} released")

def capture_synchronized_frames(cameras):
    """Capture frames from all cameras simultaneously"""
    frames = []
    timestamps = []
    
    for cap in cameras:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            timestamps.append(cv2.getTickCount())
        else:
            frames.append(None)
            timestamps.append(None)
    
    return frames, timestamps

def load_camera_calibration(camera_id, calibration_path):
    """Load calibration data for a specific camera"""
    mtx_file = os.path.join(calibration_path, f'camera_{camera_id}_mtx.npy')
    dist_file = os.path.join(calibration_path, f'camera_{camera_id}_dist.npy')
    
    if os.path.exists(mtx_file) and os.path.exists(dist_file):
        mtx = np.load(mtx_file)
        dist = np.load(dist_file)
        return mtx, dist
    else:
        print(f"Calibration data not found for camera {camera_id}")
        return None, None

def undistort_frame(frame, mtx, dist):
    """Apply distortion correction to a frame"""
    if mtx is not None and dist is not None:
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        return undistorted
    return frame