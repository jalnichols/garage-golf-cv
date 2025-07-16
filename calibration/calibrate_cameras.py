#!/usr/bin/env python3
"""
Camera calibration script for multi-camera golf swing analysis
Uses checkerboard pattern to calibrate each camera individually
"""

import cv2
import numpy as np
import os
import glob
import sys
sys.path.append('..')

from config.settings import (
    CHECKERBOARD_SIZE, SQUARE_SIZE, CAMERA_COUNT,
    CALIBRATION_DATA_PATH, CHECKERBOARD_IMAGES_PATH,
    RECORDING_WIDTH, RECORDING_HEIGHT
)
from utils.camera_utils import initialize_cameras, release_cameras

def capture_checkerboard_images():
    """Capture checkerboard images from all cameras for calibration"""
    print("Starting checkerboard image capture...")
    print("Press 'c' to capture images from all cameras")
    print("Press 'q' to quit")
    
    # Initialize cameras
    cameras = initialize_cameras(RECORDING_WIDTH, RECORDING_HEIGHT, 30)
    
    if len(cameras) == 0:
        print("No cameras detected!")
        return False
    
    # Create directories for each camera
    for i in range(len(cameras)):
        camera_dir = os.path.join(CHECKERBOARD_IMAGES_PATH, f'camera_{i}')
        os.makedirs(camera_dir, exist_ok=True)
    
    capture_count = 0
    
    while True:
        # Capture frames from all cameras
        frames = []
        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret:
                # Resize for display
                display_frame = cv2.resize(frame, (640, 480))
                cv2.putText(display_frame, f"Camera {i}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append((frame, display_frame))
                cv2.imshow(f'Camera {i}', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Capture and save images
            print(f"Capturing image set {capture_count + 1}...")
            for i, (full_frame, _) in enumerate(frames):
                filename = os.path.join(CHECKERBOARD_IMAGES_PATH, f'camera_{i}', 
                                      f'checkerboard_{capture_count:03d}.jpg')
                cv2.imwrite(filename, full_frame)
                print(f"  Saved: {filename}")
            capture_count += 1
            
        elif key == ord('q'):
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    release_cameras(cameras)
    
    print(f"Captured {capture_count} image sets for calibration")
    return capture_count > 0

def calibrate_single_camera(camera_id):
    """Calibrate a single camera using captured checkerboard images"""
    print(f"Calibrating camera {camera_id}...")
    
    # Prepare object points
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale by actual square size in mm
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Load images for this camera
    camera_dir = os.path.join(CHECKERBOARD_IMAGES_PATH, f'camera_{camera_id}')
    images = glob.glob(os.path.join(camera_dir, '*.jpg'))
    
    if len(images) == 0:
        print(f"No images found for camera {camera_id}")
        return False
    
    print(f"Processing {len(images)} images...")
    
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            print(f"  ✓ Found corners in {os.path.basename(img_path)}")
        else:
            print(f"  ✗ No corners found in {os.path.basename(img_path)}")
    
    if len(objpoints) < 10:
        print(f"Not enough valid images for camera {camera_id} (need at least 10, got {len(objpoints)})")
        return False
    
    # Perform camera calibration
    print(f"Performing calibration with {len(objpoints)} valid images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    if ret:
        # Save calibration data
        os.makedirs(CALIBRATION_DATA_PATH, exist_ok=True)
        np.save(os.path.join(CALIBRATION_DATA_PATH, f'camera_{camera_id}_mtx.npy'), mtx)
        np.save(os.path.join(CALIBRATION_DATA_PATH, f'camera_{camera_id}_dist.npy'), dist)
        
        # Calculate and display reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(objpoints)
        print(f"Camera {camera_id} calibration complete!")
        print(f"  Reprojection error: {mean_error:.3f} pixels")
        print(f"  Camera matrix saved to: {CALIBRATION_DATA_PATH}/camera_{camera_id}_mtx.npy")
        print(f"  Distortion coeffs saved to: {CALIBRATION_DATA_PATH}/camera_{camera_id}_dist.npy")
        
        return True
    else:
        print(f"Calibration failed for camera {camera_id}")
        return False

def main():
    """Main calibration workflow"""
    print("=== Golf Swing Camera Calibration ===")
    print(f"Checkerboard pattern: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} internal corners")
    print(f"Square size: {SQUARE_SIZE}mm")
    print()
    
    # Step 1: Capture checkerboard images
    choice = input("Capture new checkerboard images? (y/n): ").lower()
    if choice == 'y':
        if not capture_checkerboard_images():
            print("Image capture failed or cancelled")
            return
    
    # Step 2: Calibrate each camera
    print("\nStarting camera calibration...")
    
    # Find available camera directories
    camera_dirs = []
    if os.path.exists(CHECKERBOARD_IMAGES_PATH):
        for item in os.listdir(CHECKERBOARD_IMAGES_PATH):
            if item.startswith('camera_') and os.path.isdir(os.path.join(CHECKERBOARD_IMAGES_PATH, item)):
                camera_id = int(item.split('_')[1])
                camera_dirs.append(camera_id)
    
    camera_dirs.sort()
    
    if not camera_dirs:
        print("No camera image directories found!")
        return
    
    # Calibrate each camera
    successful_calibrations = 0
    for camera_id in camera_dirs:
        if calibrate_single_camera(camera_id):
            successful_calibrations += 1
    
    print(f"\nCalibration complete: {successful_calibrations}/{len(camera_dirs)} cameras calibrated successfully")
    
    if successful_calibrations == len(camera_dirs):
        print("All cameras calibrated! Ready for recording.")
    else:
        print("Some cameras failed calibration. Check image quality and try again.")

if __name__ == "__main__":
    main()