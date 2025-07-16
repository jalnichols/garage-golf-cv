#!/usr/bin/env python3
"""
Live preview system for multi-camera golf swing setup
Shows low-latency feeds from all three cameras for positioning
"""

import cv2
import numpy as np
import sys
import time
sys.path.append('..')

from config.settings import (
    CAMERA_COUNT, PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_FPS,
    CALIBRATION_DATA_PATH
)
from utils.camera_utils import (
    initialize_cameras, release_cameras, capture_synchronized_frames,
    load_camera_calibration, undistort_frame
)

class LivePreview:
    def __init__(self, use_calibration=True):
        """Initialize live preview system"""
        self.cameras = []
        self.calibration_data = []
        self.use_calibration = use_calibration
        self.running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def initialize(self):
        """Initialize cameras and load calibration data"""
        print("Initializing cameras for live preview...")
        
        # Initialize cameras with preview settings
        self.cameras = initialize_cameras(PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_FPS)
        
        if len(self.cameras) == 0:
            print("No cameras detected!")
            return False
        
        print(f"Initialized {len(self.cameras)} cameras")
        
        # Load calibration data if requested
        if self.use_calibration:
            print("Loading calibration data...")
            for i in range(len(self.cameras)):
                mtx, dist = load_camera_calibration(i, CALIBRATION_DATA_PATH)
                self.calibration_data.append((mtx, dist))
                if mtx is not None:
                    print(f"  Camera {i}: Calibration loaded")
                else:
                    print(f"  Camera {i}: No calibration data found")
        
        return True
    
    def create_combined_display(self, frames):
        """Create a combined display showing all camera feeds"""
        valid_frames = [f for f in frames if f is not None]
        
        if len(valid_frames) == 0:
            return None
        
        # Resize frames to consistent size for display
        display_frames = []
        for i, frame in enumerate(frames):
            if frame is not None:
                # Apply calibration if available
                if self.use_calibration and i < len(self.calibration_data):
                    mtx, dist = self.calibration_data[i]
                    frame = undistort_frame(frame, mtx, dist)
                
                # Add camera label
                labeled_frame = frame.copy()
                cv2.putText(labeled_frame, f"Camera {i}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add calibration status
                if self.use_calibration and i < len(self.calibration_data):
                    mtx, _ = self.calibration_data[i]
                    status = "Calibrated" if mtx is not None else "Uncalibrated"
                    color = (0, 255, 0) if mtx is not None else (0, 0, 255)
                    cv2.putText(labeled_frame, status, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                display_frames.append(labeled_frame)
            else:
                # Create placeholder for missing camera
                placeholder = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Camera {i} Offline", (50, PREVIEW_HEIGHT//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                display_frames.append(placeholder)
        
        # Arrange cameras in a grid
        if len(display_frames) == 1:
            combined = display_frames[0]
        elif len(display_frames) == 2:
            combined = np.hstack(display_frames)
        else:  # 3 or more cameras
            # Top row: first two cameras
            top_row = np.hstack(display_frames[:2])
            # Bottom row: third camera centered (pad with black if needed)
            if len(display_frames) >= 3:
                bottom_frame = display_frames[2]
                # Center the third camera
                padding = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH//2, 3), dtype=np.uint8)
                bottom_row = np.hstack([padding, bottom_frame, padding])
            else:
                bottom_row = np.zeros((PREVIEW_HEIGHT, top_row.shape[1], 3), dtype=np.uint8)
            
            combined = np.vstack([top_row, bottom_row])
        
        # Add FPS counter
        cv2.putText(combined, f"FPS: {self.current_fps:.1f}", 
                   (combined.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def run(self):
        """Main preview loop"""
        if not self.initialize():
            return
        
        print("\n=== Live Preview ===")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Toggle calibration on/off")
        print("  'f' - Show FPS info")
        print("  'r' - Reset FPS counter")
        print()
        
        self.running = True
        self.fps_start_time = time.time()
        
        try:
            while self.running:
                # Capture frames from all cameras
                frames, timestamps = capture_synchronized_frames(self.cameras)
                
                # Create combined display
                combined_display = self.create_combined_display(frames)
                
                if combined_display is not None:
                    cv2.imshow('Golf Swing Camera Preview', combined_display)
                
                # Update FPS
                self.update_fps()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting preview...")
                    break
                elif key == ord('c'):
                    self.use_calibration = not self.use_calibration
                    print(f"Calibration: {'ON' if self.use_calibration else 'OFF'}")
                elif key == ord('f'):
                    print(f"Current FPS: {self.current_fps:.2f}")
                elif key == ord('r'):
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                    print("FPS counter reset")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        cv2.destroyAllWindows()
        release_cameras(self.cameras)
        self.running = False

def main():
    """Main function"""
    print("=== Golf Swing Live Preview ===")
    
    # Ask user about calibration
    use_cal = input("Use camera calibration? (y/n, default=y): ").lower()
    use_calibration = use_cal != 'n'
    
    # Create and run preview
    preview = LivePreview(use_calibration=use_calibration)
    preview.run()

if __name__ == "__main__":
    main()