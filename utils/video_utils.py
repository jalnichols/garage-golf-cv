"""
Utility functions for video processing and merging
"""
import cv2
import numpy as np
import os
from datetime import datetime

def create_video_writer(filename, width, height, fps, fourcc='mp4v'):
    """Create a video writer with specified parameters"""
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    return cv2.VideoWriter(filename, fourcc_code, fps, (width, height))

def merge_videos_side_by_side(video_paths, output_path):
    """Merge multiple videos into a side-by-side layout"""
    if len(video_paths) != 3:
        raise ValueError("Expected exactly 3 video paths")
    
    # Open video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    # Get video properties from first video
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    
    # Create output video writer (3x width for side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 3, height))
    
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                # End of video
                break
            frames.append(frame)
        
        if len(frames) != 3:
            break
        
        # Concatenate frames horizontally
        combined_frame = np.hstack(frames)
        out.write(combined_frame)
    
    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"Merged video saved to: {output_path}")

def generate_output_filename(base_name="golf_swing"):
    """Generate timestamped output filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.mp4"

def resize_frame(frame, width, height):
    """Resize frame to specified dimensions"""
    return cv2.resize(frame, (width, height))

def add_timestamp_to_frame(frame, timestamp_text):
    """Add timestamp overlay to frame"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2
    
    # Get text size
    text_size = cv2.getTextSize(timestamp_text, font, font_scale, thickness)[0]
    
    # Position in top-left corner
    x = 10
    y = text_size[1] + 10
    
    # Add black background for text
    cv2.rectangle(frame, (x-5, y-text_size[1]-5), (x+text_size[0]+5, y+5), (0, 0, 0), -1)
    
    # Add text
    cv2.putText(frame, timestamp_text, (x, y), font, font_scale, color, thickness)
    
    return frame