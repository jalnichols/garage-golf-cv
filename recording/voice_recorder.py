#!/usr/bin/env python3
"""
Voice-controlled recording system for golf swing analysis
Listens for voice commands to start/stop synchronized recording from all cameras
"""

import cv2
import numpy as np
import speech_recognition as sr
import threading
import queue
import time
import os
import sys
from collections import deque
from datetime import datetime
sys.path.append('..')

from config.settings import (
    CAMERA_COUNT, RECORDING_WIDTH, RECORDING_HEIGHT, RECORDING_FPS,
    VOICE_COMMANDS, PRE_SWING_BUFFER, POST_SWING_BUFFER,
    OUTPUT_VIDEOS_PATH, CALIBRATION_DATA_PATH
)
from utils.camera_utils import (
    initialize_cameras, release_cameras, capture_synchronized_frames,
    load_camera_calibration, undistort_frame
)
from utils.video_utils import (
    create_video_writer, merge_videos_side_by_side, 
    generate_output_filename, add_timestamp_to_frame
)

class VoiceControlledRecorder:
    def __init__(self, use_calibration=True):
        """Initialize the voice-controlled recording system"""
        self.cameras = []
        self.calibration_data = []
        self.use_calibration = use_calibration
        
        # Recording state
        self.is_recording = False
        self.recording_started = False
        self.stop_recording_flag = False
        
        # Frame buffers for pre-swing recording
        self.frame_buffers = []
        self.buffer_size = int(PRE_SWING_BUFFER * RECORDING_FPS)
        
        # Video writers
        self.video_writers = []
        self.recording_start_time = None
        self.current_recording_files = []
        
        # Voice recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_queue = queue.Queue()
        self.listening = False
        
        # Threading
        self.voice_thread = None
        self.recording_thread = None
        self.running = False
        
        # Multithreaded frame capture
        self.capture_threads = []
        self.frame_queues = [queue.Queue(maxsize=100) for _ in range(CAMERA_COUNT)]
        
    def initialize(self):
        """Initialize cameras, calibration, and voice recognition"""
        print("Initializing cameras...")
        
        # Initialize cameras
        self.cameras = initialize_cameras(RECORDING_WIDTH, RECORDING_HEIGHT, RECORDING_FPS)
        
        if len(self.cameras) == 0:
            print("No cameras detected!")
            return False
        
        print(f"Initialized {len(self.cameras)} cameras")
        
        # Initialize frame buffers
        for i in range(len(self.cameras)):
            self.frame_buffers.append(deque(maxlen=self.buffer_size))
        
        # Load calibration data
        if self.use_calibration:
            print("Loading calibration data...")
            for i in range(len(self.cameras)):
                mtx, dist = load_camera_calibration(i, CALIBRATION_DATA_PATH)
                self.calibration_data.append((mtx, dist))
                if mtx is not None:
                    print(f"  Camera {i}: Calibration loaded")
                else:
                    print(f"  Camera {i}: No calibration data")
        
        # Initialize voice recognition
        print("Initializing voice recognition...")
        try:
            with self.microphone as source:
                print("Adjusting for ambient noise... (please be quiet)")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Voice recognition ready!")
            return True
        except Exception as e:
            print(f"Voice recognition setup failed: {e}")
            print("Continuing without voice control (use keyboard instead)")
            return True
    
    def capture_loop(self, cam_index):
        """Continuous frame capture loop for a specific camera"""
        cam = self.cameras[cam_index]
        while self.running:
            ret, frame = cam.read()
            if not ret:
                continue
            if self.use_calibration and cam_index < len(self.calibration_data):
                mtx, dist = self.calibration_data[cam_index]
                frame = undistort_frame(frame, mtx, dist)
            try:
                self.frame_queues[cam_index].put_nowait((frame, datetime.now()))
            except queue.Full:
                pass
    
    def listen_for_commands(self):
        """Voice recognition thread function"""
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    # Recognize speech
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{command}'")
                    self.voice_queue.put(command)
                except sr.UnknownValueError:
                    # Speech not recognized, ignore
                    pass
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    
            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                pass
            except Exception as e:
                print(f"Voice recognition error: {e}")
                time.sleep(1)
    
    def process_voice_command(self, command):
        """Process recognized voice commands"""
        # Check for start commands
        for start_cmd in VOICE_COMMANDS['start']:
            if start_cmd in command:
                if not self.is_recording:
                    print(f"Voice command detected: '{command}' -> START RECORDING")
                    self.start_recording()
                return
        
        # Check for stop commands
        for stop_cmd in VOICE_COMMANDS['stop']:
            if stop_cmd in command:
                if self.is_recording:
                    print(f"Voice command detected: '{command}' -> STOP RECORDING")
                    self.stop_recording()
                return
    
    def start_recording(self):
        """Start recording from all cameras"""
        if self.is_recording:
            print("Already recording!")
            return
        
        print("Starting recording...")
        self.is_recording = True
        self.recording_started = True
        self.stop_recording_flag = False
        self.recording_start_time = time.time()
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_recording_files = []
        
        # Create video writers
        self.video_writers = []
        for i in range(len(self.cameras)):
            filename = os.path.join(OUTPUT_VIDEOS_PATH, f"camera_{i}_{timestamp}.mp4")
            writer = create_video_writer(filename, RECORDING_WIDTH, RECORDING_HEIGHT, RECORDING_FPS)
            self.video_writers.append(writer)
            self.current_recording_files.append(filename)
            print(f"  Camera {i}: {filename}")
        
        # Write buffered frames (pre-swing)
        print(f"Writing {len(self.frame_buffers[0])} buffered frames...")
        max_buffer_len = max(len(buf) for buf in self.frame_buffers)
        
        for frame_idx in range(max_buffer_len):
            for cam_idx, buffer in enumerate(self.frame_buffers):
                if frame_idx < len(buffer) and self.video_writers[cam_idx] is not None:
                    frame = buffer[frame_idx]
                    self.video_writers[cam_idx].write(frame)
    
    def stop_recording(self):
        """Stop recording and save files"""
        if not self.is_recording:
            print("Not currently recording!")
            return
        
        print("Stopping recording...")
        self.stop_recording_flag = True
        
        # Continue recording for post-swing buffer
        post_frames = int(POST_SWING_BUFFER * RECORDING_FPS)
        print(f"Recording {post_frames} additional frames...")
        
        frames_recorded = 0
        while frames_recorded < post_frames and self.running:
            frames, _ = capture_synchronized_frames(self.cameras)
            
            for i, frame in enumerate(frames):
                if frame is not None and i < len(self.video_writers):
                    # Apply calibration if available
                    if self.use_calibration and i < len(self.calibration_data):
                        mtx, dist = self.calibration_data[i]
                        frame = undistort_frame(frame, mtx, dist)
                    
                    # Add timestamp
                    timestamp_text = f"Camera {i} - {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                    frame = add_timestamp_to_frame(frame, timestamp_text)
                    
                    self.video_writers[i].write(frame)
            
            frames_recorded += 1
        
        # Close video writers
        for writer in self.video_writers:
            if writer is not None:
                writer.release()
        
        self.is_recording = False
        self.recording_started = False
        
        recording_duration = time.time() - self.recording_start_time
        print(f"Recording complete! Duration: {recording_duration:.2f} seconds")
        
        # Merge videos
        if len(self.current_recording_files) >= 2:
            print("Merging videos...")
            merged_filename = os.path.join(OUTPUT_VIDEOS_PATH, 
                                         generate_output_filename("golf_swing_merged"))
            try:
                merge_videos_side_by_side(self.current_recording_files, merged_filename)
                print(f"Merged video saved: {merged_filename}")
            except Exception as e:
                print(f"Video merging failed: {e}")
        
        self.video_writers = []
        self.current_recording_files = []
    
    def run(self):
        """Main recording loop"""
        if not self.initialize():
            return
        
        # Create output directory
        os.makedirs(OUTPUT_VIDEOS_PATH, exist_ok=True)
        
        print("\n=== Voice-Controlled Golf Swing Recorder ===")
        print("Voice commands:")
        print(f"  Start: {', '.join(VOICE_COMMANDS['start'])}")
        print(f"  Stop: {', '.join(VOICE_COMMANDS['stop'])}")
        print("\nKeyboard controls:")
        print("  'Space' - Start/Stop recording")
        print("  'q' - Quit")
        print("  'v' - Toggle voice control")
        print()
        
        self.running = True
        
        # Start voice recognition thread
        self.listening = True
        self.voice_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
        self.voice_thread.start()
        
        # Start capture threads
        for i in range(len(self.cameras)):
            t = threading.Thread(target=self.capture_loop, args=(i,), daemon=True)
            self.capture_threads.append(t)
            t.start()
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while self.running:
                # Capture frames from queues
                frames = []
                timestamps = []
                for i in range(len(self.cameras)):
                    try:
                        frame, ts = self.frame_queues[i].get_nowait()
                    except queue.Empty:
                        frame, ts = None, None
                    frames.append(frame)
                    timestamps.append(ts)
                
                # Process frames
                display_frames = []
                for i, frame in enumerate(frames):
                    if frame is not None:
                        # Frame is already calibrated in capture_loop
                        processed_frame = frame.copy()
                        
                        # Add to buffer
                        self.frame_buffers[i].append(processed_frame.copy())
                        
                        # If recording, write to file
                        if self.is_recording and not self.stop_recording_flag:
                            if i < len(self.video_writers) and self.video_writers[i] is not None:
                                # Add timestamp to recorded frame
                                timestamp_text = f"Camera {i} - {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                                recorded_frame = add_timestamp_to_frame(processed_frame.copy(), timestamp_text)
                                self.video_writers[i].write(recorded_frame)
                        
                        # Create display frame (smaller)
                        display_frame = cv2.resize(processed_frame, (320, 240))
                        
                        # Add recording indicator
                        if self.is_recording:
                            cv2.circle(display_frame, (300, 20), 8, (0, 0, 255), -1)
                            cv2.putText(display_frame, "REC", (270, 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        cv2.putText(display_frame, f"Cam {i}", (10, 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        display_frames.append(display_frame)
                
                # Show combined display
                if display_frames:
                    if len(display_frames) == 1:
                        combined = display_frames[0]
                    else:
                        combined = np.hstack(display_frames[:min(3, len(display_frames))])
                    
                    # Add status info
                    status = "RECORDING" if self.is_recording else "READY"
                    color = (0, 0, 255) if self.is_recording else (0, 255, 0)
                    cv2.putText(combined, status, (10, combined.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if not self.is_recording:
                       cv2.imshow('Golf Swing Recorder', combined)
                    else:
                       cv2.destroyWindow('Golf Swing Recorder')
                
                # Process voice commands
                try:
                    while not self.voice_queue.empty():
                        command = self.voice_queue.get_nowait()
                        self.process_voice_command(command)
                except queue.Empty:
                    pass
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    if self.is_recording:
                        self.stop_recording()
                    break
                elif key == ord(' '):  # Spacebar
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('v'):
                    self.listening = not self.listening
                    print(f"Voice control: {'ON' if self.listening else 'OFF'}")
                
                # Update FPS counter
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = 30 / elapsed
                    print(f"Processing FPS: {fps:.1f}")
                    fps_start = time.time()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        if self.is_recording:
            self.stop_recording()
        
        self.running = False
        self.listening = False
        
        # Wait for threads to finish
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=2)
        
        # Join capture threads
        for t in self.capture_threads:
            if t.is_alive():
                t.join(timeout=1)
        
        # Release cameras and close windows
        cv2.destroyAllWindows()
        release_cameras(self.cameras)

def main():
    """Main function"""
    print("=== Golf Swing Voice Recorder ===")
    
    # Ask user about calibration
    use_cal = input("Use camera calibration? (y/n, default=y): ").lower()
    use_calibration = use_cal != 'n'
    
    # Create and run recorder
    recorder = VoiceControlledRecorder(use_calibration=use_calibration)
    recorder.run()

if __name__ == "__main__":
    main()