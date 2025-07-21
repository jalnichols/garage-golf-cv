#!/usr/bin/env python3
"""
Voice-controlled recording system for golf swing analysis
Listens for voice commands to start/stop synchronized recording from all cameras
FIXED VERSION - addresses GUI flickering and FPS issues
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
        
        # GUI state tracking
        self.window_visible = False
        self.was_recording = False
        
        # Improved frame capture with rate limiting
        self.capture_threads = []
        self.frame_queues = []  # Will be initialized based on actual camera count
        self.last_frame_time = []
        self.target_capture_fps = 60  # Limit capture FPS to match camera max
        
        # Main loop timing
        self.target_main_fps = 60
        self.frame_interval = 1.0 / self.target_main_fps
        
    def filter_c922_cameras(self, all_cameras):
        """Filter cameras to only include c922s, excluding internal webcams"""
        filtered_cameras = []
        
        print("Filtering cameras to exclude internal webcam...")
        
        for i, cam in enumerate(all_cameras):
            if cam is None:
                continue
                
            # Get camera properties
            width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cam.get(cv2.CAP_PROP_FPS)
            
            print(f"  Camera {i}: {width}x{height} @ {fps}fps")
            
            # c922 typically supports these combinations:
            # - 1920x1080 @ 30fps (some setups might show 60fps)
            # - 1280x720 @ 60fps
            # - Lower resolutions at 60fps
            
            # Filter criteria for c922 (exclude typical internal webcam resolutions)
            is_c922 = False
            
            # Check for common c922 resolution/fps combinations
            if (width >= 1280 and height >= 720):  # c922 supports HD and above
                if (width == 1920 and height == 1080):  # Full HD
                    is_c922 = True
                    print(f"    → Detected as c922 (Full HD capable)")
                elif (width == 1280 and height == 720):  # HD
                    is_c922 = True  
                    print(f"    → Detected as c922 (HD capable)")
                else:
                    print(f"    → Unknown high-res camera (including)")
                    is_c922 = True
            else:
                print(f"    → Likely internal webcam (excluding)")
            
            # Additional check: exclude cameras that can't do at least 720p
            if width < 1280 or height < 720:
                print(f"    → Resolution too low for c922 (excluding)")
                is_c922 = False
            
            if is_c922:
                filtered_cameras.append(cam)
                print(f"    ✅ Camera {i} included")
            else:
                print(f"    ❌ Camera {i} excluded")
                cam.release()  # Release excluded cameras
        
        return filtered_cameras
        
    def initialize(self):
        """Initialize cameras, calibration, and voice recognition"""
        print("Initializing cameras...")
        
        # Initialize all cameras first
        all_cameras = initialize_cameras(RECORDING_WIDTH, RECORDING_HEIGHT, RECORDING_FPS)
        
        if len(all_cameras) == 0:
            print("No cameras detected!")
            return False
        
        # Filter cameras to only include c922s (exclude internal webcam)
        self.cameras = self.filter_c922_cameras(all_cameras)
        
        if len(self.cameras) == 0:
            print("No c922 cameras detected! Check camera connections.")
            return False
        
        print(f"Initialized {len(self.cameras)} c922 cameras")
        
        # Initialize queues and timing arrays based on actual camera count
        self.frame_queues = [queue.Queue(maxsize=10) for _ in range(len(self.cameras))]
        self.last_frame_time = [0] * len(self.cameras)
        
        # Initialize frame buffers
        for i in range(len(self.cameras)):
            self.frame_buffers.append(deque(maxlen=self.buffer_size))
        
        # Load calibration data for filtered cameras
        if self.use_calibration:
            print("Loading calibration data for c922 cameras...")
            for i in range(len(self.cameras)):
                # Note: calibration files should be numbered based on camera order
                mtx, dist = load_camera_calibration(i, CALIBRATION_DATA_PATH)
                self.calibration_data.append((mtx, dist))
                if mtx is not None:
                    print(f"  c922 Camera {i}: Calibration loaded")
                else:
                    print(f"  c922 Camera {i}: No calibration data")
        
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
        """Continuous frame capture loop for a specific camera with rate limiting"""
        cam = self.cameras[cam_index]
        frame_interval = 1.0 / self.target_capture_fps
        
        while self.running:
            current_time = time.time()
            
            # Rate limiting
            if current_time - self.last_frame_time[cam_index] < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
                
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.01)  # Sleep on failed read
                continue
                
            self.last_frame_time[cam_index] = current_time
            
            # Apply calibration if available
            if self.use_calibration and cam_index < len(self.calibration_data):
                mtx, dist = self.calibration_data[cam_index]
                if mtx is not None:
                    frame = undistort_frame(frame, mtx, dist)
            
            try:
                # Only add to queue if not full (non-blocking)
                self.frame_queues[cam_index].put_nowait((frame, datetime.now()))
            except queue.Full:
                # Drop frame if queue is full
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
        
        print("Starting recording with c922 cameras...")
        print("*** GUI HIDDEN DURING RECORDING ***")
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
            filename = os.path.join(OUTPUT_VIDEOS_PATH, f"c922_{i}_{timestamp}.mp4")
            writer = create_video_writer(filename, RECORDING_WIDTH, RECORDING_HEIGHT, RECORDING_FPS)
            self.video_writers.append(writer)
            self.current_recording_files.append(filename)
            print(f"  c922 Camera {i}: {filename}")
        
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
                        if mtx is not None:
                            frame = undistort_frame(frame, mtx, dist)
                    
                    # Add timestamp
                    timestamp_text = f"c922-{i} - {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
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
        print("*** GUI WILL REAPPEAR ***")
        
        # Merge videos
        if len(self.current_recording_files) >= 2:
            print("Merging c922 videos...")
            merged_filename = os.path.join(OUTPUT_VIDEOS_PATH, 
                                         generate_output_filename("c922_golf_swing_merged"))
            try:
                merge_videos_side_by_side(self.current_recording_files, merged_filename)
                print(f"Merged c922 video saved: {merged_filename}")
            except Exception as e:
                print(f"Video merging failed: {e}")
        
        self.video_writers = []
        self.current_recording_files = []
    
    def run(self):
        """Main recording loop with proper timing control"""
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
        self.was_recording = False  # Initialize recording state tracking
        
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
        last_fps_log = time.time()  # Track when we last logged FPS
        fps_log_interval = 15.0  # Log FPS every 15 seconds
        last_loop_time = time.time()
        
        try:
            while self.running:
                loop_start_time = time.time()
                
                # Get latest frames from queues (with timeout to prevent blocking)
                frames = []
                timestamps = []
                for i in range(len(self.cameras)):
                    try:
                        # Try to get the most recent frame, discarding older ones
                        frame, ts = None, None
                        while True:
                            try:
                                frame, ts = self.frame_queues[i].get_nowait()
                            except queue.Empty:
                                break
                        if frame is None:
                            # If no frame available, try once more with short timeout
                            try:
                                frame, ts = self.frame_queues[i].get(timeout=0.001)
                            except queue.Empty:
                                pass
                    except:
                        frame, ts = None, None
                    
                    frames.append(frame)
                    timestamps.append(ts)
                
                # Handle GUI state changes (recording start/stop)
                if self.is_recording != self.was_recording:
                    if self.is_recording:
                        # Just started recording - hide the GUI
                        cv2.destroyWindow('Golf Swing Recorder')
                        self.window_visible = False
                        print("GUI hidden during recording")
                    else:
                        # Just stopped recording - will show GUI again below
                        self.window_visible = False  # Reset to recreate window
                        print("GUI will be shown again")
                    self.was_recording = self.is_recording
                
                # Process frames if we have any
                valid_frames = [f for f in frames if f is not None]
                if valid_frames:
                    display_frames = []
                    for i, frame in enumerate(frames):
                        if frame is not None:
                            processed_frame = frame.copy()
                            
                            # Add to buffer
                            self.frame_buffers[i].append(processed_frame.copy())
                            
                            # If recording, write to file
                            if self.is_recording and not self.stop_recording_flag:
                                if i < len(self.video_writers) and self.video_writers[i] is not None:
                                    # Add timestamp to recorded frame
                                    timestamp_text = f"c922-{i} - {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                                    recorded_frame = add_timestamp_to_frame(processed_frame.copy(), timestamp_text)
                                    self.video_writers[i].write(recorded_frame)
                            
                            # Only create display frames when NOT recording
                            if not self.is_recording:
                                # Create display frame (smaller)
                                display_frame = cv2.resize(processed_frame, (320, 240))
                                
                                cv2.putText(display_frame, f"c922-{i}", (10, 20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                display_frames.append(display_frame)
                    
                    # Show GUI only when NOT recording and we have enough frames
                    if not self.is_recording and display_frames and len(valid_frames) >= len(self.cameras):
                        if len(display_frames) == 1:
                            combined = display_frames[0]
                        else:
                            combined = np.hstack(display_frames[:min(3, len(display_frames))])
                        
                        # Add status info
                        cv2.putText(combined, "READY", (10, combined.shape[0] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow('Golf Swing Recorder', combined)
                        self.window_visible = True
                
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
                
                # Update FPS counter every 15 seconds
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_log >= fps_log_interval:
                    elapsed = current_time - fps_start
                    if elapsed > 0:
                        fps = frame_count / elapsed
                        status_msg = "RECORDING" if self.is_recording else "READY"
                        print(f"Processing FPS: {fps:.1f} - Status: {status_msg}")
                    
                    # Reset counters
                    frame_count = 0
                    fps_start = current_time
                    last_fps_log = current_time
                
                # FIXED: Rate limiting for main loop
                elapsed_time = time.time() - loop_start_time
                sleep_time = self.frame_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
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
        self.window_visible = False
        release_cameras(self.cameras)

def main():
    """Main function"""
    print("=== Golf Swing Voice Recorder ===")
    print("This version automatically filters to c922 cameras only")
    print("(Internal webcams will be excluded)")
    print()
    
    # Ask user about calibration
    use_cal = input("Use camera calibration? (y/n, default=y): ").lower()
    use_calibration = use_cal != 'n'
    
    # Create and run recorder
    recorder = VoiceControlledRecorder(use_calibration=use_calibration)
    recorder.run()
    recorder.run()

if __name__ == "__main__":
    main()