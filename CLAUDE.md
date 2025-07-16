# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-camera golf swing analysis system using three Logitech C922 webcams (60fps @ 1080p) for capturing multi-perspective golf swing videos in a garage setup.

## Technology Stack

- **Language**: Python
- **Computer Vision**: OpenCV for camera handling, calibration, and video processing
- **Audio**: speech_recognition library for voice commands
- **GUI**: tkinter or OpenCV for live preview interface
- **Video**: FFmpeg for video merging and processing

## Core Components

### 1. Camera Calibration (`calibration/`)
- Checkerboard-based camera calibration for each C922 webcam
- Intrinsic and extrinsic parameter calculation
- Distortion correction matrices
- Multi-camera synchronization calibration

### 2. Live Preview System (`preview/`)
- Low-latency, low-resolution feed from all three cameras
- Real-time display for swing positioning
- Minimal processing overhead to maintain responsiveness

### 3. Voice-Controlled Recording (`recording/`)
- Voice command recognition ("start swing", "stop swing")
- Synchronized recording across all three cameras
- Automatic video clipping and compilation
- Export to combined multi-perspective video file

## Development Commands

When setting up the development environment:

```bash
# Install dependencies
pip install opencv-python speech_recognition pyaudio numpy

# Run camera calibration
python calibration/calibrate_cameras.py

# Start live preview
python preview/live_preview.py

# Start recording system
python recording/voice_recorder.py
```

## Project Structure

```
garage-golf-cv/
├── calibration/
│   ├── calibrate_cameras.py
│   ├── checkerboard_images/
│   └── camera_params/
├── preview/
│   └── live_preview.py
├── recording/
│   ├── voice_recorder.py
│   └── output_videos/
├── utils/
│   ├── camera_utils.py
│   └── video_utils.py
└── config/
    └── settings.py
```

## Camera Setup

- Three Logitech C922 webcams positioned at different angles
- 1080p resolution at 60fps for recording
- Lower resolution (480p) for live preview to reduce latency
- USB 3.0 connections recommended for bandwidth

## Key Development Notes

- Handle USB camera enumeration and assignment consistently
- Implement frame synchronization across cameras for recording
- Buffer management for smooth voice-triggered recording
- Consider lighting conditions in garage environment