#!/usr/bin/env python3
"""
Setup Script for Multi-Person Drowsiness Detection System
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(" Python 3.7 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    print(f" Python {version.major}.{version.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\n Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n Creating directories...")
    directories = ['models', 'outputs', 'test_videos', 'logs']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f" Created directory: {directory}")
        else:
            print(f" Directory already exists: {directory}")

def download_models():
    """Download required models"""
    print("\n Downloading models...")
    try:
        # YOLOv8 will be downloaded automatically on first use
        print(" YOLOv8 model will be downloaded automatically on first run")

        # Create placeholder for age detection models
        models_dir = 'models'
        placeholder_file = os.path.join(models_dir, 'README.md')

        if not os.path.exists(placeholder_file):
            with open(placeholder_file, 'w') as f:
                f.write("""# Models Directory

This directory will contain the pre-trained models for the drowsiness detection system.

## Automatic Downloads:
- YOLOv8 model (models\yolov8n.pt) - Downloads automatically on first run
- MediaPipe models - Included with MediaPipe package

## Optional Age Detection Models:
For enhanced age prediction, you can place pre-trained Caffe models here:
- age_net.caffemodel
- age_deploy.prototxt
- opencv_face_detector_uint8.pb
- opencv_face_detector.pbtxt

If these models are not available, the system will use heuristic age estimation.
""")
        print(" Models directory configured")
        return True
    except Exception as e:
        print(f" Error setting up models: {e}")
        return False

def test_installation():
    """Test if installation was successful"""
    print("\n Testing installation...")
    try:
        # Test core imports
        import cv2
        import mediapipe as mp
        import numpy as np
        from PyQt5.QtWidgets import QApplication
        from ultralytics import YOLO

        print(" All core packages imported successfully!")

        # Test OpenCV
        print(f" OpenCV version: {cv2.__version__}")

        # Test MediaPipe
        print(" MediaPipe imported successfully")

        # Test camera availability (optional)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(" Camera detected and accessible")
            cap.release()
        else:
            print("  Camera not detected (optional for image/video processing)")

        return True

    except ImportError as e:
        print(f" Import error: {e}")
        return False
    except Exception as e:
        print(f" Test error: {e}")
        return False

def main():
    """Main setup function"""
    print(" Multi-Person Drowsiness Detection System Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        print("\n Setup failed during package installation")
        sys.exit(1)

    # Create directories
    create_directories()

    # Download/setup models
    if not download_models():
        print("\n  Model setup had issues, but continuing...")

    # Test installation
    if test_installation():
        print("\n" + "=" * 50)
        print(" Setup completed successfully!")
        print("\n Next steps:")
        print("1. Run 'python main.py' to start the application")
        print("2. Or run 'python test_system.py' to test components")
        print("3. Check config.json to adjust detection parameters")
    else:
        print("\n Setup completed with errors. Please check the installation.")

if __name__ == "__main__":
    main()
