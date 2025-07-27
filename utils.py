#!/usr/bin/env python3
"""
Utility Classes and Functions
Includes video processing, popup management, and configuration handling
"""

import cv2
import json
import os
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QIcon

class VideoProcessor:
    """
    Video processing utilities for handling different input sources
    """

    def __init__(self):
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

    def get_file_type(self, file_path):
        """Determine if file is image, video, or unsupported"""
        _, ext = os.path.splitext(file_path.lower())

        if ext in self.supported_image_formats:
            return 'image'
        elif ext in self.supported_video_formats:
            return 'video'
        else:
            return 'unsupported'

    def get_video_info(self, video_path):
        """Get video information"""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration
            }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None

    def resize_frame(self, frame, max_width=640, max_height=480):
        """Resize frame while maintaining aspect ratio"""
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Calculate scaling factor
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)

        # Only resize if frame is larger than max dimensions
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame

    def save_frame(self, frame, output_path, timestamp=None):
        """Save frame to file with optional timestamp"""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"frame_{timestamp}.png"
            full_path = os.path.join(output_path, filename)

            # Create directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)

            cv2.imwrite(full_path, frame)
            return full_path
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None

class PopupManager(QObject):
    """
    Manages popup alerts for drowsiness detection
    Non-blocking popup system
    """

    show_popup_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.show_popup_signal.connect(self._show_popup_slot)
        self.popup_active = False
        self.popup_cooldown = 5  # seconds between popups
        self.last_popup_time = 0

    def show_popup(self, title, message):
        """Show non-blocking popup message"""
        current_time = time.time()

        # Check cooldown to prevent spam
        if current_time - self.last_popup_time < self.popup_cooldown:
            return

        if not self.popup_active:
            self.show_popup_signal.emit(title, message)
            self.last_popup_time = current_time

    def _show_popup_slot(self, title, message):
        """Internal slot to show popup in main thread"""
        try:
            self.popup_active = True

            msg_box = QMessageBox()
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStandardButtons(QMessageBox.Ok)

            # Set popup style
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #ff4444;
                    color: #ffffff;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ff6666;
                }
            """)

            # Show popup and wait for user response
            msg_box.exec_()

            self.popup_active = False

        except Exception as e:
            print(f"Error showing popup: {e}")
            self.popup_active = False

    def set_popup_cooldown(self, seconds):
        """Set cooldown period between popups"""
        self.popup_cooldown = seconds

class ConfigManager:
    """
    Configuration management for the drowsiness detection system
    """

    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file"""
        default_config = {
            # Drowsiness detection parameters
            "ear_threshold": 0.25,
            "consecutive_frames": 20,
            "yawn_threshold": 0.6,

            # Detection confidence thresholds
            "detection_confidence": 0.5,
            "tracking_confidence": 0.5,
            "person_confidence": 0.5,
            "age_confidence": 0.8,

            # Model paths
            "yolo_model_path": "models\yolov8n.pt",
            "age_model_path": "models/age_net.caffemodel",
            "age_proto_path": "models/age_deploy.prototxt",
            "face_model_path": "models/opencv_face_detector_uint8.pb",
            "face_proto_path": "models/opencv_face_detector.pbtxt",

            # Processing parameters
            "max_persons": 10,
            "frame_skip": 1,
            "nms_threshold": 0.4,

            # UI settings
            "popup_cooldown": 5,
            "log_max_lines": 1000,
            "auto_save_frames": False,
            "output_directory": "outputs",

            # Performance settings
            "use_gpu": False,
            "processing_threads": 2,
            "max_fps": 30
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_config.update(loaded_config)
                    return default_config
            else:
                # Create default config file
                self.save_config(default_config)
                return default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return default_config

    def save_config(self, config=None):
        """Save configuration to file"""
        try:
            config_to_save = config if config is not None else self.config

            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=4)

        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
        self.save_config()

    def update(self, updates):
        """Update multiple configuration values"""
        self.config.update(updates)
        self.save_config()

    def reset_to_defaults(self):
        """Reset configuration to default values"""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        self.config = self.load_config()

class PerformanceMonitor:
    """
    Monitor system performance and FPS
    """

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.start_time = None

    def start_frame(self):
        """Mark the start of frame processing"""
        self.start_time = time.time()

    def end_frame(self):
        """Mark the end of frame processing and calculate FPS"""
        if self.start_time is None:
            return 0

        frame_time = time.time() - self.start_time
        self.frame_times.append(frame_time)

        # Keep only recent frame times
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

        return self.get_fps()

    def get_fps(self):
        """Get current FPS"""
        if not self.frame_times:
            return 0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    def get_avg_frame_time(self):
        """Get average frame processing time in ms"""
        if not self.frame_times:
            return 0

        return (sum(self.frame_times) / len(self.frame_times)) * 1000

class Logger:
    """
    Simple logging utility for the application
    """

    def __init__(self, log_file='drowsiness_detection.log', max_size=1024*1024):
        self.log_file = log_file
        self.max_size = max_size
        self.ensure_log_directory()

    def ensure_log_directory(self):
        """Create log directory if it doesn't exist"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log(self, message, level='INFO'):
        """Write log message"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}\n"

            # Check file size and rotate if necessary
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > self.max_size:
                self.rotate_log()

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)

        except Exception as e:
            print(f"Error writing to log: {e}")

    def rotate_log(self):
        """Rotate log file when it gets too large"""
        try:
            if os.path.exists(self.log_file):
                backup_file = f"{self.log_file}.old"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                os.rename(self.log_file, backup_file)
        except Exception as e:
            print(f"Error rotating log: {e}")

    def info(self, message):
        """Log info message"""
        self.log(message, 'INFO')

    def warning(self, message):
        """Log warning message"""
        self.log(message, 'WARNING')

    def error(self, message):
        """Log error message"""
        self.log(message, 'ERROR')

# Utility functions
def create_output_directory(base_dir='outputs'):
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"session_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def validate_input_file(file_path):
    """Validate if input file exists and is supported"""
    if not os.path.exists(file_path):
        return False, "File does not exist"

    processor = VideoProcessor()
    file_type = processor.get_file_type(file_path)

    if file_type == 'unsupported':
        return False, "Unsupported file format"

    return True, file_type

def get_system_info():
    """Get basic system information"""
    try:
        import platform
        import psutil

        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'opencv_version': cv2.__version__
        }
    except ImportError:
        return {
            'platform': 'Unknown',
            'opencv_version': cv2.__version__
        }

# Test functions
def test_config_manager():
    """Test configuration manager"""
    print("Testing ConfigManager...")

    config = ConfigManager('test_config.json')
    print(f"EAR threshold: {config.get('ear_threshold')}")

    config.set('ear_threshold', 0.3)
    print(f"Updated EAR threshold: {config.get('ear_threshold')}")

    # Cleanup
    if os.path.exists('test_config.json'):
        os.remove('test_config.json')

    print("ConfigManager test completed")

def test_performance_monitor():
    """Test performance monitor"""
    print("Testing PerformanceMonitor...")

    monitor = PerformanceMonitor()

    for i in range(10):
        monitor.start_frame()
        time.sleep(0.033)  # Simulate 30 FPS
        fps = monitor.end_frame()
        print(f"Frame {i+1}: {fps:.1f} FPS")

    print("PerformanceMonitor test completed")

if __name__ == "__main__":
    test_config_manager()
    test_performance_monitor()
    print("\nSystem Info:")
    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
