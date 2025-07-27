#!/usr/bin/env python3
"""
Multi-Person Drowsiness Detection System (No dlib)
Main application using MediaPipe for facial landmark detection
"""

import sys
import os
import json
import threading
import time
from datetime import datetime
import cv2
from networkx import is_path
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                            QFileDialog, QMessageBox, QFrame, QGridLayout,
                            QGroupBox, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette

# Import our custom modules
from drowsiness_detector import DrowsinessDetector
from person_detector import PersonDetector
from age_predictor import AgePredictor
from utils import VideoProcessor, PopupManager, ConfigManager

class VideoThread(QThread):
    """Dedicated thread for video processing to prevent GUI freezing"""
    change_pixmap = pyqtSignal(np.ndarray)
    update_stats = pyqtSignal(dict)
    drowsiness_alert = pyqtSignal(int, list)  # count, ages

    def __init__(self):
        super().__init__()
        self.running = False
        self.source = None
        self.source_type = None  # 'camera', 'video', 'image'

        # Initialize detection modules
        self.drowsiness_detector = DrowsinessDetector()
        self.person_detector = PersonDetector()
        self.age_predictor = AgePredictor()
        self.config = ConfigManager()

    def set_source(self, source, source_type):
        """Set video source (camera index, video path, or image path)"""
        self.source = source
        self.source_type = source_type

    def run(self):
        """Main video processing loop"""
        self.running = True

        if self.source_type == 'camera':
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while self.running:
                ret, frame = cap.read()
                if ret:
                    processed_frame, stats = self.process_frame(frame)
                    self.change_pixmap.emit(processed_frame)
                    self.update_stats.emit(stats)

                    # Check for drowsiness alerts
                    if stats['sleeping_count'] > 0:
                        self.drowsiness_alert.emit(stats['sleeping_count'], stats['sleeping_ages'])

                self.msleep(33)  # ~30 FPS

            cap.release()

        elif self.source_type == 'video':
            cap = cv2.VideoCapture(self.source)

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, stats = self.process_frame(frame)
                self.change_pixmap.emit(processed_frame)
                self.update_stats.emit(stats)

                # Check for drowsiness alerts
                if stats['sleeping_count'] > 0:
                    self.drowsiness_alert.emit(stats['sleeping_count'], stats['sleeping_ages'])

                self.msleep(33)  # ~30 FPS

            cap.release()

        elif self.source_type == 'image':
            frame = cv2.imread(self.source)
            if frame is not None:
                processed_frame, stats = self.process_frame(frame)
                self.change_pixmap.emit(processed_frame)
                self.update_stats.emit(stats)

                # Check for drowsiness alerts
                if stats['sleeping_count'] > 0:
                    self.drowsiness_alert.emit(stats['sleeping_count'], stats['sleeping_ages'])

    def process_frame(self, frame):
        """Process a single frame for drowsiness detection"""
        try:
            # Detect persons in the frame
            person_boxes = self.person_detector.detect_persons(frame)

            sleeping_count = 0
            awake_count = 0
            sleeping_ages = []

            for box in person_boxes:
                x1, y1, x2, y2 = box
                person_roi = frame[y1:y2, x1:x2]

                # Detect drowsiness for this person
                is_drowsy, ear_value, landmarks = self.drowsiness_detector.detect_drowsiness(person_roi)

                if is_drowsy:
                    # Draw red bounding box for sleeping person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, 'SLEEPING', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Predict age for sleeping person
                    age = self.age_predictor.predict_age(person_roi)
                    sleeping_ages.append(age)
                    cv2.putText(frame, f'Age: {age}', (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    sleeping_count += 1
                else:
                    # Draw green bounding box for awake person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'AWAKE', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    awake_count += 1

                # Display EAR value
                cv2.putText(frame, f'EAR: {ear_value:.3f}', (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            stats = {
                'total_persons': len(person_boxes),
                'sleeping_count': sleeping_count,
                'awake_count': awake_count,
                'sleeping_ages': sleeping_ages,
                'timestamp': timestamp
            }

            return frame, stats

        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, {'total_persons': 0, 'sleeping_count': 0, 'awake_count': 0, 'sleeping_ages': [], 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    def stop(self):
        """Stop the video processing thread"""
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.video_thread = VideoThread()
        self.popup_manager = PopupManager()
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Multi-Person Drowsiness Detection System (MediaPipe)')
        self.setGeometry(100, 100, 1200, 800)

        # Set modern dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
                border-color: #777777;
            }
            QPushButton:pressed {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 11px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
            QGroupBox {
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel for video display
        left_panel = QVBoxLayout()

        # Video display area
        video_group = QGroupBox("Live Video Feed")
        video_layout = QVBoxLayout(video_group)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #555555; background-color: #1e1e1e;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No Video Source\nSelect input source below")

        video_layout.addWidget(self.video_label)
        left_panel.addWidget(video_group)

        # Control buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)

        self.btn_load_image = QPushButton("📷 Load Image")
        self.btn_load_video = QPushButton("🎥 Load Video")
        self.btn_camera = QPushButton("📹 Start Camera")
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setEnabled(False)

        controls_layout.addWidget(self.btn_load_image)
        controls_layout.addWidget(self.btn_load_video)
        controls_layout.addWidget(self.btn_camera)
        controls_layout.addWidget(self.btn_stop)

        left_panel.addWidget(controls_group)

        # Right panel for statistics and logs
        right_panel_widget = QWidget()
        right_panel_widget.setMaximumWidth(350)

        right_panel=QVBoxLayout(right_panel_widget)

        # Statistics display
        stats_group = QGroupBox("Detection Statistics")
        stats_layout = QGridLayout(stats_group)

        self.lbl_total_persons = QLabel("Total Persons: 0")
        self.lbl_sleeping_count = QLabel("Sleeping: 0")
        self.lbl_awake_count = QLabel("Awake: 0")
        self.lbl_timestamp = QLabel("Last Update: --")

        # Progress bars for visual representation
        self.progress_sleeping = QProgressBar()
        self.progress_sleeping.setStyleSheet("QProgressBar::chunk { background-color: #ff4444; }")
        self.progress_awake = QProgressBar()
        self.progress_awake.setStyleSheet("QProgressBar::chunk { background-color: #44ff44; }")

        stats_layout.addWidget(self.lbl_total_persons, 0, 0, 1, 2)
        stats_layout.addWidget(QLabel("Sleeping:"), 1, 0)
        stats_layout.addWidget(self.progress_sleeping, 1, 1)
        stats_layout.addWidget(self.lbl_sleeping_count, 2, 0, 1, 2)
        stats_layout.addWidget(QLabel("Awake:"), 3, 0)
        stats_layout.addWidget(self.progress_awake, 3, 1)
        stats_layout.addWidget(self.lbl_awake_count, 4, 0, 1, 2)
        stats_layout.addWidget(self.lbl_timestamp, 5, 0, 1, 2)

        right_panel.addWidget(stats_group)

        # Activity log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.append("System initialized successfully")
        self.log_text.append("Ready for drowsiness detection...")

        log_layout.addWidget(self.log_text)
        right_panel.addWidget(log_group)

        # Configuration panel
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        self.lbl_ear_threshold = QLabel("EAR Threshold: 0.25")
        self.lbl_frame_threshold = QLabel("Frame Threshold: 20")
        config_layout.addWidget(self.lbl_ear_threshold)
        config_layout.addWidget(self.lbl_frame_threshold)

        right_panel.addWidget(config_group)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 3)
        main_layout.addWidget(right_panel_widget, 1)


        # Status bar
        self.statusBar().showMessage("Ready - Select an input source to begin detection")

    def connect_signals(self):
        """Connect button signals and video thread signals"""
        # Button connections
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_processing)

        # Video thread connections
        self.video_thread.change_pixmap.connect(self.update_image)
        self.video_thread.update_stats.connect(self.update_statistics)
        self.video_thread.drowsiness_alert.connect(self.show_drowsiness_alert)

    def load_image(self):
        """Load and process a single image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image files (*.jpg *.jpeg *.png *.bmp)")

        if file_path:
            self.log_text.append(f"Loading image: {file_path}")
            if self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.set_source(file_path, 'image')
                self.video_thread.start()

            
            self.update_button_states(True)
            self.statusBar().showMessage("Processing image...")

    def load_video(self):
        """Load and process a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", 
            "Video files (*.mp4 *.avi *.mov *.mkv)")

        if file_path:
            self.log_text.append(f"Loading video: {file_path}")
            if self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.set_source(file_path, 'video')
                self.video_thread.start()

            self.update_button_states(True)
            self.statusBar().showMessage("Processing video...")

    def start_camera(self):
        """Start camera feed processing"""
        self.log_text.append("Starting camera feed...")
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.set_source(0, 'camera')
            self.video_thread.start()

        self.update_button_states(True)
        self.statusBar().showMessage("Camera active - Detecting drowsiness in real-time")

    def stop_processing(self):
        """Stop video processing"""
        if self.video_thread.isRunning():
            self.log_text.append("Stopping video processing...")
            self.video_thread.stop()
            self.update_button_states(False)
            self.video_label.setText("Processing Stopped\nSelect input source to continue")
            self.statusBar().showMessage("Ready - Select an input source to begin detection")

    def update_button_states(self, processing):
        """Update button enabled states based on processing status"""
        self.btn_load_image.setEnabled(not processing)
        self.btn_load_video.setEnabled(not processing)
        self.btn_camera.setEnabled(not processing)
        self.btn_stop.setEnabled(processing)

    def update_image(self, frame):
        """Update the video display with new frame"""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Scale image to fit display
            scaled_image = qt_image.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pixmap = QPixmap.fromImage(scaled_image)
            self.video_label.setPixmap(pixmap)

        except Exception as e:
            self.log_text.append(f"Error updating display: {e}")

    def update_statistics(self, stats):
        """Update statistics display"""
        total = stats['total_persons']
        sleeping = stats['sleeping_count']
        awake = stats['awake_count']

        # Update labels
        self.lbl_total_persons.setText(f"Total Persons: {total}")
        self.lbl_sleeping_count.setText(f"Sleeping: {sleeping}")
        self.lbl_awake_count.setText(f"Awake: {awake}")
        self.lbl_timestamp.setText(f"Last Update: {stats['timestamp']}")

        # Update progress bars
        if total > 0:
            sleeping_percent = int((sleeping / total) * 100)
            awake_percent = int((awake / total) * 100)
            self.progress_sleeping.setValue(sleeping_percent)
            self.progress_awake.setValue(awake_percent)
        else:
            self.progress_sleeping.setValue(0)
            self.progress_awake.setValue(0)

        # Log detection events
        if sleeping > 0:
            ages_str = ", ".join([str(age) for age in stats['sleeping_ages']])
            self.log_text.append(f"ALERT: {sleeping} person(s) sleeping - Ages: {ages_str}")

    def show_drowsiness_alert(self, count, ages):
        """Show popup alert for drowsiness detection"""
        ages_str = ", ".join([str(age) for age in ages])
        message = f"⚠️ DROWSINESS DETECTED!\n\n{count} person(s) detected as sleeping\nEstimated ages: {ages_str}"

        self.popup_manager.show_popup("Drowsiness Alert", message)

    def closeEvent(self, event):
        """Handle application close event"""
        if self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    # Set application properties
    app.setApplicationName("Drowsiness Detection System")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("AI Vision Solutions")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
