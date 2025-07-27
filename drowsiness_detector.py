#!/usr/bin/env python3
"""
Drowsiness Detection Module using MediaPipe
Replaces dlib with MediaPipe Face Mesh for facial landmark detection
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import json

class DrowsinessDetector:
    """
    Drowsiness detection using Eye Aspect Ratio (EAR) with MediaPipe Face Mesh
    """

    def __init__(self, config_path='config.json'):
        """Initialize the drowsiness detector with MediaPipe"""
        self.config = self.load_config(config_path)

        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # EAR calculation parameters
        self.EAR_THRESH = self.config.get('ear_threshold', 0.25)
        self.EAR_CONSEC_FRAMES = self.config.get('consecutive_frames', 20)

        # Frame counter for consecutive closed eye detection
        self.COUNTER = 0

        # MediaPipe landmark indices for eyes
        # Left eye landmarks (viewer's perspective)
        self.LEFT_EYE = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]

        # Right eye landmarks (viewer's perspective) 
        self.RIGHT_EYE = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ]

        # Specific points for EAR calculation
        # Left eye EAR points: horizontal corners and vertical points
        self.LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # P1, P2, P3, P4, P5, P6
        self.RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]  # P1, P2, P3, P4, P5, P6

        # Mouth landmarks for yawn detection
        self.MOUTH_LANDMARKS = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318
        ]

        print("DrowsinessDetector initialized with MediaPipe Face Mesh")

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            default_config = {
                "ear_threshold": 0.25,
                "consecutive_frames": 20,
                "yawn_threshold": 0.6,
                "detection_confidence": 0.5,
                "tracking_confidence": 0.5
            }
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config

    def calculate_ear(self, landmarks, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR) for given eye landmarks
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        try:
            # Extract the eye coordinates
            eye_coords = []
            for point_idx in eye_points:
                if point_idx < len(landmarks):
                    landmark = landmarks[point_idx]
                    x = int(landmark.x * self.frame_width)
                    y = int(landmark.y * self.frame_height)
                    eye_coords.append((x, y))
                else:
                    return 0.3  # Default EAR for open eye

            if len(eye_coords) < 6:
                return 0.3

            # Calculate vertical distances
            vertical_1 = dist.euclidean(eye_coords[1], eye_coords[5])  # p2-p6
            vertical_2 = dist.euclidean(eye_coords[2], eye_coords[4])  # p3-p5

            # Calculate horizontal distance
            horizontal = dist.euclidean(eye_coords[0], eye_coords[3])  # p1-p4

            # Calculate EAR
            if horizontal > 0:
                ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            else:
                ear = 0.3

            return ear

        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.3  # Default value for open eye

    def detect_yawn(self, landmarks):
        """Detect yawning based on mouth aspect ratio"""
        try:
            mouth_coords = []
            for point_idx in self.MOUTH_LANDMARKS:
                if point_idx < len(landmarks):
                    landmark = landmarks[point_idx]
                    x = int(landmark.x * self.frame_width)
                    y = int(landmark.y * self.frame_height)
                    mouth_coords.append((x, y))

            if len(mouth_coords) < 6:
                return False

            # Calculate mouth aspect ratio (similar to EAR)
            vertical_1 = dist.euclidean(mouth_coords[1], mouth_coords[5])
            vertical_2 = dist.euclidean(mouth_coords[2], mouth_coords[4])
            horizontal = dist.euclidean(mouth_coords[0], mouth_coords[3])

            if horizontal > 0:
                mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
                return mar > self.config.get('yawn_threshold', 0.6)

            return False

        except Exception as e:
            print(f"Error detecting yawn: {e}")
            return False

    def draw_landmarks(self, frame, landmarks, eye_points, color=(0, 255, 0)):
        """Draw eye landmarks on the frame"""
        try:
            for point_idx in eye_points:
                if point_idx < len(landmarks):
                    landmark = landmarks[point_idx]
                    x = int(landmark.x * self.frame_width)
                    y = int(landmark.y * self.frame_height)
                    cv2.circle(frame, (x, y), 2, color, -1)
        except Exception as e:
            print(f"Error drawing landmarks: {e}")

    def detect_drowsiness(self, frame):
        """
        Main drowsiness detection function
        Returns: (is_drowsy, ear_value, landmarks)
        """
        try:
            if frame is None or frame.size == 0:
                return False, 0.3, None

            # Store frame dimensions for coordinate conversion
            self.frame_height, self.frame_width = frame.shape[:2]

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Face Mesh
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark

                # Calculate EAR for both eyes
                left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_EAR)
                right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_EAR)

                # Average EAR
                avg_ear = (left_ear + right_ear) / 2.0

                # Check for yawning
                is_yawning = self.detect_yawn(landmarks)

                # Draw eye landmarks for visualization
                self.draw_landmarks(frame, landmarks, self.LEFT_EYE, (0, 255, 0))
                self.draw_landmarks(frame, landmarks, self.RIGHT_EYE, (0, 255, 0))

                # EAR-based drowsiness detection
                if avg_ear < self.EAR_THRESH:
                    self.COUNTER += 1

                    # If eyes closed for consecutive frames
                    if self.COUNTER >= self.EAR_CONSEC_FRAMES:
                        # Draw red landmarks for drowsy detection
                        self.draw_landmarks(frame, landmarks, self.LEFT_EYE, (0, 0, 255))
                        self.draw_landmarks(frame, landmarks, self.RIGHT_EYE, (0, 0, 255))

                        return True, avg_ear, landmarks
                else:
                    self.COUNTER = 0

                # Additional check for yawning (indicator of drowsiness)
                if is_yawning:
                    # Draw mouth landmarks
                    for point_idx in self.MOUTH_LANDMARKS[:6]:
                        if point_idx < len(landmarks):
                            landmark = landmarks[point_idx]
                            x = int(landmark.x * self.frame_width)
                            y = int(landmark.y * self.frame_height)
                            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

                return False, avg_ear, landmarks

            else:
                # No face detected
                self.COUNTER = 0
                return False, 0.3, None

        except Exception as e:
            print(f"Error in drowsiness detection: {e}")
            return False, 0.3, None

    def get_detection_info(self):
        """Get current detection parameters"""
        return {
            'ear_threshold': self.EAR_THRESH,
            'consecutive_frames': self.EAR_CONSEC_FRAMES,
            'current_counter': self.COUNTER,
            'detector_type': 'MediaPipe Face Mesh'
        }

    def reset_counter(self):
        """Reset the consecutive frame counter"""
        self.COUNTER = 0

    def update_threshold(self, new_threshold):
        """Update EAR threshold dynamically"""
        self.EAR_THRESH = new_threshold
        self.config['ear_threshold'] = new_threshold

    def update_frame_threshold(self, new_frame_threshold):
        """Update consecutive frames threshold"""
        self.EAR_CONSEC_FRAMES = new_frame_threshold
        self.config['consecutive_frames'] = new_frame_threshold

# Test function for standalone usage
def test_drowsiness_detector():
    """Test the drowsiness detector with webcam"""
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)

    print("Testing Drowsiness Detector...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect drowsiness
        is_drowsy, ear, landmarks = detector.detect_drowsiness(frame)

        # Display results
        if is_drowsy:
            cv2.putText(frame, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Drowsiness Detection Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_drowsiness_detector()
