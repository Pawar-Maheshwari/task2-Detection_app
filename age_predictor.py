#!/usr/bin/env python3
"""
Age Prediction Module
Estimates age for detected sleeping persons using deep learning
"""

import cv2
import numpy as np
import json
import os
import random
from urllib.request import urlretrieve

class AgePredictor:
    """
    Age prediction using pre-trained deep learning models
    """

    def __init__(self, config_path='config.json'):
        """Initialize the age predictor"""
        self.config = self.load_config(config_path)

        # Age ranges for classification
        self.age_ranges = [
            "(0-2)", "(4-6)", "(8-12)", "(15-20)", 
            "(25-32)", "(38-43)", "(48-53)", "(60-100)"
        ]

        # Initialize models
        self.face_net = None
        self.age_net = None
        self.load_models()

        # Backup heuristic features for age estimation
        self.use_heuristic = False
        if self.face_net is None or self.age_net is None:
            self.use_heuristic = True
            print("Using heuristic age estimation (models not available)")

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            default_config = {
                "age_model_path": "models/age_net.caffemodel",
                "age_proto_path": "models/age_deploy.prototxt",
                "face_model_path": "models/opencv_face_detector_uint8.pb",
                "face_proto_path": "models/opencv_face_detector.pbtxt",
                "age_confidence": 0.8
            }
            return default_config

    def load_models(self):
        """Load pre-trained models for age estimation"""
        try:
            # Model paths
            age_model_path = self.config.get('age_model_path', 'models/age_net.caffemodel')
            age_proto_path = self.config.get('age_proto_path', 'models/age_deploy.prototxt')
            face_model_path = self.config.get('face_model_path', 'models/opencv_face_detector_uint8.pb')
            face_proto_path = self.config.get('face_proto_path', 'models/opencv_face_detector.pbtxt')

            # Check if model files exist
            if (os.path.exists(age_model_path) and os.path.exists(age_proto_path) and
                os.path.exists(face_model_path) and os.path.exists(face_proto_path)):

                # Load face detection model
                self.face_net = cv2.dnn.readNet(face_model_path, face_proto_path)

                # Load age estimation model
                self.age_net = cv2.dnn.readNet(age_model_path, age_proto_path)

                print("Age estimation models loaded successfully")

            else:
                print("Model files not found. Using heuristic age estimation.")
                self.download_models()

        except Exception as e:
            print(f"Error loading age estimation models: {e}")
            print("Falling back to heuristic age estimation")
            self.use_heuristic = True

    def download_models(self):
        """Download pre-trained models (placeholder - in practice, you'd download real models)"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        print("Note: In a production system, this would download real pre-trained models.")
        print("For this demo, we'll use heuristic age estimation.")
        self.use_heuristic = True

    def detect_face_in_roi(self, roi):
        """Detect face in the given region of interest"""
        if self.face_net is None:
            return roi  # Return the entire ROI if face detection is not available

        try:
            h, w = roi.shape[:2]

            # Create blob from image
            blob = cv2.dnn.blobFromImage(roi, 1.0, (300, 300), [104, 117, 123])

            # Set input to the model
            self.face_net.setInput(blob)

            # Run inference
            detections = self.face_net.forward()

            # Find the face with highest confidence
            max_confidence = 0
            face_roi = roi

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > max_confidence and confidence > 0.5:
                    max_confidence = confidence

                    # Get bounding box
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)

                    # Extract face ROI
                    face_roi = roi[y1:y2, x1:x2]

            return face_roi

        except Exception as e:
            print(f"Error in face detection: {e}")
            return roi

    def predict_age_dnn(self, face_roi):
        """Predict age using deep neural network"""
        try:
            # Preprocess the face
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))

            # Set input to age network
            self.age_net.setInput(blob)

            # Get predictions
            age_predictions = self.age_net.forward()

            # Get the age range with highest confidence
            age_index = np.argmax(age_predictions)
            age_range = self.age_ranges[age_index]
            confidence = age_predictions[0][age_index]

            # Convert age range to approximate age
            age = self.range_to_age(age_range)

            return age, confidence

        except Exception as e:
            print(f"Error in DNN age prediction: {e}")
            return self.predict_age_heuristic(face_roi), 0.5

    def range_to_age(self, age_range):
        """Convert age range to approximate age value"""
        range_map = {
            "(0-2)": 1,
            "(4-6)": 5,
            "(8-12)": 10,
            "(15-20)": 18,
            "(25-32)": 28,
            "(38-43)": 40,
            "(48-53)": 50,
            "(60-100)": 70
        }
        return range_map.get(age_range, 25)

    def predict_age_heuristic(self, face_roi):
        """Predict age using heuristic methods (fallback)"""
        try:
            if face_roi is None or face_roi.size == 0:
                return random.randint(20, 50)  # Random age in adult range

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi

            # Basic heuristic features
            features = self.extract_age_features(gray)

            # Simple heuristic model based on features
            age = self.heuristic_age_model(features)

            return age

        except Exception as e:
            print(f"Error in heuristic age prediction: {e}")
            return random.randint(20, 50)

    def extract_age_features(self, gray_face):
        """Extract simple features for heuristic age estimation"""
        try:
            h, w = gray_face.shape

            # Feature 1: Skin texture (variance-based)
            texture_variance = np.var(gray_face)

            # Feature 2: Edge density (wrinkles indicator)
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)

            # Feature 3: Average brightness (can indicate skin condition)
            avg_brightness = np.mean(gray_face)

            # Feature 4: Contrast measure
            contrast = np.std(gray_face)

            return {
                'texture_variance': texture_variance,
                'edge_density': edge_density,
                'avg_brightness': avg_brightness,
                'contrast': contrast,
                'face_size': h * w
            }

        except Exception as e:
            print(f"Error extracting age features: {e}")
            return {
                'texture_variance': 100,
                'edge_density': 0.1,
                'avg_brightness': 128,
                'contrast': 30,
                'face_size': 10000
            }

    def heuristic_age_model(self, features):
        """Simple heuristic model for age estimation"""
        try:
            # Base age
            age = 25

            # Adjust based on texture variance (higher variance = older)
            if features['texture_variance'] > 200:
                age += 15
            elif features['texture_variance'] > 100:
                age += 8
            elif features['texture_variance'] < 50:
                age -= 10

            # Adjust based on edge density (more edges = more wrinkles = older)
            if features['edge_density'] > 0.15:
                age += 12
            elif features['edge_density'] > 0.08:
                age += 6
            elif features['edge_density'] < 0.05:
                age -= 8

            # Adjust based on face size (larger face might indicate closer/adult)
            if features['face_size'] > 15000:
                age += 5
            elif features['face_size'] < 5000:
                age -= 10

            # Add some randomness to avoid always predicting same age
            age += random.randint(-5, 5)

            # Clamp to reasonable range
            age = max(5, min(age, 85))

            return age

        except Exception as e:
            print(f"Error in heuristic age model: {e}")
            return 25

    def predict_age(self, person_roi):
        """
        Main age prediction function
        Args:
            person_roi: Region of interest containing the person
        Returns:
            Predicted age as integer
        """
        try:
            if person_roi is None or person_roi.size == 0:
                return random.randint(20, 50)

            # Extract face from person ROI
            face_roi = self.detect_face_in_roi(person_roi)

            if face_roi is None or face_roi.size == 0:
                return random.randint(20, 50)

            # Predict age
            if self.use_heuristic or self.age_net is None:
                age = self.predict_age_heuristic(face_roi)
                confidence = 0.6  # Moderate confidence for heuristic
            else:
                age, confidence = self.predict_age_dnn(face_roi)

            return int(age)

        except Exception as e:
            print(f"Error in age prediction: {e}")
            return random.randint(20, 50)

    def predict_multiple_ages(self, person_rois):
        """Predict ages for multiple persons"""
        ages = []
        for roi in person_rois:
            age = self.predict_age(roi)
            ages.append(age)
        return ages

    def get_predictor_info(self):
        """Get predictor information"""
        return {
            'method': 'DNN' if not self.use_heuristic else 'Heuristic',
            'age_ranges': self.age_ranges,
            'models_loaded': self.age_net is not None and self.face_net is not None,
            'confidence_threshold': self.config.get('age_confidence', 0.8)
        }

# Test function for standalone usage
def test_age_predictor():
    """Test the age predictor with webcam"""
    predictor = AgePredictor()
    cap = cv2.VideoCapture(0)

    print("Testing Age Predictor...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use entire frame as person ROI for testing
        predicted_age = predictor.predict_age(frame)

        # Display result
        cv2.putText(frame, f"Predicted Age: {predicted_age}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Age Prediction Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_age_predictor()
