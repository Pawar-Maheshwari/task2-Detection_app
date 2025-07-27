#!/usr/bin/env python3
"""
Person Detection Module using YOLO
Detects multiple people in images/video frames
"""

import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

class PersonDetector:
    """
    Multi-person detection using YOLOv8
    """

    def __init__(self, config_path='config.json'):
        """Initialize the person detector"""
        self.config = self.load_config(config_path)

        # Initialize YOLO model
        self.model_path = self.config.get('yolo_model_path', 'models\yolov8n.pt')
        self.confidence_threshold = self.config.get('person_confidence', 0.5)

        try:
            # Try to load YOLOv8 model
            self.yolo_model = YOLO(self.model_path)
            self.detection_method = 'yolo'
            print(f"Loaded YOLOv8 model from {self.model_path}")
        except Exception as e:
            print(f"Failed to load YOLOv8 model: {e}")
            print("Falling back to OpenCV HOG detector...")
            self.init_hog_detector()
            self.detection_method = 'hog'

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            default_config = {
                "yolo_model_path": "models\yolov8n.pt",
                "person_confidence": 0.5,
                "nms_threshold": 0.4,
                "max_persons": 10
            }
            return default_config

    def init_hog_detector(self):
        """Initialize OpenCV HOG detector as fallback"""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("HOG detector initialized as fallback")

    def detect_persons_yolo(self, frame):
        """Detect persons using YOLOv8"""
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)

            person_boxes = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a person (class 0 in COCO dataset)
                        if int(box.cls[0]) == 0:  # Person class
                            confidence = float(box.conf[0])

                            if confidence >= self.confidence_threshold:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                                # Ensure coordinates are within frame bounds
                                h, w = frame.shape[:2]
                                x1 = max(0, min(x1, w-1))
                                y1 = max(0, min(y1, h-1))
                                x2 = max(x1+1, min(x2, w))
                                y2 = max(y1+1, min(y2, h))

                                person_boxes.append([x1, y1, x2, y2, confidence])

            # Apply Non-Maximum Suppression to remove overlapping boxes
            if person_boxes:
                person_boxes = self.apply_nms(person_boxes)

            return person_boxes

        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []

    def detect_persons_hog(self, frame):
        """Detect persons using HOG detector (fallback)"""
        try:
            # Convert to grayscale for HOG
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect people
            boxes, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(32, 32),
                scale=1.05,
                finalThreshold=2
            )

            person_boxes = []
            for i, (x, y, w, h) in enumerate(boxes):
                confidence = weights[i] if i < len(weights) else 0.5
                person_boxes.append([x, y, x + w, y + h, confidence])

            return person_boxes

        except Exception as e:
            print(f"Error in HOG detection: {e}")
            return []

    def apply_nms(self, boxes, overlap_threshold=0.4):
        """Apply Non-Maximum Suppression to remove overlapping bounding boxes"""
        if not boxes:
            return []

        # Convert to numpy array
        boxes = np.array(boxes)

        # Extract coordinates and confidences
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        confidences = boxes[:, 4]

        # Calculate areas
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by confidence scores in descending order
        indices = np.argsort(confidences)[::-1]

        keep = []
        while len(indices) > 0:
            # Pick the first index
            current = indices[0]
            keep.append(current)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[current], x1[indices[1:]])
            yy1 = np.maximum(y1[current], y1[indices[1:]])
            xx2 = np.minimum(x2[current], x2[indices[1:]])
            yy2 = np.minimum(y2[current], y2[indices[1:]])

            # Calculate intersection area
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h

            # Calculate IoU
            union = areas[current] + areas[indices[1:]] - intersection
            iou = intersection / union

            # Keep boxes with IoU less than threshold
            indices = indices[1:][iou <= overlap_threshold]

        return boxes[keep].tolist()

    def detect_persons(self, frame):
        """
        Main person detection function
        Returns list of bounding boxes [x1, y1, x2, y2, confidence]
        """
        if frame is None or frame.size == 0:
            return []

        try:
            if self.detection_method == 'yolo':
                boxes = self.detect_persons_yolo(frame)
            else:
                boxes = self.detect_persons_hog(frame)

            # Limit number of detections
            max_persons = self.config.get('max_persons', 10)
            if len(boxes) > max_persons:
                # Sort by confidence and take top detections
                boxes.sort(key=lambda x: x[4], reverse=True)
                boxes = boxes[:max_persons]

            # Return only coordinate boxes (remove confidence for compatibility)
            return [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in boxes]

        except Exception as e:
            print(f"Error in person detection: {e}")
            return []

    def draw_detections(self, frame, person_boxes, color=(0, 255, 0)):
        """Draw bounding boxes on frame"""
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Person {i+1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def get_detection_info(self):
        """Get detector information"""
        return {
            'method': self.detection_method,
            'model_path': self.model_path if self.detection_method == 'yolo' else 'HOG',
            'confidence_threshold': self.confidence_threshold,
            'max_persons': self.config.get('max_persons', 10)
        }

    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold"""
        self.confidence_threshold = new_threshold
        self.config['person_confidence'] = new_threshold

def download_yolo_model():
    """Download YOLOv8 model if not available"""
    try:
        model_path = 'models\yolov8n.pt'
        if not os.path.exists(model_path):
            print("Downloading YOLOv8 nano model...")
            model = YOLO(model_path)  # This will download the model automatically
            print(f"YOLOv8 model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"Failed to download YOLOv8 model: {e}")
        return None

# Test function for standalone usage
def test_person_detector():
    """Test the person detector with webcam"""
    detector = PersonDetector()
    cap = cv2.VideoCapture(0)

    print("Testing Person Detector...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons
        person_boxes = detector.detect_persons(frame)

        # Draw detections
        frame = detector.draw_detections(frame, person_boxes)

        # Display count
        cv2.putText(frame, f"Persons detected: {len(person_boxes)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Person Detection Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Try to download YOLO model first
    download_yolo_model()
    test_person_detector()
