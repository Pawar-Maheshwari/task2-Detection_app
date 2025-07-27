#!/usr/bin/env python3
"""
System Test Script for Multi-Person Drowsiness Detection System
Tests all components without dlib dependency
"""

import sys
import os
import cv2
import numpy as np
import json
from datetime import datetime

def test_imports():
    """Test all required imports"""
    print(" Testing imports...")
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from PyQt5.QtWidgets import QApplication
        from scipy.spatial import distance
        from ultralytics import YOLO
        print(" All imports successful!")
        return True
    except ImportError as e:
        print(f" Import failed: {e}")
        return False

def test_config_system():
    """Test configuration management"""
    print("\n Testing configuration system...")
    try:
        from utils import ConfigManager

        # Test config loading
        config = ConfigManager('test_config.json')

        # Test config operations
        original_threshold = config.get('ear_threshold')
        config.set('ear_threshold', 0.3)
        new_threshold = config.get('ear_threshold')

        if new_threshold == 0.3:
            print(" Configuration system working!")
            # Cleanup
            if os.path.exists('test_config.json'):
                os.remove('test_config.json')
            return True
        else:
            print(" Configuration system failed!")
            return False

    except Exception as e:
        print(f" Configuration test failed: {e}")
        return False

def test_mediapipe_detection():
    """Test MediaPipe face mesh detection"""
    print("\n Testing MediaPipe face detection...")
    try:
        import mediapipe as mp

        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create a test image (simple face-like pattern)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = face_mesh.process(rgb_image)

        print(" MediaPipe Face Mesh initialized successfully!")
        return True

    except Exception as e:
        print(f" MediaPipe test failed: {e}")
        return False

def test_drowsiness_detector():
    """Test drowsiness detection module"""
    print("\n Testing drowsiness detection...")
    try:
        from drowsiness_detector import DrowsinessDetector

        # Initialize detector
        detector = DrowsinessDetector()

        # Create test image
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Test detection (should handle no face gracefully)
        is_drowsy, ear, landmarks = detector.detect_drowsiness(test_frame)

        # Check if function returns expected types
        if isinstance(is_drowsy, bool) and isinstance(ear, (int, float)):
            print(" Drowsiness detector working!")

            # Test info retrieval
            info = detector.get_detection_info()
            print(f"   Detector type: {info.get('detector_type', 'Unknown')}")
            print(f"   EAR threshold: {info.get('ear_threshold', 'N/A')}")
            return True
        else:
            print(" Drowsiness detector returned unexpected types!")
            return False

    except Exception as e:
        print(f" Drowsiness detector test failed: {e}")
        return False

def test_person_detector():
    """Test person detection module"""
    print("\n Testing person detection...")
    try:
        from person_detector import PersonDetector

        # Initialize detector
        detector = PersonDetector()

        # Create test image
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Test detection
        person_boxes = detector.detect_persons(test_frame)

        # Check if function returns a list
        if isinstance(person_boxes, list):
            print(" Person detector working!")

            # Test info retrieval
            info = detector.get_detection_info()
            print(f"   Detection method: {info.get('method', 'Unknown')}")
            print(f"   Confidence threshold: {info.get('confidence_threshold', 'N/A')}")
            return True
        else:
            print(" Person detector returned unexpected type!")
            return False

    except Exception as e:
        print(f" Person detector test failed: {e}")
        return False

def test_age_predictor():
    """Test age prediction module"""
    print("\n Testing age prediction...")
    try:
        from age_predictor import AgePredictor

        # Initialize predictor
        predictor = AgePredictor()

        # Create test image
        test_roi = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Test prediction
        predicted_age = predictor.predict_age(test_roi)

        # Check if function returns a reasonable age
        if isinstance(predicted_age, (int, float)) and 0 <= predicted_age <= 100:
            print(" Age predictor working!")

            # Test info retrieval
            info = predictor.get_predictor_info()
            print(f"   Prediction method: {info.get('method', 'Unknown')}")
            print(f"   Models loaded: {info.get('models_loaded', False)}")
            return True
        else:
            print(f" Age predictor returned invalid age: {predicted_age}")
            return False

    except Exception as e:
        print(f" Age predictor test failed: {e}")
        return False

def test_camera_access():
    """Test camera accessibility"""
    print("\n Testing camera access...")
    try:
        # Try to open default camera
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                print(" Camera accessible and working!")
                print(f"   Frame shape: {frame.shape}")
                return True
            else:
                print("  Camera opened but failed to read frame")
                return False
        else:
            print("  Camera not accessible (this is okay for image/video processing)")
            return True

    except Exception as e:
        print(f"  Camera test failed: {e}")
        return True  # Not critical failure

def test_gui_components():
    """Test GUI components without actually showing window"""
    print("\n Testing GUI components...")
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QCoreApplication

        # Test if QApplication can be created
        if QCoreApplication.instance() is None:
            app = QApplication([])
            print(" Qt GUI framework working!")
            app.quit()
            return True
        else:
            print(" Qt GUI framework available!")
            return True

    except Exception as e:
        print(f" GUI test failed: {e}")
        return False

def test_file_operations():
    """Test file I/O operations"""
    print("\n Testing file operations...")
    try:
        from utils import VideoProcessor, create_output_directory

        # Test output directory creation
        output_dir = create_output_directory('test_outputs')

        if os.path.exists(output_dir):
            print(" Output directory creation working!")

            # Test video processor
            processor = VideoProcessor()

            # Test file type detection
            image_type = processor.get_file_type('test.jpg')
            video_type = processor.get_file_type('test.mp4')

            if image_type == 'image' and video_type == 'video':
                print(" File type detection working!")

                # Cleanup
                import shutil
                if os.path.exists('test_outputs'):
                    shutil.rmtree('test_outputs')

                return True
            else:
                print(" File type detection failed!")
                return False
        else:
            print(" Output directory creation failed!")
            return False

    except Exception as e:
        print(f" File operations test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print(" Multi-Person Drowsiness Detection System - Comprehensive Test")
    print("=" * 70)

    tests = [
        ("Import Test", test_imports),
        ("Configuration System", test_config_system),
        ("MediaPipe Detection", test_mediapipe_detection),
        ("Drowsiness Detector", test_drowsiness_detector),
        ("Person Detector", test_person_detector),
        ("Age Predictor", test_age_predictor),
        ("Camera Access", test_camera_access),
        ("GUI Components", test_gui_components),
        ("File Operations", test_file_operations)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'─' * 50}")
        result = test_func()
        if result:
            passed += 1

    print(f"\n{'═' * 70}")
    print(f" TEST SUMMARY")
    print(f"{'═' * 70}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n ALL TESTS PASSED! System is ready to use.")
        print("\n Ready to run:")
        print("   python main.py  - Start the GUI application")
    elif passed >= total * 0.8:
        print("\n MOST TESTS PASSED! System should work with minor issues.")
        print("\n You can try running:")
        print("   python main.py  - Start the GUI application")
    else:
        print("\n  SEVERAL TESTS FAILED! Please check your installation.")
        print("\n Try running:")
        print("   python setup.py  - Re-run setup")

    return passed == total

def run_quick_test():
    """Run essential tests only"""
    print(" Quick System Test")
    print("=" * 30)

    essential_tests = [
        ("Imports", test_imports),
        ("MediaPipe", test_mediapipe_detection),
        ("Drowsiness Detection", test_drowsiness_detector),
        ("GUI Framework", test_gui_components)
    ]

    all_passed = True
    for test_name, test_func in essential_tests:
        print(f"\nTesting {test_name}...")
        if not test_func():
            all_passed = False

    if all_passed:
        print("\n Essential components working! Ready to use.")
    else:
        print("\n Some essential components failed. Check installation.")

    return all_passed

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        return run_quick_test()
    else:
        return run_comprehensive_test()

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  Unexpected error during testing: {e}")
        sys.exit(1)
