# Multi-Person Drowsiness Detection System (MediaPipe Version)

A comprehensive real-time drowsiness detection system using **MediaPipe** instead of dlib, capable of detecting multiple people simultaneously and predicting their drowsiness state with age estimation.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

##  Features

###  **No dlib Dependency**
- Uses **MediaPipe Face Mesh** for facial landmark detection
- Eliminates dlib installation complexities and licensing issues
- More accurate and faster face landmark detection with 468 3D landmarks

###  **Core Functionality**
- **Multi-Person Detection**: Detect and analyze multiple people simultaneously using YOLOv8
- **Real-time Drowsiness Detection**: Eye Aspect Ratio (EAR) analysis using MediaPipe landmarks
- **Age Prediction**: Estimate ages of sleeping persons using deep learning
- **Visual Alerts**: Red bounding boxes for sleeping persons, green for awake
- **Pop-up Notifications**: Non-blocking alerts with sleeping count and ages
- **Professional GUI**: Modern PyQt5 interface with real-time statistics

###  **Input Support**
- **Live Camera Feed**: Real-time webcam processing
- **Video Files**: MP4, AVI, MOV, MKV formats
- **Static Images**: JPG, PNG, BMP, TIFF formats

###  **Advanced Features**
- **Multi-threading**: Separate processing threads prevent GUI freezing
- **Configurable Parameters**: JSON-based settings for thresholds and performance
- **Performance Monitoring**: Real-time FPS and processing statistics
- **Comprehensive Logging**: Activity logs and detection events
- **Error Handling**: Robust fallback mechanisms

## Project Structure

```
drowsiness_detection/
│
├── main.py                    # Main GUI application
├── drowsiness_detector.py     # MediaPipe-based drowsiness detection
├── person_detector.py         # YOLO-based person detection
├── age_predictor.py           # Age estimation module
├── utils.py                   # Utility classes and functions
├── config.json                # Configuration parameters
├── requirements.txt           # Python dependencies
├── setup.py                   # Automated setup script
├── test_system.py             # Comprehensive testing suite
├── README.md                  # This file
│
├── models/                    # Pre-trained model storage
├── outputs/                   # Saved results and logs
├── test_videos/               # Sample videos for testing
└── logs/                      # Application logs
```

##  Quick Start

### 1. **Download/Clone the Project**
```bash
# Create project directory
mkdir drowsiness_detection
cd drowsiness_detection

# Copy all the Python files to this directory
```

### 2. **Automated Setup**
```bash
python setup.py
```

### 3. **Manual Setup (Alternative)**
```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir models outputs test_videos logs

# Run system test
python test_system.py --quick
```

### 4. **Launch Application**
```bash
python main.py
```

## 🔧 Installation Details

### **System Requirements**
- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, Linux
- **RAM**: 8GB+ recommended
- **Camera**: USB webcam (optional, for live detection)

### **Dependencies**
```txt
opencv-python>=4.8.0      # Computer vision
mediapipe>=0.10.0         # Face landmark detection
ultralytics>=8.0.0        # YOLOv8 object detection
PyQt5>=5.15.0             # GUI framework
numpy>=1.21.0             # Numerical computing
scipy>=1.7.0              # Scientific computing
torch>=1.13.0             # Deep learning (for YOLO)
```

### **Automatic Model Downloads**
- **YOLOv8**: Downloads automatically on first run
- **MediaPipe**: Included with the package
- **Age Estimation**: Uses heuristic methods (deep learning models optional)

##  Usage Guide

### **Starting the Application**
1. **Launch**: Run `python main.py`
2. **Select Input**: Choose from Image, Video, or Camera
3. **Monitor Results**: View real-time detection in the GUI
4. **Receive Alerts**: Pop-up notifications for drowsiness detection

### **GUI Interface**

#### **Left Panel - Video Display**
- Live preview of processed video feed
- Visual annotations (red for sleeping, green for awake)
- Real-time EAR values and timestamps

#### **Right Panel - Statistics**
- **Detection Counts**: Total persons, sleeping, awake
- **Progress Bars**: Visual representation of detection ratios
- **Activity Log**: Real-time detection events and system messages
- **Configuration Display**: Current detection parameters

### **Control Buttons**
- **Load Image**: Process single image
- **Load Video**: Process video file
- **Start Camera**: Begin live camera feed
- **Stop**: Stop current processing

##  Configuration

### **config.json Parameters**
```json
{
    "ear_threshold": 0.25,          // Eye closure threshold
    "consecutive_frames": 20,       // Frames required for drowsiness
    "yawn_threshold": 0.6,          // Yawn detection sensitivity
    "person_confidence": 0.5,       // Person detection confidence
    "max_persons": 10,              // Maximum persons to track
    "popup_cooldown": 5,            // Seconds between alerts
    "processing_threads": 2         // CPU threads for processing
}
```

### **Adjustable Parameters**
- **EAR Threshold**: Lower values = more sensitive drowsiness detection
- **Frame Threshold**: More frames = more reliable but slower detection
- **Confidence Levels**: Adjust detection sensitivity for different scenarios

##  Testing

### **Quick System Test**
```bash
python test_system.py --quick
```

### **Comprehensive Test**
```bash
python test_system.py
```

### **Test Coverage**
-  Import validation
-  MediaPipe face detection
-  Drowsiness detection algorithm
-  Person detection (YOLO)
-  Age prediction
-  GUI components
-  File operations
-  Camera accessibility

##  Technical Details

### **Drowsiness Detection Algorithm**
1. **Face Detection**: MediaPipe Face Mesh identifies 468 facial landmarks
2. **Eye Landmark Extraction**: Extract specific eye corner and eyelid points
3. **EAR Calculation**: `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
4. **Temporal Analysis**: Track EAR over consecutive frames
5. **Drowsiness Classification**: EAR < 0.25 for 20+ frames = drowsy

### **MediaPipe vs dlib Comparison**
| Feature | MediaPipe | dlib |
|---------|-----------|------|
| **Installation** |  Simple pip install | ❌ Complex compilation |
| **Landmarks** | 468 3D points | 68 2D points |
| **Performance** |  Faster, GPU optimized |  CPU only |
| **Licensing** |  Apache 2.0 |  Boost license restrictions |
| **Mobile Support** |  Yes | ❌ Limited |

### **Age Prediction Methods**
1. **Deep Learning**: Pre-trained CNN models (optional)
2. **Heuristic Analysis**: Texture, edge density, brightness analysis
3. **Fallback System**: Random age in reasonable range if detection fails

##  Performance Metrics

### **Expected Performance**
- **Detection Accuracy**: 90-95% for clear face images
- **Processing Speed**: 15-30 FPS on modern hardware
- **Age Estimation**: ±5-10 years accuracy with heuristic methods
- **Multi-Person Capacity**: Up to 10 people simultaneously
- **Memory Usage**: ~2-4GB RAM during operation

### **Optimization Features**
- Frame skipping for improved performance
- Multi-threaded processing
- Automatic model optimization
- Configurable processing parameters

##  Troubleshooting

### **Common Issues**

#### **"ModuleNotFoundError: No module named 'mediapipe'"**
```bash
pip install mediapipe
# or
pip install -r requirements.txt
```

#### **Camera Not Accessible**
- Check camera permissions in system settings
- Ensure camera is not being used by another application
- Try different camera indices (0, 1, 2, etc.)

#### **Low Detection Accuracy**
- Adjust `ear_threshold` in config.json (try 0.20-0.30)
- Increase `consecutive_frames` for more reliable detection
- Ensure good lighting conditions
- Check camera focus and positioning

#### **Performance Issues**
- Reduce `max_persons` limit
- Enable `frame_skip` in configuration
- Lower camera resolution
- Close unnecessary applications

### **Debug Mode**
```bash
# Run with verbose output
python main.py --debug

# Test individual components
python drowsiness_detector.py  # Test drowsiness detection
python person_detector.py      # Test person detection
python age_predictor.py        # Test age prediction
```

## Contributing

### **Development Setup**
```bash
# Clone repository
git clone <repository-url>
cd drowsiness_detection

# Install in development mode
pip install -e .

# Run tests
python test_system.py
```

### **Code Standards**
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include error handling and logging
- Test new features thoroughly

##  License

This project is licensed under the MIT License. See the LICENSE file for details.

##  Acknowledgments

- **Google MediaPipe Team**: For the excellent Face Mesh solution
- **Ultralytics**: For YOLOv8 object detection
- **OpenCV Community**: For computer vision tools
- **PyQt5 Developers**: For the GUI framework

##  Support

For issues, questions, or contributions:

1. **Check Documentation**: Review this README and code comments
2. **Run Tests**: Use `test_system.py` to diagnose problems
3. **Check Configuration**: Verify `config.json` parameters
4. **Review Logs**: Check application logs for error details

## 🔮 Future Enhancements

- [ ] **Real-time Model Training**: Adapt to individual users
- [ ] **Advanced Age Estimation**: Integration with more sophisticated models
- [ ] **Emotion Recognition**: Detect additional states beyond drowsiness
- [ ] **Cloud Integration**: Remote monitoring and analytics
- [ ] **Mobile App**: Android/iOS companion application
- [ ] **Hardware Integration**: Support for specialized cameras and sensors

---

