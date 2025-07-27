# Models Directory

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
