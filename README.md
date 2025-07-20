# Deadmouse
The deadmouseV2.py program is an interactive application that uses hand recognition through machine learning models to control the movement of a mouse pointer on a screen.

🎥 [Guarda il video demo](https://github.com/AlbertoVari/Deadmouse/blob/main/Deadmouse.mp4)


### **Summary Description**

1. **Initialization and Configuration**:
- The program initializes Pygame to create a graphics window (800x600 pixels) with a central black square and a red dot initially positioned at the center.
- It uses a webcam to capture video at a resolution of 1280x720 pixels, reducing the region of interest (ROI) to 256x256 pixels centered on the image.

2. **Hand Recognition with TensorFlow Lite**:
- Load the `hand_landmark.tflite` model (located in the `/tflite/` subfolder) using TensorFlow Lite to detect hand keypoints, specifically keypoint 8 (the tip of the index finger). This model, based on a convolutional neural network (CNN) optimized for hand landmark detection, returns normalized coordinates of the 21 main keypoints.
- The ROI image is preprocessed (resized and normalized) and passed to the model, which generates an output of 63 values (3 coordinates for each keypoint: x, y, z).

3. **Using the Intel NPU (MYRIAD) with OpenVINO Models**:
- The program is configured to leverage an Intel MYRIAD NPU (`device = "MYRIAD"`) to accelerate the inference of two OpenVINO models: `keypoint_classifier` and `point_history_classifier`, stored in the `/ir_models/` directory.
- The `keypoint_classifier` model, a classifier based on a lightweight neural network, analyzes the detected keypoints to identify gestures or hand states.
- The `point_history_classifier` model, also optimized for MYRIAD, uses a sequence of 16 keypoint (history) frames to determine the direction of index finger movement (up, down, left, right). These models, converted to IR (Intermediate Representation) format for OpenVINO, leverage the NPU to offload intensive computation from the CPU, improving real-time performance and energy efficiency, provided the MYRIAD device is connected and configured correctly.

4. **Tracking and Green Frame**:
- The coordinates of keypoint 8 are used to draw a 20x20 pixel green frame around the index in the "ROI" window, providing visual feedback.
- A green circle marks the exact location of keypoint 8 for greater precision.

5. **Dot Control**:
- The position of the index finger is monitored through a 16-frame history, calculating the distance and direction of movement with a threshold (`threshold_movement = 0.02`) and a stability (`direction_stability_threshold = 10`).
- The detected direction (up, down, left, right) moves the red dot within the square, with an increment of `step_size = 5` pixels, limited to the edges of the square.

6. **Management and Exit**:
- The program slows down processing with a 50ms delay (20 FPS) and exits by pressing the ESC key.
- Releases resources (webcam, windows) upon exit.

### **Role of the Intel NPU and ML Models**
The Intel MYRIAD NPU is integrated to perform inference of OpenVINO models (keypoint_classifier and point_history_classifier), reducing CPU computational load and enabling faster processing of keypoint data and direction classification. The TensorFlow Lite model hand_landmark.tflite handles initial keypoint detection on the CPU, while OpenVINO models leverage the NPU for advanced processing, optimizing performance in real-time applications. If MYRIAD is unavailable, the program can fallback to the CPU, resulting in lower performance.
