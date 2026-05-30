# Real-Time Multi-Person Tracking System with Trajectory Visualization

## 📌 Overview

This project implements a real-time multi-person tracking system using YOLOv8 and a custom centroid-based tracking algorithm. The system detects persons in video frames, assigns persistent IDs across frames, and visualizes movement trajectories.

The objective was to create a lightweight and explainable tracking pipeline suitable for drone-based surveillance scenarios where camera motion, small targets, and tracking consistency are challenging problems.

---

## ⚙️ System Pipeline

Input Video → YOLOv8 Person Detection → Centroid Extraction → ID Matching → Trajectory Visualization → Output Video

---

## 🧠 Architecture Choice

### Detection Model

* YOLOv8n (Nano version)
* Lightweight architecture suitable for real-time applications
* Fast inference speed with low memory usage
* Chosen to satisfy edge-device deployment requirements

### Tracking Algorithm

A custom centroid-based tracker was implemented.

Workflow:

1. Detect persons using YOLOv8.
2. Compute centroid of each detected bounding box.
3. Compare centroids with previous frame detections.
4. Assign the nearest matching ID using Euclidean distance.
5. Maintain trajectory history for visualization.

This approach is lightweight and computationally efficient compared to appearance-based trackers.

---

## 🔹 Object Detection

* Model: YOLOv8n
* Target Class: Person
* Bounding boxes generated for detected persons
* Lightweight architecture selected for drone deployment scenarios

---

## 🔹 Tracking

* Custom centroid-based tracking algorithm
* Unique IDs assigned to each detected person
* Euclidean-distance matching between consecutive frames
* Previous track information retained to reduce ID loss

---

## 🔹 Trajectory Visualization

* Historical positions stored for each track
* Trajectory lines drawn using previous centroids
* Visual representation of movement patterns over time

---

## ⚠️ Challenges Faced

### ID Switching

Fast movement, temporary occlusions, and camera motion occasionally caused incorrect ID assignments.

### Drone Ego-Motion

Because the camera itself is moving, object positions can change significantly between frames, making tracking more difficult.

### Detection Noise

Small variations in bounding-box positions may cause tracking inconsistencies.

---

## 🛠️ Mitigation Strategies

To reduce ID switching:

* Increased centroid matching threshold
* Retained previous track information
* Maintained short trajectory history
* Applied nearest-centroid matching between frames

While effective for lightweight tracking, appearance-based approaches such as DeepSORT would further improve robustness.

---

## 📊 Performance

### Hardware Used

* Apple MacBook Air
* Python 3.x
* YOLOv8n

### Performance Notes

* Lightweight model selected for real-time processing
* Performance depends on hardware configuration and video resolution
* Designed to prioritize inference speed while maintaining acceptable tracking quality

---

## 🚀 Edge Deployment Strategy

For deployment on edge hardware such as NVIDIA Jetson:

* Convert model to TensorRT format
* Apply INT8 quantization
* Reduce input resolution when necessary
* Use GPU acceleration available on Jetson devices
* Optimize inference pipeline for low-power environments

YOLOv8n was selected because of its small size and suitability for edge deployment.

---

## 📊 Output

Generated output:

final_output.mp4

Output includes:

* Person bounding boxes
* Persistent ID labels
* Movement trajectory visualization

---

## 🔧 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python3 advanced_track.py
```

### Input

Place the video in the project directory as:

```text
input.mp4
```

### Output

Generated file:

```text
final_output.mp4
```

---

## 💡 Key Learnings

* Difference between object detection and tracking
* Centroid-based multi-object tracking
* Challenges caused by camera motion
* ID-switching mitigation techniques
* Trade-offs between accuracy and computational efficiency
* Designing lightweight computer vision systems for deployment

---

## 🚀 Future Improvements

* DeepSORT integration
* ByteTrack integration
* Kalman Filter motion prediction
* Improved handling of occlusions
* Small-object optimization for aerial imagery
* TensorRT deployment pipeline

---

## 🏁 Conclusion

This project demonstrates a lightweight end-to-end multi-person tracking pipeline using YOLOv8 and centroid-based tracking. The system balances simplicity, speed, and interpretability while addressing common challenges encountered in drone-based surveillance applications.
