# Real-Time Multi-Person Tracking System with Trajectory Visualization

## 📌 Overview

This project implements a real-time multi-person tracking system using computer vision techniques. The system detects people in video frames, assigns consistent IDs across frames, and visualizes their movement using trajectory paths.

The goal was to build an efficient and explainable tracking pipeline suitable for applications such as drone surveillance and crowd monitoring.

---

## ⚙️ System Pipeline

Input Video → YOLOv8 Detection → Centroid-Based Tracking → ID Assignment → Trajectory Visualization → Output Video

---

## 🧠 Key Components

### 🔹 Object Detection

* Model: YOLOv8 (lightweight version)
* Detects multiple persons per frame with bounding boxes and confidence scores.

### 🔹 Tracking (Custom Implementation)

* Implemented a centroid-based tracking approach.
* Each detected object is assigned a unique ID.
* Matching between frames is done using Euclidean distance between object centroids.

### 🔹 Trajectory Visualization

* Stores previous positions of each tracked ID.
* Draws path lines to show movement across frames.
* Helps analyze motion patterns and object movement.

---

## ⚠️ Challenges Faced

* **ID Instability:** Same person getting different IDs across frames due to motion and detection variation.
* **Camera Movement:** Camera motion affected tracking consistency.
* **Detection Noise:** Small changes in bounding boxes caused tracking mismatches.

---

## 🛠️ Improvements Made

* Increased distance threshold for better matching.
* Added persistence of previous tracks to reduce ID loss.
* Implemented trajectory visualization for better analysis.
* Improved tracking stability through track-history maintenance.

---

## 🚀 Future Improvements

* Integrate DeepSORT for appearance-based tracking.
* Use Kalman Filters for motion prediction.
* Improve small-object detection for drone footage.
* Optimize FPS for real-time deployment.
* Handle occlusions more effectively.

---

## 📊 Output

The system generates an output video:

```text
final_output.mp4
```

The output includes:

* Bounding boxes around detected persons
* Unique IDs for each person
* Trajectory paths showing movement

---

## 🔧 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Tracker

```bash
python advanced_track.py
```

### Input

Place your video file in the project directory as:

```text
input.mp4
```

### Output

The processed video will be saved as:

```text
final_output.mp4
```

---

## 💡 Key Learnings

* Understanding the difference between object detection and object tracking.
* Implementing a custom tracking algorithm using centroid matching.
* Handling challenges such as ID switching and tracking instability.
* Balancing tracking accuracy and computational efficiency.
* Building an end-to-end computer vision pipeline.

---

## 🏁 Conclusion

This project demonstrates a complete end-to-end multi-person tracking pipeline using YOLOv8 and a custom centroid-based tracker. The project highlights practical challenges in real-world tracking systems and explores methods for improving robustness, scalability, and tracking accuracy.
