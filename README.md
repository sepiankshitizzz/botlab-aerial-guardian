## 🔧 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

requirements.txt includes:

* ultralytics
* opencv-python
* deep-sort-realtime

### Run Centroid Tracker

```bash
python3 advanced_track.py
```

### Run DeepSORT Tracker (Final Implementation)

```bash
python3 deepsort_track.py
```

### Input

Place the video file in the project directory as:

```text
input.mp4
```

### Outputs

```text
final_output.mp4
deepsort_output.mp4
```
## 🏁 Conclusion

This project demonstrates a complete end-to-end multi-person tracking pipeline using YOLOv8 and DeepSORT. The system combines accurate person detection, persistent identity tracking, and trajectory visualization while addressing practical challenges encountered in aerial surveillance scenarios.

DeepSORT was selected as the final tracking approach due to improved identity consistency and robustness compared to centroid-only matching.

The inclusion of both centroid-based and DeepSORT-based trackers highlights the trade-offs between lightweight tracking and more advanced multi-object tracking approaches.
