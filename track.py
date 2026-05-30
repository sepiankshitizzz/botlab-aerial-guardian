from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.track(
    source="input.mp4",
    tracker="bytetrack.yaml",
    save=True,
    persist=True,
    conf=0.5,      # 🔥 increase confidence
    iou=0.5,       # 🔥 better matching
    project="./runs_track",
    name="exp"
)