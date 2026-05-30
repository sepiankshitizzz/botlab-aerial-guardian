from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

model = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("input.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "deepsort_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]

    detections = []

    for box, cls, conf in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.conf
    ):

        if int(cls) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())

        detections.append(
            ([x1, y1, x2 - x1, y2 - y1], float(conf), "person")
        )

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)

    cv2.imshow("DeepSORT Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()