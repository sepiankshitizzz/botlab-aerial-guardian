from ultralytics import YOLO
import cv2
import math

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("input.mp4")

# VIDEO SAVE SETUP
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "final_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

tracks = {}
track_id = 0
track_history = {}

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]

    new_tracks = {}

    # PERSON ONLY TRACKING
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):

        # YOLO class 0 = person
        if int(cls) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())

        center = get_center((x1, y1, x2, y2))

        matched = False

        for tid, prev_center in tracks.items():

            if distance(center, prev_center) < 100:

                new_tracks[tid] = center
                matched = True

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID {tid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                if tid not in track_history:
                    track_history[tid] = []

                track_history[tid].append(center)

                if len(track_history[tid]) > 20:
                    track_history[tid].pop(0)

                for i in range(1, len(track_history[tid])):
                    cv2.line(
                        frame,
                        track_history[tid][i - 1],
                        track_history[tid][i],
                        (255, 0, 0),
                        2
                    )

                break

        if not matched:

            track_id += 1

            new_tracks[track_id] = center

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            track_history[track_id] = [center]

    for tid, prev_center in tracks.items():
        if tid not in new_tracks:
            new_tracks[tid] = prev_center

    tracks = new_tracks

    cv2.imshow("Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()