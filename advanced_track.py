from ultralytics import YOLO
import cv2
import math

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("input.mp4")

# 🔥 VIDEO SAVE SETUP
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "final_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Store previous positions
tracks = {}
track_id = 0

# Store trajectory history
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

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        center = get_center((x1, y1, x2, y2))

        matched = False

        # Match with old tracks
        for tid, prev_center in tracks.items():
            if distance(center, prev_center) < 100:
                new_tracks[tid] = center
                matched = True

                # Draw box + ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Trajectory
                if tid not in track_history:
                    track_history[tid] = []

                track_history[tid].append(center)

                if len(track_history[tid]) > 20:
                    track_history[tid].pop(0)

                for i in range(1, len(track_history[tid])):
                    cv2.line(frame,
                             track_history[tid][i-1],
                             track_history[tid][i],
                             (255, 0, 0), 2)

                break

        # New ID
        if not matched:
            track_id += 1
            new_tracks[track_id] = center

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            track_history[track_id] = [center]

    # Keep old tracks
    for tid, prev_center in tracks.items():
        if tid not in new_tracks:
            new_tracks[tid] = prev_center

    tracks = new_tracks

    # Show + Save
    cv2.imshow("Tracking", frame)
    out.write(frame)   # 🔥 SAVE FRAME

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()   # 🔥 RELEASE VIDEO
cv2.destroyAllWindows()