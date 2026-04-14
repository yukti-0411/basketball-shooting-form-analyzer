import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


# ----- Video paths -----
input_video_path = "test_video1.mp4"
output_video_path = "output_combined.mp4"

# ----- Ball detection (YOLO + HSV fallback) -----
SPORTS_BALL_CLASS_ID = 32
YOLO_MIN_CONF = 0.25
HSV_ORANGE_LOWER = np.array([8, 80, 80], dtype=np.uint8)
HSV_ORANGE_UPPER = np.array([28, 255, 255], dtype=np.uint8)
MIN_ORANGE_AREA = 400

# YOLOv8 extra-large — accurate but slower; downloads on first run.
model = YOLO("yolov8x.pt")

# MediaPipe shortcuts
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def xyxy_to_xywh(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def draw_ball_circle(frame_img, bx, by, bw, bh):
    cx = int(bx + bw / 2)
    cy = int(by + bh / 2)
    r = int(min(bw, bh) / 2)
    if r < 1:
        return
    cv2.circle(frame_img, (cx, cy), r, (0, 255, 0), 2)


def yolo_best_sports_ball(frame):
    results = model(frame, verbose=False)
    result = results[0]
    best_xywh = None
    best_conf = -1.0
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id != SPORTS_BALL_CLASS_ID:
            continue
        conf = float(box.conf[0].item())
        if conf < YOLO_MIN_CONF:
            continue
        if conf > best_conf:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, w, h = xyxy_to_xywh(x1, y1, x2, y2)
            if w >= 2 and h >= 2:
                best_xywh = (x1, y1, w, h)
                best_conf = conf
    return best_xywh


def hsv_orange_bbox(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_ORANGE_LOWER, HSV_ORANGE_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_ORANGE_AREA:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    if w < 2 or h < 2:
        return None
    return x, y, w, h


def find_ball_init_scan(cap, use_yolo):
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return None, None
        if use_yolo:
            box = yolo_best_sports_ball(frame)
        else:
            box = hsv_orange_bbox(frame)
        if box is not None:
            return index, box
        index += 1


def create_csrt():
    return cv2.legacy.TrackerCSRT_create()


# ----- Pass 1: find first frame where we can start ball tracking -----
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open input video: {input_video_path}")
    raise SystemExit

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

init_frame_index, init_bbox = find_ball_init_scan(cap, use_yolo=True)
cap.release()

if init_frame_index is None:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not reopen video for HSV fallback.")
        raise SystemExit
    init_frame_index, init_bbox = find_ball_init_scan(cap, use_yolo=False)
    cap.release()

# ----- Pass 2: pose + ball on every frame, one output video -----
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video for combined output pass.")
    raise SystemExit

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

tracker = None
frame_number = 0

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Tracker must see raw video pixels only (no skeleton or circle drawn yet).
        clean = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        ball_bbox_xywh = None
        if init_bbox is not None:
            if frame_number < init_frame_index:
                pass
            elif frame_number == init_frame_index:
                tracker = create_csrt()
                x, y, w, h = init_bbox
                if tracker.init(clean, (x, y, w, h)):
                    ball_bbox_xywh = (x, y, w, h)
                else:
                    print("Warning: CSRT init failed; ball circle disabled.")
                    tracker = None
            elif tracker is not None:
                success, bbox = tracker.update(clean)
                if success:
                    bx, by, bw, bh = bbox
                    ball_bbox_xywh = (bx, by, bw, bh)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2
                ),
            )

        if ball_bbox_xywh is not None:
            bx, by, bw, bh = ball_bbox_xywh
            draw_ball_circle(frame, bx, by, bw, bh)

        out.write(frame)
        frame_number += 1

cap.release()
out.release()

if init_bbox is None:
    print("Warning: No ball found (YOLO or HSV). Output has skeleton only.")
else:
    print(f"Done! Saved: {output_video_path} (skeleton + ball from frame {init_frame_index})")
