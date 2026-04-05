import math

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from angle_analysis import analyze_release_frame
from feedback import generate_feedback
from load_position import detect_load_position


# ----- Video and output -----
input_video_path = "test_video1.mp4"
release_image_path = "release_frame.jpg"

# ----- Ball detection -----
SPORTS_BALL_CLASS_ID = 32
YOLO_MIN_CONF = 0.25
HSV_ORANGE_LOWER = np.array([8, 80, 80], dtype=np.uint8)
HSV_ORANGE_UPPER = np.array([28, 255, 255], dtype=np.uint8)
MIN_ORANGE_AREA = 400
model = YOLO("yolov8x.pt")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

NEAR_WRIST_FRAC = 0.35
JUMP_FRAC = 0.03


def xyxy_to_xywh(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return x1, y1, x2 - x1, y2 - y1


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
        box = yolo_best_sports_ball(frame) if use_yolo else hsv_orange_bbox(frame)
        if box is not None:
            return index, box
        index += 1


def create_csrt():
    return cv2.legacy.TrackerCSRT_create()


def distance_ball_to_closest_wrist(landmarks, frame_w, frame_h, ball_cx, ball_cy):
    best = None
    for idx in (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST):
        lm = landmarks.landmark[idx]
        if lm.visibility < 0.5:
            continue
        wx = lm.x * frame_w
        wy = lm.y * frame_h
        d = math.hypot(ball_cx - wx, ball_cy - wy)
        if best is None or d < best:
            best = d
    return best


# ----- Pass 1: find ball -----
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open video: {input_video_path}")
    raise SystemExit

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

init_frame_index, init_bbox = find_ball_init_scan(cap, use_yolo=True)
cap.release()

if init_frame_index is None:
    cap = cv2.VideoCapture(input_video_path)
    init_frame_index, init_bbox = find_ball_init_scan(cap, use_yolo=False)
    cap.release()

if init_bbox is None:
    print("Error: No ball found.")
    raise SystemExit

min_dim = min(frame_width, frame_height)
near_wrist_max = NEAR_WRIST_FRAC * min_dim
jump_min = JUMP_FRAC * min_dim

# ----- Pass 2: detect release -----
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video for processing.")
    raise SystemExit

tracker = None
frame_number = 0
prev_dist = None
release_found = False
release_frame_number = None

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

        clean = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        ball_bbox_xywh = None
        if frame_number < init_frame_index:
            pass
        elif frame_number == init_frame_index:
            tracker = create_csrt()
            x, y, w, h = init_bbox
            if tracker.init(clean, (x, y, w, h)):
                ball_bbox_xywh = (x, y, w, h)
            else:
                print("Error: CSRT init failed.")
                tracker = None
        elif tracker is not None:
            success, bbox = tracker.update(clean)
            if success:
                bx, by, bw, bh = bbox
                ball_bbox_xywh = (bx, by, bw, bh)

        dist = None
        if ball_bbox_xywh is not None and pose_results.pose_landmarks:
            bx, by, bw, bh = ball_bbox_xywh
            ball_cx = bx + bw / 2
            ball_cy = by + bh / 2
            dist = distance_ball_to_closest_wrist(
                pose_results.pose_landmarks,
                frame_width, frame_height, ball_cx, ball_cy,
            )

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

        if ball_bbox_xywh is not None:
            bx, by, bw, bh = ball_bbox_xywh
            draw_ball_circle(frame, bx, by, bw, bh)

        if (
            dist is not None
            and prev_dist is not None
            and prev_dist < near_wrist_max
            and (dist - prev_dist) > jump_min
        ):
            cv2.imwrite(release_image_path, frame)
            release_frame_number = frame_number
            print(f"Release point frame number: {frame_number}")
            print(f"Saved image: {release_image_path}")

            if pose_results.pose_landmarks is not None and ball_bbox_xywh is not None:
                # Analyze release frame
                release_metrics = analyze_release_frame(
                    frame,
                    pose_results.pose_landmarks,
                    ball_bbox_xywh,
                    frame_width,
                    frame_height,
                )

                # Detect load position now that we know release frame number
                print("\nDetecting load position...")
                load_metrics = detect_load_position(
                    input_video_path,
                    release_frame_number=frame_number,
                )

                # Generate combined feedback
                generate_feedback(
                    release_metrics=release_metrics,
                    load_metrics=load_metrics,
                )
            else:
                print("Skipping analysis — missing pose or ball data.")

            release_found = True
            break

        if dist is not None:
            prev_dist = dist
        else:
            prev_dist = None

        frame_number += 1

cap.release()

if not release_found:
    print("No release spike found. Try lowering NEAR_WRIST_FRAC or JUMP_FRAC.")