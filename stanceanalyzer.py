import os
import math
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from utils import (
    get_point, distance_ball_to_closest_wrist,
    shooting_arm_side, draw_skeleton
)
from load_analysis import analyze_load_frame
from release_analysis import analyze_release_frame
from followthrough_analysis import analyze_followthrough_frame
from feedback import generate_feedback


# --- Constants ---
SPORTS_BALL_CLASS_ID = 32
YOLO_MIN_CONF = 0.25
HSV_ORANGE_LOWER = np.array([8, 80, 80], dtype=np.uint8)
HSV_ORANGE_UPPER = np.array([28, 255, 255], dtype=np.uint8)
MIN_ORANGE_AREA = 400
NEAR_WRIST_FRAC = 0.35
JUMP_FRAC = 0.03
STANDARD_WIDTH = 720  # All videos resized to this width for consistent text rendering

# YOLO model loaded once at import time
model = YOLO("yolov8x.pt")
mp_pose = mp.solutions.pose


# --- Ball detection ---

def yolo_best_sports_ball(frame):
    """Run YOLO on a frame and return the highest confidence sports ball bounding box, or None."""
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
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            if w >= 2 and h >= 2:
                best_xywh = (x1, y1, w, h)
                best_conf = conf
    return best_xywh


def hsv_orange_bbox(frame):
    """Fallback ball detection using HSV color filtering and circularity check."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_ORANGE_LOWER, HSV_ORANGE_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = None
    best_score = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_ORANGE_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 2:
            continue
        aspect_ratio = w / h
        if not (0.4 <= aspect_ratio <= 2.5):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < 0.5:
            continue
        score = area * circularity
        if score > best_score:
            best = (x, y, w, h)
            best_score = score
    return best


def get_min_knee_angle(landmarks, frame_w, frame_h):
    """Return the smaller of the two knee angles — used to track the load position."""
    from utils import angle_at_vertex_deg
    angles = []
    for side in [
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    ]:
        hip = get_point(landmarks, side[0], frame_w, frame_h)
        knee = get_point(landmarks, side[1], frame_w, frame_h)
        ankle = get_point(landmarks, side[2], frame_w, frame_h)
        if hip and knee and ankle:
            a = angle_at_vertex_deg(hip, knee, ankle)
            if a is not None:
                angles.append(a)
    if not angles:
        return None
    return min(angles)


def slow_down_video(input_path, output_path, speed=0.5):
    """Write a copy of the video at reduced FPS to create slow motion effect."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_fps = max(1, fps * speed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    cap.release()
    out.release()
    return output_path


def standardize_video(input_path, output_path):
    """Resize video to standard width so text and annotations are consistent across resolutions."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w == STANDARD_WIDTH:
        cap.release()
        return input_path
    scale = STANDARD_WIDTH / orig_w
    new_h = int(orig_h * scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (STANDARD_WIDTH, new_h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (STANDARD_WIDTH, new_h))
        out.write(resized)
    cap.release()
    out.release()
    return output_path


# --- Main pipeline ---

def run_analysis(video_path, output_dir, groq_api_key=None, progress_callback=None, speed=1.0):
    """
    Full shooting form analysis pipeline.
    Detects ball, tracks load position and release point, analyzes all three frames,
    generates AI coaching. Calls progress_callback at each real milestone for SSE streaming.
    """

    def progress(msg):
        if progress_callback:
            progress_callback(msg)

    os.makedirs(output_dir, exist_ok=True)

    # Slow down video if requested
    if speed < 1.0:
        slowed_path = os.path.join(output_dir, "slowed_video.mp4")
        video_path = slow_down_video(video_path, slowed_path, speed)

    # Standardize resolution so text is consistent across all videos
    standard_path = os.path.join(output_dir, "standard_video.mp4")
    video_path = standardize_video(video_path, standard_path)

    release_image_path = os.path.join(output_dir, "release_frame.jpg")

    # --- Pass 1: Find ball using YOLO ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FPS based follow through offset — always 0.15 seconds after release
    followthrough_offset = max(1, int(fps * 0.15))

    init_frame_index, init_bbox = None, None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        box = yolo_best_sports_ball(frame)
        if box is not None:
            init_frame_index = index
            init_bbox = box
            break
        index += 1
    cap.release()

    # Fallback to HSV color detection if YOLO found nothing
    if init_frame_index is None:
        cap = cv2.VideoCapture(video_path)
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            box = hsv_orange_bbox(frame)
            if box is not None:
                init_frame_index = index
                init_bbox = box
                break
            index += 1
        cap.release()

    if init_bbox is None:
        return {"error": "No basketball detected. Ensure ball is clearly visible."}

    progress("step:1:Basketball located")

    # Calculate release detection thresholds in pixels
    min_dim = min(frame_width, frame_height)
    near_wrist_max = NEAR_WRIST_FRAC * min_dim
    jump_min = JUMP_FRAC * min_dim

    # --- Pass 2: Single pass — track load, detect release, collect follow through ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not reopen video for processing."}

    tracker = cv2.legacy.TrackerCSRT_create()
    tracker_initialized = False
    frame_number = 0
    prev_dist = None
    release_found = False
    release_frame_number = None
    release_shooting_side = None
    release_landmarks_stored = None
    release_ball_bbox_stored = None

    # Load position tracking — updated every frame until release
    best_knee_angle = float("inf")
    best_load_frame = None
    best_load_landmarks = None
    best_load_frame_number = 0

    followthrough_frame_data = {}
    pose_detected = False

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Keep clean copy for tracker — never pass drawn frames to tracker
            clean = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)

            # --- Ball tracking ---
            ball_bbox_xywh = None
            if frame_number == init_frame_index:
                if tracker.init(clean, init_bbox):
                    ball_bbox_xywh = init_bbox
                    tracker_initialized = True
            elif tracker_initialized and frame_number > init_frame_index:
                success, bbox = tracker.update(clean)
                if success:
                    bx, by, bw, bh = bbox
                    ball_bbox_xywh = (bx, by, bw, bh)

            # Fire step 2 exactly once when pose is first detected
            if pose_results.pose_landmarks and not pose_detected:
                pose_detected = True
                progress("step:2:Body pose detected")

            # --- Track load position — frame with minimum knee angle before release ---
            if pose_results.pose_landmarks and not release_found:
                knee_angle = get_min_knee_angle(pose_results.pose_landmarks, frame_width, frame_height)
                if knee_angle is not None and knee_angle < best_knee_angle:
                    best_knee_angle = knee_angle
                    best_load_frame = frame.copy()
                    best_load_frame_number = frame_number
                    best_load_landmarks = pose_results.pose_landmarks

            # --- Calculate ball to wrist distance for release detection ---
            dist = None
            if ball_bbox_xywh is not None and pose_results.pose_landmarks:
                bx, by, bw, bh = ball_bbox_xywh
                ball_cx = bx + bw / 2
                ball_cy = by + bh / 2
                dist = distance_ball_to_closest_wrist(
                    pose_results.pose_landmarks, frame_width, frame_height, ball_cx, ball_cy)

            # Draw skeleton on frame after clean copy saved
            if pose_results.pose_landmarks:
                draw_skeleton(frame, pose_results.pose_landmarks)

            # Save skeleton-only frame — used for release and follow through (no ball circle)
            skeleton_only_frame = frame.copy()

            # --- Release detection — ball suddenly jumps away from wrist ---
            if (not release_found and dist is not None and prev_dist is not None
                    and prev_dist < near_wrist_max and (dist - prev_dist) > jump_min):

                cv2.imwrite(release_image_path, skeleton_only_frame)
                release_frame_number = frame_number
                progress("step:3:Release point detected")

                if pose_results.pose_landmarks and ball_bbox_xywh:
                    bx, by, bw, bh = ball_bbox_xywh
                    release_shooting_side = shooting_arm_side(
                        pose_results.pose_landmarks, frame_width, frame_height,
                        bx + bw / 2, by + bh / 2)
                    release_landmarks_stored = pose_results.pose_landmarks
                    release_ball_bbox_stored = ball_bbox_xywh

                release_found = True

            # --- Collect follow through frame at FPS based offset after release ---
            if release_found and release_frame_number is not None:
                if frame_number == release_frame_number + followthrough_offset:
                    if pose_results.pose_landmarks:
                        followthrough_frame_data = {
                            "frame": skeleton_only_frame.copy(),
                            "landmarks": pose_results.pose_landmarks,
                        }
                    break

            if dist is not None:
                prev_dist = dist
            else:
                prev_dist = None
            frame_number += 1

    cap.release()

    if not release_found:
        return {"error": "No release point detected. Ensure ball and player are clearly visible."}

    # --- Run all three analyses ---
    load_metrics = analyze_load_frame(
        best_load_frame, best_load_landmarks,
        frame_width, frame_height, output_dir)

    # Re-open video to get clean release frame for analysis
    release_metrics = None
    if release_landmarks_stored and release_ball_bbox_stored:
        cap = cv2.VideoCapture(video_path)
        fn = 0
        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if fn == release_frame_number:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(frame_rgb)
                    if pose_results.pose_landmarks:
                        release_metrics = analyze_release_frame(
                            frame, pose_results.pose_landmarks,
                            release_ball_bbox_stored, frame_width, frame_height,
                            output_dir)
                    break
                fn += 1
        cap.release()

    followthrough_metrics = None
    if followthrough_frame_data:
        followthrough_metrics = analyze_followthrough_frame(
            followthrough_frame_data["frame"],
            followthrough_frame_data["landmarks"],
            frame_width, frame_height,
            release_shooting_side or "right",
            output_dir)

    # Step 4 fires after all three analyses are complete
    progress("step:4:Angles calculated")

    if not release_metrics:
        return {"error": "Could not analyze release frame."}

    coaching = generate_feedback(
        release_metrics=release_metrics,
        load_metrics=load_metrics,
        followthrough_metrics=followthrough_metrics,
        api_key=groq_api_key)

    progress("step:5:AI coaching generated")

    return {
        "load_metrics": load_metrics,
        "release_metrics": release_metrics,
        "followthrough_metrics": followthrough_metrics,
        "coaching": coaching,
        "load_frame_number": best_load_frame_number,
        "release_frame_number": release_frame_number,
        "images": {
            "load": "load_frame.jpg",
            "release": "release_frame.jpg",
            "analyzed": "analyzed_frame.jpg",
            "followthrough": "followthrough_frame.jpg",
        }
    }