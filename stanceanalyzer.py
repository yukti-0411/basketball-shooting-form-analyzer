import os
import logging
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import math
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from angle_analysis import analyze_release_frame
from feedback import generate_feedback


# ----- Constants -----
SPORTS_BALL_CLASS_ID = 32
YOLO_MIN_CONF = 0.25
HSV_ORANGE_LOWER = np.array([8, 80, 80], dtype=np.uint8)
HSV_ORANGE_UPPER = np.array([28, 255, 255], dtype=np.uint8)
MIN_ORANGE_AREA = 400
NEAR_WRIST_FRAC = 0.35
JUMP_FRAC = 0.03
FOLLOWTHROUGH_OFFSET = 4
MIN_VISIBILITY = 0.5
FLARE_FRAC = 0.10

# Load YOLO model once at module level
model = YOLO("yolov8x.pt")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ----- Utility functions -----
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


def lm_px(lm, frame_w, frame_h):
    return lm.x * frame_w, lm.y * frame_h


def get_point(landmarks, idx, frame_w, frame_h):
    lm = landmarks.landmark[idx]
    if lm.visibility < MIN_VISIBILITY:
        return None
    return lm_px(lm, frame_w, frame_h)


def angle_at_vertex_deg(p_prev, p_vertex, p_next):
    ba = np.array(p_prev, dtype=np.float64) - np.array(p_vertex, dtype=np.float64)
    bc = np.array(p_next, dtype=np.float64) - np.array(p_vertex, dtype=np.float64)
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    c = float(np.dot(ba, bc) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def draw_angle_highlight(frame, p1, vertex, p2, label, color=(0, 255, 255)):
    i1 = (int(p1[0]), int(p1[1]))
    iv = (int(vertex[0]), int(vertex[1]))
    i2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, iv, i1, color, 2, cv2.LINE_AA)
    cv2.line(frame, iv, i2, color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (iv[0] + 8, iv[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_line_with_label(frame, p1, p2, label, color=(255, 255, 255)):
    i1 = (int(p1[0]), int(p1[1]))
    i2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, i1, i2, color, 2, cv2.LINE_AA)
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)
    cv2.putText(frame, label, (mid_x + 8, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


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


def shooting_arm_side(landmarks, frame_w, frame_h, ball_cx, ball_cy):
    best_d = None
    best_side = None
    for side, idx in (
        ("left", mp_pose.PoseLandmark.LEFT_WRIST),
        ("right", mp_pose.PoseLandmark.RIGHT_WRIST),
    ):
        lm = landmarks.landmark[idx]
        if lm.visibility < MIN_VISIBILITY:
            continue
        wx, wy = lm_px(lm, frame_w, frame_h)
        d = math.hypot(ball_cx - wx, ball_cy - wy)
        if best_d is None or d < best_d:
            best_d = d
            best_side = side
    return best_side


def get_wrist_elbow_y_diff(landmarks, shooting_side, frame_w, frame_h):
    if shooting_side == "right":
        elbow_i = mp_pose.PoseLandmark.RIGHT_ELBOW
        wrist_i = mp_pose.PoseLandmark.RIGHT_WRIST
    else:
        elbow_i = mp_pose.PoseLandmark.LEFT_ELBOW
        wrist_i = mp_pose.PoseLandmark.LEFT_WRIST
    elbow = get_point(landmarks, elbow_i, frame_w, frame_h)
    wrist = get_point(landmarks, wrist_i, frame_w, frame_h)
    if elbow and wrist:
        return wrist[1] - elbow[1]
    return None


def analyze_load_frame(frame, landmarks, frame_width, frame_height, output_dir):
    frame_out = frame.copy()
    mp_drawing.draw_landmarks(
        frame_out, landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
    )

    knee_angle_left = None
    knee_angle_right = None

    left_hip = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP, frame_width, frame_height)
    left_knee = get_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, frame_width, frame_height)
    left_ankle = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, frame_width, frame_height)
    right_hip = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, frame_width, frame_height)
    right_knee = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, frame_width, frame_height)
    right_ankle = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, frame_width, frame_height)

    if left_hip and left_knee and left_ankle:
        knee_angle_left = angle_at_vertex_deg(left_hip, left_knee, left_ankle)
        draw_angle_highlight(frame_out, left_hip, left_knee, left_ankle,
                             f"Knee {knee_angle_left:.0f} deg")

    if right_hip and right_knee and right_ankle:
        knee_angle_right = angle_at_vertex_deg(right_hip, right_knee, right_ankle)
        draw_angle_highlight(frame_out, right_hip, right_knee, right_ankle,
                             f"Knee {knee_angle_right:.0f} deg")

    elbow_angle = None
    for side in [
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    ]:
        shoulder = get_point(landmarks, side[0], frame_width, frame_height)
        elbow = get_point(landmarks, side[1], frame_width, frame_height)
        wrist = get_point(landmarks, side[2], frame_width, frame_height)
        if shoulder and elbow and wrist:
            a = angle_at_vertex_deg(shoulder, elbow, wrist)
            if a is not None:
                elbow_angle = a
                draw_angle_highlight(frame_out, shoulder, elbow, wrist,
                                     f"Elbow {a:.0f} deg", color=(255, 165, 0))
                break

    hip_square = None
    if left_hip and right_hip:
        hip_y_diff = abs(left_hip[1] - right_hip[1])
        hip_square = hip_y_diff < (0.05 * frame_height)
        draw_line_with_label(frame_out, left_hip, right_hip,
                             f"Hips {'Square' if hip_square else 'Tilted'}", color=(0, 165, 255))

    balance_ok = None
    if left_hip and right_hip and left_ankle and right_ankle:
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
        balance_offset = abs(hip_mid_x - ankle_mid_x)
        balance_ok = balance_offset < (0.08 * frame_width)

    load_image_path = os.path.join(output_dir, "load_frame.jpg")
    cv2.imwrite(load_image_path, frame_out)

    return {
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "elbow_angle_deg": elbow_angle,
        "hip_square": hip_square,
        "ball_height_ok": None,
        "balance_ok": balance_ok,
    }


def analyze_followthrough_frame(frame, landmarks, frame_width, frame_height,
                                 shooting_side, wrist_elbow_diff_at_release, output_dir):
    frame_out = frame.copy()
    mp_drawing.draw_landmarks(
        frame_out, landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
    )

    if shooting_side == "right":
        shoulder_i = mp_pose.PoseLandmark.RIGHT_SHOULDER
        elbow_i = mp_pose.PoseLandmark.RIGHT_ELBOW
        wrist_i = mp_pose.PoseLandmark.RIGHT_WRIST
    else:
        shoulder_i = mp_pose.PoseLandmark.LEFT_SHOULDER
        elbow_i = mp_pose.PoseLandmark.LEFT_ELBOW
        wrist_i = mp_pose.PoseLandmark.LEFT_WRIST

    left_hip = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP, frame_width, frame_height)
    right_hip = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, frame_width, frame_height)
    left_ankle = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, frame_width, frame_height)
    right_ankle = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, frame_width, frame_height)

    shoulder = get_point(landmarks, shoulder_i, frame_width, frame_height)
    elbow = get_point(landmarks, elbow_i, frame_width, frame_height)
    wrist = get_point(landmarks, wrist_i, frame_width, frame_height)

    elbow_angle = None
    if shoulder and elbow and wrist:
        elbow_angle = angle_at_vertex_deg(shoulder, elbow, wrist)
        cv2.line(frame_out, (int(shoulder[0]), int(shoulder[1])),
                 (int(elbow[0]), int(elbow[1])), (255, 165, 0), 2, cv2.LINE_AA)
        cv2.line(frame_out, (int(wrist[0]), int(wrist[1])),
                 (int(elbow[0]), int(elbow[1])), (255, 165, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_out, f"Elbow {elbow_angle:.0f} deg",
                    (int(elbow[0]) + 10, int(elbow[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2, cv2.LINE_AA)

    wrist_snapped = None
    if wrist and elbow and wrist_elbow_diff_at_release is not None:
        wrist_elbow_diff_now = wrist[1] - elbow[1]
        wrist_snapped = wrist_elbow_diff_now > wrist_elbow_diff_at_release
        snap_text = "Snap: YES" if wrist_snapped else "Snap: NO"
        snap_color = (0, 255, 0) if wrist_snapped else (0, 0, 255)
        cv2.putText(frame_out, snap_text,
                    (int(wrist[0]) + 10, int(wrist[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, snap_color, 2, cv2.LINE_AA)

    balance_ok = None
    if left_hip and right_hip and left_ankle and right_ankle:
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
        balance_offset = abs(hip_mid_x - ankle_mid_x)
        balance_ok = balance_offset < (0.08 * frame_width)
        balance_text = "Balance: OK" if balance_ok else "Balance: LEANING"
        balance_color = (0, 255, 0) if balance_ok else (0, 0, 255)
        cv2.putText(frame_out, balance_text,
                    (20, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, balance_color, 2, cv2.LINE_AA)

    followthrough_image_path = os.path.join(output_dir, "followthrough_frame.jpg")
    cv2.imwrite(followthrough_image_path, frame_out)

    return {
        "wrist_snapped": wrist_snapped,
        "elbow_angle_deg": elbow_angle,
        "balance_ok": balance_ok,
    }


def run_analysis(video_path, output_dir, groq_api_key=None):
    """
    Main analysis function called by Flask.
    Returns dict with all metrics, coaching text, and image paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    release_image_path = os.path.join(output_dir, "release_frame.jpg")

    # ----- Pass 1: find ball -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        return {"error": "No basketball detected in video."}

    min_dim = min(frame_width, frame_height)
    near_wrist_max = NEAR_WRIST_FRAC * min_dim
    jump_min = JUMP_FRAC * min_dim

    # ----- Pass 2: detect load + release + follow through -----
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
    wrist_elbow_diff_at_release = None

    best_knee_angle = float("inf")
    best_load_frame = None
    best_load_landmarks = None
    best_load_frame_number = 0
    followthrough_frame_data = {}

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
            if frame_number == init_frame_index:
                if tracker.init(clean, init_bbox):
                    ball_bbox_xywh = init_bbox
                    tracker_initialized = True
            elif tracker_initialized and frame_number > init_frame_index:
                success, bbox = tracker.update(clean)
                if success:
                    bx, by, bw, bh = bbox
                    ball_bbox_xywh = (bx, by, bw, bh)

            if pose_results.pose_landmarks and not release_found:
                knee_angle = get_min_knee_angle(pose_results.pose_landmarks, frame_width, frame_height)
                if knee_angle is not None and knee_angle < best_knee_angle:
                    best_knee_angle = knee_angle
                    best_load_frame = frame.copy()
                    best_load_frame_number = frame_number
                    best_load_landmarks = pose_results.pose_landmarks

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
                not release_found
                and dist is not None
                and prev_dist is not None
                and prev_dist < near_wrist_max
                and (dist - prev_dist) > jump_min
            ):
                cv2.imwrite(release_image_path, frame)
                release_frame_number = frame_number

                if pose_results.pose_landmarks and ball_bbox_xywh:
                    bx, by, bw, bh = ball_bbox_xywh
                    release_shooting_side = shooting_arm_side(
                        pose_results.pose_landmarks,
                        frame_width, frame_height,
                        bx + bw / 2, by + bh / 2
                    )
                    release_landmarks_stored = pose_results.pose_landmarks
                    release_ball_bbox_stored = ball_bbox_xywh
                    wrist_elbow_diff_at_release = get_wrist_elbow_y_diff(
                        pose_results.pose_landmarks,
                        release_shooting_side or "right",
                        frame_width, frame_height
                    )

                release_found = True

            if release_found and release_frame_number is not None:
                if frame_number == release_frame_number + FOLLOWTHROUGH_OFFSET:
                    if pose_results.pose_landmarks:
                        followthrough_frame_data = {
                            "frame": frame.copy(),
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

    # ----- Run analyses -----
    load_metrics = analyze_load_frame(
        best_load_frame, best_load_landmarks, frame_width, frame_height, output_dir
    )

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
                        from angle_analysis import ANALYZED_FRAME_PATH
                        import angle_analysis
                        angle_analysis.ANALYZED_FRAME_PATH = os.path.join(output_dir, "analyzed_frame.jpg")
                        release_metrics = analyze_release_frame(
                            frame, pose_results.pose_landmarks,
                            release_ball_bbox_stored, frame_width, frame_height
                        )
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
            wrist_elbow_diff_at_release,
            output_dir
        )

    if not release_metrics:
        return {"error": "Could not analyze release frame."}

    coaching = generate_feedback(
        release_metrics=release_metrics,
        load_metrics=load_metrics,
        followthrough_metrics=followthrough_metrics,
        api_key=groq_api_key,
    )

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