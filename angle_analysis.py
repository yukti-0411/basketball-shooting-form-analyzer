import math
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
ANALYZED_FRAME_PATH = "analyzed_frame.jpg"
FLARE_FRAC = 0.10
MIN_VISIBILITY = 0.5


def _lm_px(lm, frame_w, frame_h):
    return lm.x * frame_w, lm.y * frame_h


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
        wx, wy = _lm_px(lm, frame_w, frame_h)
        d = math.hypot(ball_cx - wx, ball_cy - wy)
        if best_d is None or d < best_d:
            best_d = d
            best_side = side
    return best_side


def _angle_at_vertex_deg(p_prev, p_vertex, p_next):
    ba = np.array(p_prev, dtype=np.float64) - np.array(p_vertex, dtype=np.float64)
    bc = np.array(p_next, dtype=np.float64) - np.array(p_vertex, dtype=np.float64)
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    c = float(np.dot(ba, bc) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _get_point(landmarks, idx, frame_w, frame_h):
    lm = landmarks.landmark[idx]
    if lm.visibility < MIN_VISIBILITY:
        return None
    return _lm_px(lm, frame_w, frame_h)


def _draw_angle_highlight(frame, p1, vertex, p2, label, color=(0, 255, 255)):
    i1 = (int(p1[0]), int(p1[1]))
    iv = (int(vertex[0]), int(vertex[1]))
    i2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, iv, i1, color, 2, cv2.LINE_AA)
    cv2.line(frame, iv, i2, color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (iv[0] + 8, iv[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def analyze_release_frame(
    frame_bgr, landmarks, ball_bbox_xywh, frame_w, frame_h
) -> Dict[str, Any]:

    flare_threshold_px = FLARE_FRAC * frame_w

    bx, by, bw, bh = ball_bbox_xywh
    ball_cx = bx + bw / 2
    ball_cy = by + bh / 2
    side = shooting_arm_side(landmarks, frame_w, frame_h, ball_cx, ball_cy)

    if side is None:
        print("\n--- Angle analysis ---")
        print("Could not determine shooting arm.")
        cv2.imwrite(ANALYZED_FRAME_PATH, frame_bgr)
        return {
            "elbow_angle_deg": None,
            "knee_angle_deg": None,
            "wrist_above_shoulder": None,
            "elbow_offset_px": None,
            "flare_threshold_px": flare_threshold_px,
        }

    if side == "right":
        shoulder_i = mp_pose.PoseLandmark.RIGHT_SHOULDER
        elbow_i = mp_pose.PoseLandmark.RIGHT_ELBOW
        wrist_i = mp_pose.PoseLandmark.RIGHT_WRIST
        hip_i = mp_pose.PoseLandmark.RIGHT_HIP
        knee_i = mp_pose.PoseLandmark.RIGHT_KNEE
        ankle_i = mp_pose.PoseLandmark.RIGHT_ANKLE
    else:
        shoulder_i = mp_pose.PoseLandmark.LEFT_SHOULDER
        elbow_i = mp_pose.PoseLandmark.LEFT_ELBOW
        wrist_i = mp_pose.PoseLandmark.LEFT_WRIST
        hip_i = mp_pose.PoseLandmark.LEFT_HIP
        knee_i = mp_pose.PoseLandmark.LEFT_KNEE
        ankle_i = mp_pose.PoseLandmark.LEFT_ANKLE

    shoulder = _get_point(landmarks, shoulder_i, frame_w, frame_h)
    elbow = _get_point(landmarks, elbow_i, frame_w, frame_h)
    wrist = _get_point(landmarks, wrist_i, frame_w, frame_h)
    hip = _get_point(landmarks, hip_i, frame_w, frame_h)
    knee = _get_point(landmarks, knee_i, frame_w, frame_h)
    ankle = _get_point(landmarks, ankle_i, frame_w, frame_h)

    wrist_above: Optional[bool] = None
    elbow_offset_px: Optional[float] = None

    print("\n--- Angle analysis (release frame) ---")

    elbow_angle = None
    if shoulder and elbow and wrist:
        elbow_angle = _angle_at_vertex_deg(shoulder, elbow, wrist)
        print(f"Elbow angle: {elbow_angle:.1f} deg")
        _draw_angle_highlight(frame_bgr, shoulder, elbow, wrist,
                              f"Elbow {elbow_angle:.0f} deg")
    else:
        print("Elbow angle: N/A")

    knee_angle = None
    if hip and knee and ankle:
        knee_angle = _angle_at_vertex_deg(hip, knee, ankle)
        print(f"Knee angle: {knee_angle:.1f} deg")
        _draw_angle_highlight(frame_bgr, hip, knee, ankle,
                              f"Knee {knee_angle:.0f} deg")
    else:
        print("Knee angle: N/A")

    if wrist and shoulder:
        wrist_above = wrist[1] < shoulder[1]
        print("Wrist above shoulder: " + ("YES" if wrist_above else "NO"))
    else:
        print("Wrist vs shoulder: N/A")

    if elbow and shoulder:
        elbow_offset_px = float(elbow[0] - shoulder[0])
        abs_dx = abs(elbow_offset_px)
        flared = abs_dx > flare_threshold_px
        print(f"Elbow offset: {abs_dx:.1f}px — {'FLARED' if flared else 'STACKED'}")
    else:
        print("Elbow flare: N/A")

    cv2.imwrite(ANALYZED_FRAME_PATH, frame_bgr)
    print(f"Saved analyzed frame: {ANALYZED_FRAME_PATH}")

    return {
        "elbow_angle_deg": elbow_angle,
        "knee_angle_deg": knee_angle,
        "wrist_above_shoulder": wrist_above,
        "elbow_offset_px": elbow_offset_px,
        "flare_threshold_px": flare_threshold_px,
    }