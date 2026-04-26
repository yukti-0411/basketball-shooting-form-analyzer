import os
import cv2
import mediapipe as mp
from typing import Any, Dict, Optional

from utils import (
    get_point, angle_at_vertex_deg, draw_angle_highlight,
    draw_text, draw_skeleton, shooting_arm_side
)

mp_pose = mp.solutions.pose
FLARE_FRAC = 0.10


def analyze_release_frame(frame_bgr, landmarks, ball_bbox_xywh, frame_w, frame_h, output_dir=".") -> Dict[str, Any]:
    """
    Analyze the release frame — the moment the ball leaves the hand.
    Measures elbow angle, knee angle, wrist position and elbow alignment.
    Saves annotated frame to output_dir/analyzed_frame.jpg.
    """
    analyzed_frame_path = os.path.join(output_dir, "analyzed_frame.jpg")
    flare_threshold_px = FLARE_FRAC * frame_w

    # Determine which arm is the shooting arm
    bx, by, bw, bh = ball_bbox_xywh
    ball_cx = bx + bw / 2
    ball_cy = by + bh / 2
    side = shooting_arm_side(landmarks, frame_w, frame_h, ball_cx, ball_cy)

    if side is None:
        cv2.imwrite(analyzed_frame_path, frame_bgr)
        return {
            "elbow_angle_deg": None,
            "knee_angle_deg": None,
            "wrist_above_shoulder": None,
            "elbow_offset_px": None,
            "flare_threshold_px": flare_threshold_px,
        }

    # Select landmarks for the shooting side
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

    shoulder = get_point(landmarks, shoulder_i, frame_w, frame_h)
    elbow = get_point(landmarks, elbow_i, frame_w, frame_h)
    wrist = get_point(landmarks, wrist_i, frame_w, frame_h)
    hip = get_point(landmarks, hip_i, frame_w, frame_h)
    knee = get_point(landmarks, knee_i, frame_w, frame_h)
    ankle = get_point(landmarks, ankle_i, frame_w, frame_h)

    frame_out = frame_bgr.copy()
    draw_skeleton(frame_out, landmarks)

    wrist_above: Optional[bool] = None
    elbow_offset_px: Optional[float] = None

    # --- Elbow angle ---
    elbow_angle = None
    if shoulder and elbow and wrist:
        elbow_angle = angle_at_vertex_deg(shoulder, elbow, wrist)
        draw_angle_highlight(frame_out, shoulder, elbow, wrist,
                             f"Elbow {elbow_angle:.0f}")

    # --- Knee angle ---
    knee_angle = None
    if hip and knee and ankle:
        knee_angle = angle_at_vertex_deg(hip, knee, ankle)
        draw_angle_highlight(frame_out, hip, knee, ankle,
                             f"Knee {knee_angle:.0f}")

    # --- Wrist above shoulder ---
    if wrist and shoulder:
        wrist_above = wrist[1] < shoulder[1]
        wrist_label = "Wrist: HIGH" if wrist_above else "Wrist: LOW"
        wrist_color = (0, 255, 0) if wrist_above else (0, 0, 255)
        draw_text(frame_out, wrist_label,
                  (int(wrist[0]) + 10, int(wrist[1])),
                  color=wrist_color)

    # --- Elbow alignment ---
    if elbow and shoulder:
        elbow_offset_px = float(elbow[0] - shoulder[0])
        abs_dx = abs(elbow_offset_px)
        flared = abs_dx > flare_threshold_px
        alignment_label = "FLARED" if flared else "STACKED"
        alignment_color = (0, 0, 255) if flared else (0, 255, 0)
        draw_text(frame_out, alignment_label,
                  (int(shoulder[0]) + 10, int(shoulder[1]) - 10),
                  color=alignment_color)

    cv2.imwrite(analyzed_frame_path, frame_out)

    return {
        "elbow_angle_deg": elbow_angle,
        "knee_angle_deg": knee_angle,
        "wrist_above_shoulder": wrist_above,
        "elbow_offset_px": elbow_offset_px,
        "flare_threshold_px": flare_threshold_px,
    }