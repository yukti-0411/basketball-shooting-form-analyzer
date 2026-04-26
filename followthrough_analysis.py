import os
import cv2
import mediapipe as mp

from utils import (
    get_point, angle_at_vertex_deg, draw_angle_highlight,
    draw_text, draw_skeleton
)

mp_pose = mp.solutions.pose


def analyze_followthrough_frame(frame, landmarks, frame_width, frame_height,
                                 shooting_side, output_dir):
    """
    Analyze the follow through frame — a few frames after release.
    Checks elbow extension and body balance after the shot.
    Saves annotated frame to output_dir/followthrough_frame.jpg.
    """
    frame_out = frame.copy()
    draw_skeleton(frame_out, landmarks)

    # Select shooting arm landmarks
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

    # --- Elbow angle at follow through — should be close to 180 for full extension ---
    elbow_angle = None
    if shoulder and elbow and wrist:
        elbow_angle = angle_at_vertex_deg(shoulder, elbow, wrist)
        draw_angle_highlight(frame_out, shoulder, elbow, wrist,
                             f"Elbow {elbow_angle:.0f}", color=(255, 165, 0))

    # --- Body balance after release ---
    balance_ok = None
    if left_hip and right_hip and left_ankle and right_ankle:
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
        balance_ok = abs(hip_mid_x - ankle_mid_x) < (0.08 * frame_width)
        balance_text = "Balance: OK" if balance_ok else "Balance: LEANING"
        balance_color = (0, 255, 0) if balance_ok else (0, 0, 255)
        draw_text(frame_out, balance_text,
                  (20, frame_height - 30),
                  color=balance_color)

    cv2.imwrite(os.path.join(output_dir, "followthrough_frame.jpg"), frame_out)

    return {
        "elbow_angle_deg": elbow_angle,
        "balance_ok": balance_ok,
    }