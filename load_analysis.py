import os
import cv2
import mediapipe as mp

from utils import (
    get_point, angle_at_vertex_deg, draw_angle_highlight,
    draw_line_with_label, draw_text, draw_skeleton
)

mp_pose = mp.solutions.pose


def analyze_load_frame(frame, landmarks, frame_width, frame_height, output_dir):
    """
    Analyze the load position frame — maximum knee bend before the shot.
    Measures both knee angles, elbow angle, hip squareness and body balance.
    Saves annotated frame to output_dir/load_frame.jpg.
    """
    frame_out = frame.copy()
    draw_skeleton(frame_out, landmarks)

    # --- Get all required landmarks ---
    left_hip = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP, frame_width, frame_height)
    left_knee = get_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, frame_width, frame_height)
    left_ankle = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, frame_width, frame_height)
    right_hip = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, frame_width, frame_height)
    right_knee = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, frame_width, frame_height)
    right_ankle = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, frame_width, frame_height)

    # --- Knee angles ---
    knee_angle_left = None
    knee_angle_right = None

    if left_hip and left_knee and left_ankle:
        knee_angle_left = angle_at_vertex_deg(left_hip, left_knee, left_ankle)
        draw_angle_highlight(frame_out, left_hip, left_knee, left_ankle,
                             f"Knee {knee_angle_left:.0f}")

    if right_hip and right_knee and right_ankle:
        knee_angle_right = angle_at_vertex_deg(right_hip, right_knee, right_ankle)
        draw_angle_highlight(frame_out, right_hip, right_knee, right_ankle,
                             f"Knee {knee_angle_right:.0f}")

    # --- Elbow angle — use whichever arm is most visible ---
    elbow_angle = None
    for side in [
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    ]:
        s = get_point(landmarks, side[0], frame_width, frame_height)
        e = get_point(landmarks, side[1], frame_width, frame_height)
        w = get_point(landmarks, side[2], frame_width, frame_height)
        if s and e and w:
            a = angle_at_vertex_deg(s, e, w)
            if a is not None:
                elbow_angle = a
                draw_angle_highlight(frame_out, s, e, w,
                                     f"Elbow {a:.0f}", color=(255, 165, 0))
                break

    # --- Hip squareness ---
    hip_square = None
    if left_hip and right_hip:
        hip_y_diff = abs(left_hip[1] - right_hip[1])
        hip_square = hip_y_diff < (0.05 * frame_height)
        draw_line_with_label(frame_out, left_hip, right_hip,
                             f"Hips {'Square' if hip_square else 'Tilted'}",
                             color=(0, 165, 255))

    # --- Body balance ---
    balance_ok = None
    if left_hip and right_hip and left_ankle and right_ankle:
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
        balance_ok = abs(hip_mid_x - ankle_mid_x) < (0.08 * frame_width)
        balance_text = "Balance: OK" if balance_ok else "Balance: LEANING"
        balance_color = (0, 255, 0) if balance_ok else (0, 0, 255)
        draw_text(frame_out, balance_text, (20, frame_height - 30), color=balance_color)

    cv2.imwrite(os.path.join(output_dir, "load_frame.jpg"), frame_out)

    return {
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "elbow_angle_deg": elbow_angle,
        "hip_square": hip_square,
        "balance_ok": balance_ok,
    }