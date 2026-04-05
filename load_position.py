import math
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

LOAD_FRAME_PATH = "load_frame.jpg"
MIN_VISIBILITY = 0.5


def _lm_px(lm, frame_w, frame_h):
    return lm.x * frame_w, lm.y * frame_h


def _angle_at_vertex_deg(p_prev, p_vertex, p_next):
    ba = np.array(p_prev) - np.array(p_vertex)
    bc = np.array(p_next) - np.array(p_vertex)
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


def _draw_line_with_label(frame, p1, p2, label, color=(255, 255, 255)):
    i1 = (int(p1[0]), int(p1[1]))
    i2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, i1, i2, color, 2, cv2.LINE_AA)
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)
    cv2.putText(frame, label, (mid_x + 8, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _get_min_knee_angle(landmarks, frame_w, frame_h):
    angles = []
    for side in [
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    ]:
        hip = _get_point(landmarks, side[0], frame_w, frame_h)
        knee = _get_point(landmarks, side[1], frame_w, frame_h)
        ankle = _get_point(landmarks, side[2], frame_w, frame_h)
        if hip and knee and ankle:
            a = _angle_at_vertex_deg(hip, knee, ankle)
            if a is not None:
                angles.append(a)
    if not angles:
        return None
    return min(angles)


def detect_load_position(video_path, release_frame_number, ball_bbox_at_load=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    best_frame = None
    best_angle = float("inf")
    best_frame_number = 0
    best_landmarks = None
    frame_number = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while frame_number <= release_frame_number:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                angle = _get_min_knee_angle(results.pose_landmarks, frame_width, frame_height)
                if angle is not None and angle < best_angle:
                    best_angle = angle
                    best_frame = frame.copy()
                    best_frame_number = frame_number
                    best_landmarks = results.pose_landmarks

            frame_number += 1

    cap.release()

    if best_frame is None:
        print("Could not detect load position — no pose landmarks found.")
        return None

    mp_drawing.draw_landmarks(
        best_frame, best_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
    )

    print("\n--- Load Position Analysis ---")
    print(f"Load position frame number: {best_frame_number}")

    # ---- 1. Knee angles ----
    knee_angle_left = None
    knee_angle_right = None

    left_hip = _get_point(best_landmarks, mp_pose.PoseLandmark.LEFT_HIP, frame_width, frame_height)
    left_knee = _get_point(best_landmarks, mp_pose.PoseLandmark.LEFT_KNEE, frame_width, frame_height)
    left_ankle = _get_point(best_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, frame_width, frame_height)
    right_hip = _get_point(best_landmarks, mp_pose.PoseLandmark.RIGHT_HIP, frame_width, frame_height)
    right_knee = _get_point(best_landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, frame_width, frame_height)
    right_ankle = _get_point(best_landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, frame_width, frame_height)

    if left_hip and left_knee and left_ankle:
        knee_angle_left = _angle_at_vertex_deg(left_hip, left_knee, left_ankle)
        _draw_angle_highlight(best_frame, left_hip, left_knee, left_ankle,
                              f"Knee {knee_angle_left:.0f} deg")
        print(f"Left knee angle: {knee_angle_left:.1f} deg")

    if right_hip and right_knee and right_ankle:
        knee_angle_right = _angle_at_vertex_deg(right_hip, right_knee, right_ankle)
        _draw_angle_highlight(best_frame, right_hip, right_knee, right_ankle,
                              f"Knee {knee_angle_right:.0f} deg")
        print(f"Right knee angle: {knee_angle_right:.1f} deg")

    # ---- 2. Elbow angle ----
    elbow_angle = None
    for side in [
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    ]:
        shoulder = _get_point(best_landmarks, side[0], frame_width, frame_height)
        elbow = _get_point(best_landmarks, side[1], frame_width, frame_height)
        wrist = _get_point(best_landmarks, side[2], frame_width, frame_height)
        if shoulder and elbow and wrist:
            a = _angle_at_vertex_deg(shoulder, elbow, wrist)
            if a is not None:
                elbow_angle = a
                _draw_angle_highlight(best_frame, shoulder, elbow, wrist,
                                      f"Elbow {a:.0f} deg", color=(255, 165, 0))
                print(f"Elbow angle at load: {elbow_angle:.1f} deg")
                break

    # ---- 3. Hip squareness ----
    hip_square = None
    if left_hip and right_hip:
        hip_y_diff = abs(left_hip[1] - right_hip[1])
        hip_square = hip_y_diff < (0.05 * frame_height)
        _draw_line_with_label(best_frame, left_hip, right_hip,
                              f"Hips {'Square' if hip_square else 'Tilted'}", color=(0, 165, 255))
        print(f"Hip squareness — y difference: {hip_y_diff:.1f}px — "
              f"{'SQUARE' if hip_square else 'TILTED'}")

    # ---- 4. Ball height at load ----
    ball_height_ok = None
    left_shoulder = _get_point(best_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, frame_width, frame_height)
    right_shoulder = _get_point(best_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, frame_width, frame_height)

    if ball_bbox_at_load is not None and left_shoulder and right_shoulder:
        bx, by, bw, bh = ball_bbox_at_load
        ball_cy = by + bh / 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = None
        if left_hip and right_hip:
            hip_y = (left_hip[1] + right_hip[1]) / 2
        if hip_y:
            ball_height_ok = hip_y > ball_cy > shoulder_y
            label = "Ball: chest OK" if ball_height_ok else "Ball: low"
            cv2.circle(best_frame, (int(bx + bw/2), int(ball_cy)), int(min(bw, bh)/2), (0, 255, 0), 2)
            cv2.putText(best_frame, label, (int(bx + bw/2) + 10, int(ball_cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Ball height at load — "
                  f"{'between hip and shoulder (GOOD)' if ball_height_ok else 'not in ideal range'}")

    # ---- 5. Body balance ----
    balance_ok = None
    if left_hip and right_hip and left_ankle and right_ankle:
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
        balance_offset = abs(hip_mid_x - ankle_mid_x)
        balance_threshold = 0.08 * frame_width
        balance_ok = balance_offset < balance_threshold
        print(f"Body balance — hip vs ankle center offset: {balance_offset:.1f}px — "
              f"{'BALANCED' if balance_ok else 'LEANING'}")

    cv2.imwrite(LOAD_FRAME_PATH, best_frame)
    print(f"Saved load frame: {LOAD_FRAME_PATH}")

    return {
        "load_frame_number": best_frame_number,
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "elbow_angle_deg": elbow_angle,
        "hip_square": hip_square,
        "ball_height_ok": ball_height_ok,
        "balance_ok": balance_ok,
    }

