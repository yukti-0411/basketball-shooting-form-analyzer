import math
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

MIN_VISIBILITY = 0.5


# --- Coordinate helpers ---

def lm_px(lm, frame_w, frame_h):
    """Convert MediaPipe normalized landmark to pixel coordinates."""
    return lm.x * frame_w, lm.y * frame_h


def get_point(landmarks, idx, frame_w, frame_h):
    """Return pixel coordinates of a landmark if visible, else None."""
    lm = landmarks.landmark[idx]
    if lm.visibility < MIN_VISIBILITY:
        return None
    return lm_px(lm, frame_w, frame_h)


# --- Math helpers ---

def angle_at_vertex_deg(p_prev, p_vertex, p_next):
    """Calculate the angle in degrees at the vertex point between three points."""
    ba = np.array(p_prev, dtype=np.float64) - np.array(p_vertex, dtype=np.float64)
    bc = np.array(p_next, dtype=np.float64) - np.array(p_vertex, dtype=np.float64)
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    c = float(np.dot(ba, bc) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


# --- Drawing helpers ---

def draw_angle_highlight(frame, p1, vertex, p2, label, color=(0, 255, 255)):
    """Draw two lines at a joint vertex with an angle label."""
    i1 = (int(p1[0]), int(p1[1]))
    iv = (int(vertex[0]), int(vertex[1]))
    i2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, iv, i1, color, 2, cv2.LINE_AA)
    cv2.line(frame, iv, i2, color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (iv[0] + 8, iv[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_line_with_label(frame, p1, p2, label, color=(255, 255, 255)):
    """Draw a straight line between two points with a label at the midpoint."""
    i1 = (int(p1[0]), int(p1[1]))
    i2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, i1, i2, color, 2, cv2.LINE_AA)
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)
    cv2.putText(frame, label, (mid_x + 8, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_text(frame, text, pos, color=(255, 255, 255), scale=0.55):
    """Draw a simple text label at a given position."""
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# --- Pose analysis helpers ---

def shooting_arm_side(landmarks, frame_w, frame_h, ball_cx, ball_cy):
    """Return 'left' or 'right' — whichever wrist is closer to the ball center."""
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


def distance_ball_to_closest_wrist(landmarks, frame_w, frame_h, ball_cx, ball_cy):
    """Return the shortest pixel distance between the ball center and either wrist."""
    best = None
    for idx in (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST):
        lm = landmarks.landmark[idx]
        if lm.visibility < MIN_VISIBILITY:
            continue
        wx = lm.x * frame_w
        wy = lm.y * frame_h
        d = math.hypot(ball_cx - wx, ball_cy - wy)
        if best is None or d < best:
            best = d
    return best


def get_wrist_elbow_y_diff(landmarks, shooting_side, frame_w, frame_h):
    """Return wrist_y minus elbow_y for the shooting arm. Negative means wrist above elbow."""
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


def draw_skeleton(frame, landmarks):
    """Draw MediaPipe skeleton — green dots and red connections — on the frame."""
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        frame, landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
    )