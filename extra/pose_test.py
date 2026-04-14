import cv2
import mediapipe as mp


# ====== CHANGE THESE TWO PATHS ======
input_video_path = "test_video1.mp4"
output_video_path = "output_with_skeleton.mp4"
# ====================================


# Create shortcuts to MediaPipe pose tools.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Open the input video file.
cap = cv2.VideoCapture(input_video_path)

# If the video cannot be opened, stop and show a helpful message.
if not cap.isOpened():
    print(f"Error: Could not open input video: {input_video_path}")
    raise SystemExit


# Read video details so output video matches input properties.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# If FPS is missing/invalid, use a safe default.
if fps <= 0:
    fps = 30.0


# Define output video format.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


# Create the pose detector once and reuse it for all frames.
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    while True:
        # Read one frame from the video.
        success, frame = cap.read()

        # If no frame is returned, we reached the end of the video.
        if not success:
            break

        # Convert BGR image (OpenCV) to RGB image (MediaPipe expects RGB).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose detection on this frame.
        results = pose.process(frame_rgb)

        # If landmarks are found, draw the full body skeleton.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2
                ),
            )

        # Write this processed frame into the output video.
        out.write(frame)


# Release files/resources properly.
cap.release()
out.release()

print(f"Done! Output saved to: {output_video_path}")
