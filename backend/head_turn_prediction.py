
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Face landmarks used to estimate head pose
LANDMARKS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291
}

# Approximate 3D reference points for head-pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye
    (225.0, 170.0, -135.0),   # Right eye
    (-150.0, -150.0, -125.0), # Left mouth
    (150.0, -150.0, -125.0)   # Right mouth
], dtype=np.float64)


def estimate_head_direction(landmarks, width, height):
    image_points = np.array([
        (landmarks[LANDMARKS["nose"]].x * width,
         landmarks[LANDMARKS["nose"]].y * height),
        (landmarks[LANDMARKS["chin"]].x * width,
         landmarks[LANDMARKS["chin"]].y * height),
        (landmarks[LANDMARKS["left_eye"]].x * width,
         landmarks[LANDMARKS["left_eye"]].y * height),
        (landmarks[LANDMARKS["right_eye"]].x * width,
         landmarks[LANDMARKS["right_eye"]].y * height),
        (landmarks[LANDMARKS["left_mouth"]].x * width,
         landmarks[LANDMARKS["left_mouth"]].y * height),
        (landmarks[LANDMARKS["right_mouth"]].x * width,
         landmarks[LANDMARKS["right_mouth"]].y * height)
    ], dtype=np.float64)

    focal_length = width
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return "Unknown", 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles

    # Initial thresholds; calibrate for your camera
    if yaw < -15:
        direction = "Looking Left"
    elif yaw > 15:
        direction = "Looking Right"
    elif pitch < -15:
        direction = "Looking Up"
    elif pitch > 15:
        direction = "Looking Down"
    else:
        direction = "Facing Forward"

    return direction, float(yaw), float(pitch)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        direction = "Face not detected"
        yaw, pitch = 0.0, 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            direction, yaw, pitch = estimate_head_direction(
                landmarks, width, height
            )

            cv2.putText(
                frame,
                f"Direction: {direction}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            )

        else:
            direction = "face not visible"
            cv2.putText(
                frame, direction, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        cv2.imshow("Head Turn Detection - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
