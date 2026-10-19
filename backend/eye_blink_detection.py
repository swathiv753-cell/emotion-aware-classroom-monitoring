
import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indexes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Settings
EAR_THRESHOLD = 0.21
MIN_CLOSED_FRAMES = 2

blink_count = 0
closed_frames = 0
eyes_closed = False


def calculate_ear(landmarks, eye_indices, width, height):
    """Calculate Eye Aspect Ratio (EAR)."""
    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

    # Vertical eye distances
    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])

    # Horizontal eye distance
    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


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
            print("Could not read webcam frame.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "Face not detected"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )
            right_ear = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            ear = (left_ear + right_ear) / 2.0

            # Mark the eye landmarks
            for index in LEFT_EYE + RIGHT_EYE:
                point = landmarks[index]
                x = int(point.x * width)
                y = int(point.y * height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if ear < EAR_THRESHOLD:
                closed_frames += 1
                status = "Eyes closed"
            else:
                status = "Eyes open"

                # Count a blink after eyes reopen
                if (
                    eyes_closed
                    and closed_frames >= MIN_CLOSED_FRAMES
                ):
                    blink_count += 1

                closed_frames = 0

            eyes_closed = ear < EAR_THRESHOLD

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2
            )

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Blink count: {blink_count}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.imshow("Eye Blink Detection - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Total blinks detected:", blink_count)
