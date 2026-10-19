
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
YOLO_MODEL = "yolo11n.pt"
CAMERA_INDEX = 0
PERSON_CONFIDENCE = 0.40
EAR_THRESHOLD = 0.21
MIN_CLOSED_FRAMES = 2
MAX_FACES = 40

# FaceMesh eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


mp_face_mesh = mp.solutions.face_mesh

# 3D reference points used for approximate head pose
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye
    (225.0, 170.0, -135.0),    # Right eye
    (-150.0, -150.0, -125.0),  # Left mouth
    (150.0, -150.0, -125.0)    # Right mouth
], dtype=np.float64)

POSE_IDS = [1, 152, 33, 263, 61, 291]

# Blink state for each temporary YOLO track ID
blink_states = {}


def calculate_ear(landmarks, indices, width, height):
    points = []

    for i in indices:
        p = landmarks[i]
        points.append((p.x * width, p.y * height))

    vertical1 = math.dist(points[1], points[5])
    vertical2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    if horizontal < 1:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def estimate_head_direction(landmarks, width, height):
    image_points = np.array([
        (landmarks[i].x * width, landmarks[i].y * height)
        for i in POSE_IDS
    ], dtype=np.float64)

    focal_length = float(width)
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    ok, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        return "Unknown", 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)

    pitch, yaw, roll = angles

    if yaw < -15:
        direction = "Left"
    elif yaw > 15:
        direction = "Right"
    elif pitch < -15:
        direction = "Up"
    elif pitch > 15:
        direction = "Down"
    else:
        direction = "Forward"

    return direction, float(yaw), float(pitch)


# ---------------- START MODELS ----------------
print("Loading YOLO...")
person_model = YOLO(YOLO_MODEL)

print("Opening webcam...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

# Use a moderate capture size for performance.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

fps = 0.0
frame_count = 0
fps_start = time.perf_counter()
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Classroom monitoring started. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        frame_count += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= 1.0:
         fps = frame_count / elapsed
         frame_count = 0
         fps_start = time.perf_counter()
         

        if not ok:
            print("Could not read webcam frame.")
            break

        # Keep the same coordinate orientation throughout the pipeline.
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # Track people across frames.
        tracking_results = person_model.track(
            frame,
            persist=True,
            classes=[0],
            conf=PERSON_CONFIDENCE,
            tracker="bytetrack.yaml",
            imgsz=640,
            max_det=60,
            verbose=False
        )

        people = []

        for result in tracking_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                people.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2)
                })

        # FaceMesh processes the full frame once per video frame.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        # Assign each detected face to the person box containing
        # its nose point. A face is assigned to at most one person.
        assigned_faces = {}
        used_people = set()

        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                lm = face.landmark

                nose_x = int(lm[1].x * width)
                nose_y = int(lm[1].y * height)

                candidates = []

                for i, person in enumerate(people):
                    if i in used_people:
                        continue

                    x1, y1, x2, y2 = person["box"]

                    if x1 <= nose_x <= x2 and y1 <= nose_y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        candidates.append((area, i))

                if candidates:
                    _, chosen_index = min(candidates)
                    used_people.add(chosen_index)
                    assigned_faces[chosen_index] = face

        faces_detected = 0
        faces_not_detected = 0

        for i, person in enumerate(people):
            track_id = person["id"]
            x1, y1, x2, y2 = person["box"]

            # A stable track ID is needed to retain blink counts.
            if track_id is None:
                label_id = "?"
            else:
                label_id = str(track_id)

            face = assigned_faces.get(i)

            if face is None:
                faces_not_detected += 1
                color = (0, 0, 255)
                label = f"ID {label_id}: Face not detected"

            else:
                faces_detected += 1
                color = (0, 255, 0)
                landmarks = face.landmark

                ear_left = calculate_ear(
                    landmarks, LEFT_EYE, width, height
                )
                ear_right = calculate_ear(
                    landmarks, RIGHT_EYE, width, height
                )
                ear = (ear_left + ear_right) / 2.0

                direction, yaw, pitch = estimate_head_direction(
                    landmarks, width, height
                )

                if track_id is not None:
                    state = blink_states.setdefault(
                        track_id,
                        {
                            "blinks": 0,
                            "closed_frames": 0,
                            "was_closed": False
                        }
                    )

                    if ear < EAR_THRESHOLD:
                        state["closed_frames"] += 1
                        eye_status = "Closed"
                    else:
                        eye_status = "Open"

                        if (
                            state["was_closed"]
                            and state["closed_frames"] >= MIN_CLOSED_FRAMES
                        ):
                            state["blinks"] += 1

                        state["closed_frames"] = 0

                    state["was_closed"] = ear < EAR_THRESHOLD
                    blink_count = state["blinks"]
                else:
                    eye_status = "Unknown"
                    blink_count = 0

                label = (
                    f"ID {label_id}: {direction}, Eyes {eye_status}, "
                    f"Blinks {blink_count}"
                )

                # Draw a small number of landmarks for visual checking.
                for index in LEFT_EYE + RIGHT_EYE:
                    px = int(landmarks[index].x * width)
                    py = int(landmarks[index].y * height)

                    cv2.circle(
                        frame, (px, py), 2, (255, 255, 0), -1
                    )

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color, 2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 2
            )

        # Summary panel
        cv2.rectangle(frame, (5, 5), (460, 100), (30, 30, 30), -1)

        cv2.putText(
            frame, f"People detected: {len(people)}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 255), 2
        )
        cv2.putText(
            frame, f"Faces detected: {faces_detected}",
            (15, 58), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Faces not detected: {faces_not_detected}",
            (15, 86), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 2
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.imshow("Classroom Monitor - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Classroom monitoring stopped.")
for track_id, state in blink_states.items():
    print(f"Track {track_id}: {state['blinks']} blinks")
