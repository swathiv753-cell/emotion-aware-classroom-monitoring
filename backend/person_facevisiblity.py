
import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLO person detector
model = YOLO("yolo11n.pt")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # Detect people
        results = model(
            frame,
            classes=[0],
            conf=0.40,
            verbose=False
        )

        people_count = 0
        faces_detected = 0
        faces_not_detected = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                # Keep coordinates within image boundaries
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                people_count += 1

                # Crop the detected person
                person_crop = frame[y1:y2, x1:x2]

                # Check whether a face is visible
                rgb_crop = cv2.cvtColor(
                    person_crop, cv2.COLOR_BGR2RGB
                )
                face_result = face_mesh.process(rgb_crop)

                if face_result.multi_face_landmarks:
                    faces_detected += 1
                    label = "Face Detected"
                    color = (0, 255, 0)

                    # Draw face landmarks within person crop
                    crop_h, crop_w = person_crop.shape[:2]

                    for face in face_result.multi_face_landmarks:
                        for landmark in face.landmark[::8]:
                            px = x1 + int(landmark.x * crop_w)
                            py = y1 + int(landmark.y * crop_h)

                            cv2.circle(
                                frame, (px, py), 1,
                                (255, 255, 0), -1
                            )
                else:
                    faces_not_detected += 1
                    label = "Face Not Detected"
                    color = (0, 0, 255)

                # Draw person bounding box and label
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    color, 2
                )

                cv2.putText(
                    frame, label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2
                )

        # Display summary
        cv2.putText(
            frame, f"People: {people_count}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.putText(
            frame, f"Faces detected: {faces_detected}",
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Faces not detected: {faces_not_detected}",
            (15, 90), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )

        cv2.imshow(
            "Person + Face Visibility - Press Q to Quit",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Person and face visibility test finished.")
