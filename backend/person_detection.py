
import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    raise SystemExit

while True:
    success, frame = cap.read()

    if not success:
        break

    # Detect people (class 0)
    results = model(frame, classes=[0], verbose=False)
    person_count = 0

    for result in results:
        for box in result.boxes:
            person_count += 1

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Person {confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

    cv2.putText(
        frame,
        f"People detected: {person_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 255), 2
    )

    cv2.imshow("Person Detection - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
