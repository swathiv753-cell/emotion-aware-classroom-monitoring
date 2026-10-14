
from pathlib import Path
from collections import Counter
import json
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

# ---------- EXISTING MODEL PATHS ----------
BASE_DIR = Path(r"C:\Users\ganesh\Desktop\emotion")
MODEL_PATH = BASE_DIR / "models" / "best_emotion_model.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"

INFERENCE_EVERY_N_FRAMES = 3
MAX_WIDTH = 960
MIN_FACE_SIZE = (40, 40)

# Rule-based proxy settings, not a validated attention detector
MIN_HISTORY = 8
HISTORY_SIZE = 40
OFF_CENTER_RATIO = 0.60


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        return [
            labels[str(i)] if str(i) in labels else labels[i]
            for i in range(len(labels))
        ]
    return labels


def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels()

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    if detector.empty():
        raise RuntimeError(
            "Face detector failed to load. Check your OpenCV installation."
        )

    if len(labels) != model.output_shape[-1]:
        raise ValueError("Number of labels does not match model outputs.")

    print("Emotion classes:", labels)
    return model, labels, detector


def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askopenfilename(
            title=title, filetypes=filetypes
        )
    finally:
        root.destroy()


def resize_frame(frame):
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        ratio = MAX_WIDTH / w
        frame = cv2.resize(
            frame, (MAX_WIDTH, int(h * ratio)),
            interpolation=cv2.INTER_AREA
        )
    return frame


def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE
    )


def predict_emotion(frame, box, model, labels):
    x, y, w, h = map(int, box)
    crop = frame[y:y+h, x:x+w]

    if crop.size == 0:
        return "unknown", 0.0

    # Match the training script: RGB, 224x224, values in [0,255].
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (224, 224)).astype(np.float32)

    probabilities = model.predict(
        crop[np.newaxis, ...], verbose=0
    )[0]

    index = int(np.argmax(probabilities))
    return labels[index], float(probabilities[index]) * 100


def face_is_off_center(box, frame_shape):
    """Face location proxy only; not true head pose or attention."""
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]
    cx = (x + w / 2) / max(frame_w, 1)
    cy = (y + h / 2) / max(frame_h, 1)

    return cx < 0.25 or cx > 0.75 or cy < 0.10 or cy > 0.90


def draw_dashboard(frame, face_count, emotion_counts,
                   unknown_frames, engagement_percent):
    lines = [
        f"Visible faces in current frame: {face_count}",
        f"Frames with no detected face: {unknown_frames}",
    ]

    if engagement_percent is None:
        lines.append("Face-visibility proxy: UNKNOWN")
    else:
        lines.append(
            f"Face-visibility proxy estimate: {engagement_percent:.1f}%"
        )

    lines.append("Expression prediction counts (separate metric):")
    for label, count in emotion_counts.items():
        lines.append(f"{label}: {count}")

    lines.append("Estimate only - not a measure of learning or attention")

    line_height = 21
    panel_height = min(
        frame.shape[0] - 5,
        12 + line_height * len(lines)
    )

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (5, 5),
        (min(frame.shape[1] - 5, 530), panel_height),
        (15, 15, 15), -1
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    for i, line in enumerate(lines):
        y = 23 + i * line_height
        if y >= frame.shape[0] - 5:
            break

        color = (
            (0, 230, 255)
            if "proxy" in line.lower()
            else (245, 245, 245)
        )
        cv2.putText(
            frame, line, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            color, 1, cv2.LINE_AA
        )

    cv2.putText(
        frame, "Press Q to stop",
        (12, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA
    )


def analyze_photo(path, model, labels, detector):
    frame = cv2.imread(str(path))
    if frame is None:
        print("Could not read photo:", path)
        return

    frame = resize_frame(frame)
    boxes = detect_faces(frame, detector)
    counts = Counter({label: 0 for label in labels})

    for box in boxes:
        emotion, confidence = predict_emotion(
            frame, box, model, labels
        )
        counts[emotion] += 1

        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
        cv2.putText(
            frame, f"{emotion} ({confidence:.1f}%)",
            (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2, cv2.LINE_AA
        )

    # A single photo cannot provide temporal attention evidence.
    draw_dashboard(
        frame, len(boxes), counts,
        1 if len(boxes) == 0 else 0, None
    )

    print("\nPHOTO SUMMARY")
    print("Visible faces:", len(boxes))
    print("Expression counts:", dict(counts))
    print("Attention: UNKNOWN - a single image is insufficient evidence.")

    cv2.imshow("Classroom AI - Photo", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def analyze_video(source, model, labels, detector, source_name):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open:", source_name)
        return

    # Every class is included, even if its count remains zero.
    emotion_counts = Counter({label: 0 for label in labels})

    frame_number = 0
    no_face_frames = 0
    face_observations = 0
    frames_with_faces = 0

    # Temporary face histories are discarded when this session ends.
    # No student identities or persistent histories are created.
    face_histories = []
    last_predictions = []
    unknown_observations = 0

    print(f"\nAnalyzing {source_name}. Press Q to stop.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_number += 1
        frame = resize_frame(frame)
        boxes = list(detect_faces(frame, detector))
        boxes = [tuple(map(int, box)) for box in boxes]

        face_observations += len(boxes)

        if boxes:
            frames_with_faces += 1
        else:
            no_face_frames += 1
            unknown_observations += 1

        # Update current emotion predictions periodically.
        if frame_number == 1 or frame_number % INFERENCE_EVERY_N_FRAMES == 0:
            last_predictions = []
            for box in boxes:
                emotion, confidence = predict_emotion(
                    frame, box, model, labels
                )
                last_predictions.append((emotion, confidence))
                emotion_counts[emotion] += 1

        # Draw current predictions. If detections change between inference
        # frames, show unknown rather than reusing another face's prediction.
        for i, box in enumerate(boxes):
            x, y, w, h = box

            if i < len(last_predictions):
                emotion, confidence = last_predictions[i]
                label_text = f"{emotion}: {confidence:.1f}%"
            else:
                label_text = "Expression: unknown"

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), (0, 200, 0), 2
            )
            cv2.putText(
                frame, label_text, (x, max(y-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (0, 255, 0), 2, cv2.LINE_AA
            )

        # This is deliberately a FACE-VISIBILITY proxy, not an attention score.
        # Frames without detected faces are unknown, not inattentive.
        proxy = (
            100.0 * frames_with_faces / frame_number
            if frame_number > 0 else None
        )

        draw_dashboard(
            frame, len(boxes), emotion_counts,
            no_face_frames, proxy
        )

        cv2.imshow(f"Classroom AI - {source_name}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n========== CLASS-LEVEL SUMMARY ==========")
    print("Input:", source_name)
    print("Frames processed:", frame_number)
    print("Total visible-face observations:", face_observations)
    print("Frames with no visible face:", no_face_frames)
    print("Expression prediction counts:", dict(emotion_counts))

    if frame_number == 0 or face_observations == 0:
        print("Face-visibility proxy: UNKNOWN - insufficient evidence.")
    else:
        percentage = 100.0 * frames_with_faces / frame_number
        print(
            f"Face-visibility proxy estimate: {percentage:.1f}% "
            "of frames contained at least one detected face."
        )

    print(
        "This percentage does NOT measure attention or engagement. "
        "It only describes whether faces were detected."
    )
    print("No face detected means UNKNOWN, not inattentive.")


def main():
    model, labels, detector = load_resources()

    while True:
        print("\n====== CLASSROOM AI PROTOTYPE ======")
        print("1. Photo")
        print("2. Webcam")
        print("3. Video")
        print("4. Exit")

        choice = input("Choose 1, 2, 3, or 4: ").strip()

        if choice == "1":
            path = select_file(
                "Choose a photo",
                [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
            )
            if path:
                analyze_photo(path, model, labels, detector)

        elif choice == "2":
            analyze_video(0, model, labels, detector, "Webcam")

        elif choice == "3":
            path = select_file(
                "Choose a video",
                [("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
            )
            if path:
                analyze_video(
                    path, model, labels, detector, Path(path).name
                )

        elif choice == "4":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
