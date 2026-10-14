from pathlib import Path
import json
import cv2
import numpy as np
import tensorflow as tf

# Paths to your trained model and emotion labels
BASE_DIR = Path(r"C:\Users\ganesh\Desktop\emotion")
MODEL_PATH = BASE_DIR / "models" / "best_emotion_model.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_PATH = BASE_DIR / "test_face.jpg"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Photo not found: {IMAGE_PATH}\n"
        "Save a clear face photo as test_face.jpg in the emotion folder."
    )

# Load model and emotion labels
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    emotion_labels = json.load(f)

# Read the test photo
image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise ValueError("Could not read the test photo.")

# Detect a face
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
faces = face_detector.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

if len(faces) == 0:
    print("No face detected. Try a clear, front-facing photo.")
else:
    for (x, y, w, h) in faces:
        # Crop face and prepare it for MobileNetV2
        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = face.astype("float32")

        # Model performs its own rescaling to [-1, 1]
        input_image = np.expand_dims(face, axis=0)
        probabilities = model.predict(input_image, verbose=0)[0]

        index = int(np.argmax(probabilities))
        emotion = emotion_labels[index]
        confidence = float(probabilities[index]) * 100

        print(f"Predicted expression: {emotion}")
        print(f"Model confidence: {confidence:.2f}%")

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            image, f"{emotion} ({confidence:.1f}%)",
            (x, max(y-10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    cv2.imshow("Emotion Expression Test", image)
    print("Close the image window to finish.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()