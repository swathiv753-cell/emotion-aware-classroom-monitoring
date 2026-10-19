import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_emotion_model.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"

# Load the model and labels once
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    CLASS_NAMES = json.load(file)


def predict_emotion(face_bgr):
    """
    Predict an emotion from a face crop.
    face_bgr: face image cropped from an OpenCV BGR frame.
    Returns the predicted label and confidence percentage.
    """
    if face_bgr is None or face_bgr.size == 0:
        return "Unknown", 0.0

    # Convert OpenCV BGR to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Match the 224x224 training image size
    face_rgb = cv2.resize(face_rgb, (224, 224))

    # Match training preprocessing: scale pixels to [-1, 1]
    image = face_rgb.astype(np.float32)
    image = image / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)

    # Predict
    probabilities = model.predict(image, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    emotion = CLASS_NAMES[best_index]
    confidence = float(probabilities[best_index] * 100.0)

    return emotion, confidence
