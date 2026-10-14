
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_emotion_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError("Run train.py first to train the model.")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)


def predict_emotion(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

    # Prepare the image in the same format used for training
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict probabilities for all emotion classes
    probabilities = model.predict(image_array, verbose=0)[0]

    best_index = int(np.argmax(probabilities))
    emotion = class_names[best_index]
    confidence = float(probabilities[best_index]) * 100

    print("\n--- Emotion Prediction ---")
    print("Predicted emotion:", emotion)
    print(f"Model confidence: {confidence:.2f}%")

    print("\nAll class probabilities:")
    for i in np.argsort(probabilities)[::-1]:
        print(f"{class_names[i]}: {probabilities[i] * 100:.2f}%")


if __name__ == "__main__":
    image_path = input("Enter the image path: ").strip().strip('"')
    predict_emotion(image_path)
