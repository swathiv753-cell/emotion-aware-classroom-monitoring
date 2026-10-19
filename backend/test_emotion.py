import cv2
from backend.emotion_model import predict_emotion

IMAGE_PATH = r"C:\path\to\your\test_image.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

emotion, confidence = predict_emotion(image)

print("Predicted emotion:", emotion)
print(f"Model confidence: {confidence:.2f}%")
