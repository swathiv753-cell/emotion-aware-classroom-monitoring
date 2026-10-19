import cv2
from deepface import DeepFace


def detect_emotions(frame):
    """Return face boxes and expression predictions for one BGR image."""

    results = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=True,
        silent=True,
    )

    if isinstance(results, dict):
        results = [results]

    predictions = []

    for index, item in enumerate(results, start=1):
        region = item.get("region", {})

        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        width = int(region.get("w", 0))
        height = int(region.get("h", 0))

        if width <= 0 or height <= 0:
            continue

        emotion = item.get("dominant_emotion", "unknown")
        scores = item.get("emotion", {})
        confidence = float(scores.get(emotion, 0.0))

        predictions.append({
            "face_id": index,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "box": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
        })

    return predictions
