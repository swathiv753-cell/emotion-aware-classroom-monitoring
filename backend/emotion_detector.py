import logging

logger = logging.getLogger(__name__)


def detect_emotions(frame):
    from deepface import DeepFace

    # Try a more robust face detector first.
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="retinaface",
            enforce_detection=True,
            silent=True,
        )
    except ValueError as exc:
        # A missed face is normal for some frames.
        if "face" in str(exc).lower() and "detect" in str(exc).lower():
            logger.info("No face detected in this frame.")
            return []
        raise

    if isinstance(results, dict):
        results = [results]

    predictions = []

    for index, result in enumerate(results, start=1):
        region = result.get("region", {})

        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))

        if w <= 0 or h <= 0:
            continue

        emotion = result.get("dominant_emotion", "unknown")
        scores = result.get("emotion", {}) or {}

        predictions.append({
            "face_id": index,
            "emotion": emotion,
            "confidence": round(float(scores.get(emotion, 0)), 2),
            "box": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            },
            "head_direction": "Unknown",
            "yaw": None,
            "pitch": None,
            "roll": None,
            "eye_state": "Unknown",
            "ear": None,
            "blink_count": 0,
            "attention_status": "Uncertain",
        })

    return predictions