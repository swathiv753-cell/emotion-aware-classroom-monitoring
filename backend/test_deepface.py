
from pathlib import Path
from deepface import DeepFace

print("=== DeepFace Emotion Detection ===")
image_path = input("Enter the full path of your image: ").strip().strip('"')

if not Path(image_path).is_file():
    print("Error: Image file not found. Check the path and try again.")
    raise SystemExit

try:
    results = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=True
    )

    for i, face in enumerate(results, start=1):
        print(f"\nFace {i}")
        print("Predicted expression:", face["dominant_emotion"])
        print("Expression scores:")

        for emotion, score in face["emotion"].items():
            print(f"  {emotion}: {score:.2f}%")

except Exception as e:
    print("Could not analyze image:", e)
