
import os
import tempfile

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="ClassVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
        "http://localhost:5177",
        "http://127.0.0.1:5177",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# HEAD DIRECTION DETECTION
# --------------------------------------------------

def detect_head_directions(frame):
    """Estimate approximate head direction for visible faces."""

    height, width = frame.shape[:2]
    detections = []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return detections

        for face in results.multi_face_landmarks:
            points = face.landmark

            # Eye and nose landmarks.
            nose_x = points[1].x * width
            nose_y = points[1].y * height

            left_eye_x = points[33].x * width
            right_eye_x = points[263].x * width

            left_eye_y = points[33].y * height
            right_eye_y = points[263].y * height

            eye_center_x = (left_eye_x + right_eye_x) / 2
            eye_center_y = (left_eye_y + right_eye_y) / 2

            eye_distance = max(
                ((right_eye_x - left_eye_x) ** 2
                 + (right_eye_y - left_eye_y) ** 2) ** 0.5,
                1.0,
            )

            # Rough landmark ratios, not calibrated head-pose angles.
            horizontal_offset = (
                (nose_x - eye_center_x) / eye_distance
            )
            vertical_offset = (
                (nose_y - eye_center_y) / eye_distance
            )

            if horizontal_offset < -0.18:
                direction = "Looking Left"
            elif horizontal_offset > 0.18:
                direction = "Looking Right"
            elif vertical_offset > 0.85:
                direction = "Looking Down"
            else:
                direction = "Facing Forward"

            # Face Mesh bounding box.
            xs = [point.x * width for point in points]
            ys = [point.y * height for point in points]

            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(width, int(max(xs)))
            y2 = min(height, int(max(ys)))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "direction": direction,
                "box": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
            })

    return detections


# --------------------------------------------------
# MATCH FACE DETECTION WITH HEAD DIRECTION
# --------------------------------------------------

def match_head_direction(face_box, head_directions):
    """Match the nearest Face Mesh box to a DeepFace face box."""

    face_cx = face_box["x"] + face_box["width"] / 2
    face_cy = face_box["y"] + face_box["height"] / 2

    best_direction = "Unknown"
    best_distance = float("inf")

    face_scale = max(
        face_box["width"],
        face_box["height"],
        1,
    )

    for detected in head_directions:
        box = detected["box"]

        head_cx = box["x"] + box["width"] / 2
        head_cy = box["y"] + box["height"] / 2

        distance = (
            (face_cx - head_cx) ** 2
            + (face_cy - head_cy) ** 2
        ) ** 0.5

        # Avoid assigning a direction from a different person.
        if distance < best_distance and distance < face_scale:
            best_distance = distance
            best_direction = detected["direction"]

    return best_direction


# --------------------------------------------------
# EMOTION + HEAD DIRECTION
# --------------------------------------------------

def detect_emotions(frame):
    results = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=False,
        silent=True,
    )

    if isinstance(results, dict):
        results = [results]

    # Run head-direction detection on the same frame.
    head_directions = detect_head_directions(frame)

    predictions = []

    for item in results:
        region = item.get("region", {})

        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        width = int(region.get("w", 0))
        height = int(region.get("h", 0))

        if width <= 0 or height <= 0:
            continue

        face_box = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

        head_direction = match_head_direction(
            face_box,
            head_directions,
        )

        emotion = item.get("dominant_emotion", "unknown")
        emotion_scores = item.get("emotion", {})

        confidence = round(
            float(emotion_scores.get(emotion, 0.0)),
            2,
        )

        # Direction provides only a rough attention indicator.
        # Eye gaze, eye openness, and movement are not yet measured.
        if head_direction == "Facing Forward":
            attention_status = "Possibly attentive"
        elif head_direction in (
            "Looking Left",
            "Looking Right",
            "Looking Up",
            "Looking Down",
        ):
            attention_status = "Possible looking away"
        else:
            attention_status = "Unknown"

        predictions.append({
            "face_id": len(predictions) + 1,
            "emotion": emotion,
            "confidence": confidence,
            "head_direction": head_direction,
            "attention_status": attention_status,
            "attention_score": None,
            "box": face_box,
        })

    return predictions


# --------------------------------------------------
# AGGREGATE RESULTS
# --------------------------------------------------


def build_response(media_type, faces, **extra):
    # This measures the proportion of detected faces oriented forward.
    # It is a head-direction proxy, not a true attention measurement.
    known_directions = [
        face for face in faces
        if face.get("head_direction") in (
            "Facing Forward",
            "Looking Left",
            "Looking Right",
            "Looking Up",
            "Looking Down",
        )
    ]

    if known_directions:
        forward_count = sum(
            face["head_direction"] == "Facing Forward"
            for face in known_directions
        )
        attention_percentage = round(
            100 * forward_count / len(known_directions), 1
        )
    else:
        attention_percentage = None

    return {
        "media_type": media_type,
        "people_detected": len(faces),
        "faces": faces,
        "attention_percentage": attention_percentage,
        # Engagement cannot be measured from emotion and head direction alone.
        "engagement_percentage": None,
        **extra,
    }



# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# IMAGE / VIDEO ANALYSIS
# --------------------------------------------------

@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    filename = (file.filename or "").lower()
    content_type = file.content_type or ""

    is_image = (
        content_type.startswith("image/")
        or filename.endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    )

    is_video = (
        content_type.startswith("video/")
        or filename.endswith(
            (".mp4", ".avi", ".mov", ".mkv", ".webm")
        )
    )

    if is_image:
        frame = cv2.imdecode(
            np.frombuffer(content, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not read this image.",
            )

        image_height, image_width = frame.shape[:2]

        try:
            faces = detect_emotions(frame)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Image analysis failed: {exc}",
            ) from exc

        return build_response(
            "image",
            faces,
            image_width=image_width,
            image_height=image_height,
        )

    if is_video:
        extension = os.path.splitext(filename)[1] or ".mp4"
        temp_path = None
        capture = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=extension,
                delete=False,
            ) as temp:
                temp.write(content)
                temp_path = temp.name

            capture = cv2.VideoCapture(temp_path)

            if not capture.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail="Could not open this video.",
                )

            fps = capture.get(cv2.CAP_PROP_FPS)
            step = max(1, int(fps)) if fps > 0 else 15

            best_faces = []
            frame_number = 0
            sampled_frames = 0

            while sampled_frames < 10:
                ok, frame = capture.read()

                if not ok:
                    break

                if frame_number % step == 0:
                    sampled_frames += 1

                    try:
                        faces = detect_emotions(frame)

                        if len(faces) > len(best_faces):
                            best_faces = faces
                    except Exception as exc:
                        print(
                            f"Skipping frame {frame_number}: {exc}"
                        )

                frame_number += 1

            if sampled_frames == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No readable video frames were found.",
                )

            return build_response(
                "video",
                best_faces,
                sampled_frames=sampled_frames,
            )

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Video analysis failed: {exc}",
            ) from exc
        finally:
            if capture is not None:
                capture.release()

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    raise HTTPException(
        status_code=400,
        detail="Please upload a supported image or video.",
    )
