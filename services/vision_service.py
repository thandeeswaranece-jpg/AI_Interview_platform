"""
services/vision_service.py

Local computer vision using OpenCV + MediaPipe's CURRENT Tasks API
(FaceLandmarker). mediapipe.solutions.face_mesh is gone in current
MediaPipe releases - this uses the model-asset-based Tasks API instead.

Requires assets/face_landmarker.task to be downloaded manually (see
README). If it's missing, vision analysis is disabled for the whole
run - never faked - and the rest of the interview continues normally.
"""

import os
import math
import logging
import asyncio
import numpy as np
import cv2

logger = logging.getLogger("interview.vision")

MODEL_ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")

_landmarker = None
_load_attempted = False
_load_error: str | None = None


def _try_load():
    global _landmarker, _load_attempted, _load_error
    if _load_attempted:
        return
    _load_attempted = True
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not os.path.exists(MODEL_ASSET_PATH):
            raise FileNotFoundError(
                f"Model not found at {os.path.abspath(MODEL_ASSET_PATH)}. "
                "Download it as described in the README setup steps."
            )

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe FaceLandmarker loaded.")
    except Exception as e:
        _load_error = str(e)
        logger.warning(f"Vision analysis disabled - could not load FaceLandmarker: {e}")


def is_available() -> bool:
    _try_load()
    return _landmarker is not None


def unavailable_reason() -> str | None:
    _try_load()
    return _load_error


def _rotation_matrix_to_euler_degrees(rot: np.ndarray) -> tuple:
    """Approximate decomposition relative to MediaPipe's canonical face
    model space - a consistent, directionally-correct signal, NOT a
    calibrated anatomical measurement."""
    pitch = math.atan2(rot[2, 1], rot[2, 2])
    yaw = math.atan2(-rot[2, 0], math.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    roll = math.atan2(rot[1, 0], rot[0, 0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class FaceMetricsExtractor:
    """One instance per question - tracks frame-to-frame movement and blink edges."""

    def __init__(self):
        self._prev_nose = None
        self._movement_samples: list[float] = []
        self._blink_was_high = False
        self.blink_count = 0
        self.frames_processed = 0
        self.frames_with_face = 0

    def _process_blocking(self, jpeg_bytes: bytes) -> dict:
        _try_load()
        self.frames_processed += 1

        if _landmarker is None:
            return {"available": False, "reason": _load_error}

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"available": True, "face_detected": False, "error": "could not decode frame"}

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {"available": True, "face_detected": False}

        self.frames_with_face += 1
        landmarks = result.face_landmarks[0]

        # Nose tip = landmark index 1 in MediaPipe's face mesh topology.
        nose = landmarks[1]
        nose_pos = np.array([nose.x, nose.y])
        movement = 0.0
        if self._prev_nose is not None:
            movement = float(np.linalg.norm(nose_pos - self._prev_nose))
        self._prev_nose = nose_pos
        self._movement_samples.append(movement)

        blink_score = None
        if result.face_blendshapes:
            categories = {c.category_name: c.score for c in result.face_blendshapes[0]}
            left = categories.get("eyeBlinkLeft", 0.0)
            right = categories.get("eyeBlinkRight", 0.0)
            blink_score = (left + right) / 2.0
            is_high = blink_score > 0.5
            if is_high and not self._blink_was_high:
                self.blink_count += 1
            self._blink_was_high = is_high

        yaw = pitch = roll = None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0])
            rot = matrix[:3, :3]
            pitch, yaw, roll = _rotation_matrix_to_euler_degrees(rot)

        return {
            "available": True,
            "face_detected": True,
            "head_yaw_deg": yaw,
            "head_pitch_deg": pitch,
            "head_roll_deg": roll,
            "blink_score": blink_score,
            "movement": movement,
        }

    async def process_frame(self, jpeg_bytes: bytes) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_blocking, jpeg_bytes)

    def summary(self) -> dict:
        face_visibility_ratio = (
            self.frames_with_face / self.frames_processed if self.frames_processed else 0.0
        )
        movement_variance = float(np.var(self._movement_samples)) if self._movement_samples else 0.0
        return {
            "frames_processed": self.frames_processed,
            "frames_with_face": self.frames_with_face,
            "face_visibility_ratio": round(face_visibility_ratio, 3),
            "blink_count": self.blink_count,
            "movement_variance": round(movement_variance, 6),
        }