"""MediaPipe video-mode hand tracking with normalized skeleton features."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gesturevision.features import normalize_landmarks


class HandLandmarkTracker:
    def __init__(self, model_path: str | Path, num_hands: int = 1) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found at {path}; run gesture-download-model first."
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Install webcam dependencies: pip install -e '.[webcam]'") from exc
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract(
        self, rgb_frame: np.ndarray, timestamp_ms: int
    ) -> tuple[np.ndarray | None, object | None]:
        frame = np.asarray(rgb_frame, dtype=np.uint8)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame)
        result = self.detector.detect_for_video(image, int(timestamp_ms))
        if not result.hand_landmarks:
            return None, result
        points = np.asarray(
            [[point.x, point.y, point.z] for point in result.hand_landmarks[0]],
            dtype=np.float32,
        )
        return normalize_landmarks(points), result

    def close(self) -> None:
        self.detector.close()
