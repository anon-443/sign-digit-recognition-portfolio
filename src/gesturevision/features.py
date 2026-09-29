"""Pure NumPy feature utilities, independent from webcam/MediaPipe runtime."""

from __future__ import annotations

import numpy as np

from gesturevision.config import FEATURE_DIM, LANDMARK_COUNT


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize 21 xyz landmarks to a wrist-centered, scale-invariant vector.

    Coordinates are translated so landmark 0 (wrist) is the origin and divided
    by wrist-to-middle-MCP length. This removes image-space translation and most
    hand-size variation, but intentionally does not rotate or mirror the hand.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"expected shape ({LANDMARK_COUNT}, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("landmarks must contain only finite coordinates")
    centered = points - points[0]
    scale = float(np.linalg.norm(centered[9]))
    if scale < 1e-6:
        scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-6:
        raise ValueError("landmark geometry is degenerate; cannot normalize zero-scale hand")
    return (centered / scale).reshape(FEATURE_DIM).astype(np.float32)


def validate_sequence(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    expected = (sequence_length, FEATURE_DIM)
    if array.shape != expected:
        raise ValueError(f"expected sequence shape {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("sequence contains NaN or infinite values")
    return array
