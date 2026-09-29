"""Shared constants for collection, training, and inference."""

DEFAULT_GESTURES = (
    "fist",
    "open_palm",
    "thumbs_up",
    "peace",
    "point",
    "ok",
    "wave",
)
LANDMARK_COUNT = 21
COORDINATES_PER_LANDMARK = 3
FEATURE_DIM = LANDMARK_COUNT * COORDINATES_PER_LANDMARK
DEFAULT_SEQUENCE_LENGTH = 30
