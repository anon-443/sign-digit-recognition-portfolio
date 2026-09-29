import numpy as np
import pytest
import torch

from gesturevision.data import SequenceDataset, load_recordings, split_by_session
from gesturevision.evaluation import calculate_metrics
from gesturevision.features import normalize_landmarks, validate_sequence
from gesturevision.model import GestureGRU


def test_landmark_normalization_is_translation_and_scale_invariant():
    rng = np.random.default_rng(8)
    points = rng.normal(size=(21, 3)).astype(np.float32)
    points[9] += np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    baseline = normalize_landmarks(points)
    transformed = normalize_landmarks(points * 4.2 + np.asarray([5, -2, 9], dtype=np.float32))
    assert np.allclose(baseline, transformed, atol=1e-5)
    assert baseline.shape == (63,)


def test_feature_validation_rejects_bad_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match="expected shape"):
        normalize_landmarks(np.zeros((20, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_sequence(np.full((30, 63), np.nan, dtype=np.float32), 30)


def test_session_split_is_disjoint_and_keeps_all_classes_in_each_partition():
    labels = np.tile(np.asarray(["fist", "palm"]), 8)
    sessions = np.repeat([f"session_{i}" for i in range(8)], 2)
    train, validation, test = split_by_session(labels, sessions, seed=42)
    groups = [set(sessions[idx]) for idx in (train, validation, test)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    for indices in (train, validation, test):
        assert set(labels[indices]) == {"fist", "palm"}


def test_split_requires_multiple_sessions():
    with pytest.raises(ValueError, match="at least 3 distinct"):
        split_by_session(np.asarray(["fist", "palm"]), np.asarray(["one", "two"]))


def test_metrics_and_sequence_dataset():
    metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0, 1])
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    dataset = SequenceDataset(
        np.zeros((2, 30, 63), dtype=np.float32), np.asarray([0, 1]), augment=True
    )
    sequence, label = dataset[0]
    assert sequence.shape == (30, 63)
    assert label.dtype == torch.long


def test_gru_returns_one_logit_per_class():
    model = GestureGRU(num_classes=8)
    logits = model(torch.zeros(3, 30, 63))
    assert logits.shape == (3, 8)


def test_load_recordings_reads_label_and_session_metadata(tmp_path):
    sample_dir = tmp_path / "session_01" / "wave"
    sample_dir.mkdir(parents=True)
    np.savez_compressed(
        sample_dir / "clip.npz",
        features=np.zeros((30, 63), dtype=np.float32),
        label=np.asarray("wave"),
        session_id=np.asarray("session_01"),
    )
    data = load_recordings(tmp_path)
    assert data.sequences.shape == (1, 30, 63)
    assert data.labels.tolist() == ["wave"]
    assert data.sessions.tolist() == ["session_01"]
    assert data.class_names == ["wave"]
