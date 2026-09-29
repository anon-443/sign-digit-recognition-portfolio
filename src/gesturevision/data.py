"""Load locally collected samples and split by recording session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from gesturevision.config import DEFAULT_SEQUENCE_LENGTH
from gesturevision.features import validate_sequence


@dataclass(frozen=True)
class GestureData:
    sequences: np.ndarray
    labels: np.ndarray
    sessions: np.ndarray
    class_names: list[str]


def load_recordings(
    data_dir: str | Path, sequence_length: int = DEFAULT_SEQUENCE_LENGTH
) -> GestureData:
    files = sorted(Path(data_dir).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No .npz recordings found under {data_dir}. Collect clips with gesture-collect first."
        )
    sequences, labels, sessions = [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as sample:
            required = {"features", "label", "session_id"}
            if not required.issubset(sample.files):
                raise ValueError(f"{path} must contain {sorted(required)}")
            features = validate_sequence(sample["features"], sequence_length)
            label = str(sample["label"].item())
            session_id = str(sample["session_id"].item())
            if not label or not session_id:
                raise ValueError(f"{path} has an empty label or session_id")
            sequences.append(features)
            labels.append(label)
            sessions.append(session_id)
    class_names = sorted(set(labels))
    return GestureData(
        sequences=np.stack(sequences).astype(np.float32),
        labels=np.asarray(labels),
        sessions=np.asarray(sessions),
        class_names=class_names,
    )


def split_by_session(
    labels: np.ndarray,
    sessions: np.ndarray,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    sessions = np.asarray(sessions)
    if labels.ndim != 1 or sessions.ndim != 1 or len(labels) != len(sessions):
        raise ValueError("labels and sessions must be one-dimensional arrays of equal length")
    if not (
        0 < test_fraction < 1
        and 0 < validation_fraction < 1
        and test_fraction + validation_fraction < 1
    ):
        raise ValueError("test and validation fractions must be positive and sum to less than 1")
    if len(np.unique(sessions)) < 3:
        raise ValueError("at least 3 distinct recording sessions are required for group holdouts")

    all_indices = np.arange(len(labels))
    train_val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(
            all_indices, labels, sessions
        )
    )
    relative_val_fraction = validation_fraction / (1.0 - test_fraction)
    train_rel, val_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=relative_val_fraction, random_state=seed + 1).split(
            train_val_idx, labels[train_val_idx], sessions[train_val_idx]
        )
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    expected_classes = set(np.unique(labels))
    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        present = set(np.unique(labels[indices]))
        missing = expected_classes - present
        if missing:
            raise ValueError(
                f"{name} split misses {len(missing)} gesture class(es); collect more balanced "
                "sessions or choose another --seed"
            )
    partitions = [set(sessions[idx]) for idx in (train_idx, val_idx, test_idx)]
    if (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    ):
        raise RuntimeError("session leakage detected between train/validation/test")
    return train_idx, val_idx, test_idx


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, label_ids: np.ndarray, augment: bool = False):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels = torch.as_tensor(label_ids, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        sequence = self.sequences[index].clone()
        if self.augment:
            sequence = sequence * float(np.random.uniform(0.95, 1.05))
            sequence = sequence + torch.randn_like(sequence) * 0.006
        return sequence, self.labels[index]
