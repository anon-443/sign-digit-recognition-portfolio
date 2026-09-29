"""Pinned sign-language digit data acquisition and duplicate-aware splitting."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

DATASET_REPO = "ardamavi/Sign-Language-Digits-Dataset"
DATASET_COMMIT = "4b09433ef9d5c60602e8805bb54b8aa2428527f8"
DATASET_URL = f"https://codeload.github.com/{DATASET_REPO}/zip/{DATASET_COMMIT}"
DATASET_CITATION = "Mavi, A. (2020), arXiv:2011.08927"


@dataclass(frozen=True)
class ImageManifest:
    paths: list[Path]
    labels: np.ndarray
    class_names: list[str]
    sha256_groups: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(data_root: str | Path = "data/sign-language-digits") -> Path:
    """Fetch a pinned upstream ZIP, extract only Dataset/<digit> image files, return image root."""
    root = Path(data_root)
    image_root = root / "images"
    existing = list(image_root.glob("[0-9]/*")) if image_root.exists() else []
    if len(existing) >= 2000:
        return image_root

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"upstream-{DATASET_COMMIT}.zip"
    if not archive_path.is_file():
        request = Request(DATASET_URL, headers={"User-Agent": "portfolio-project/1.0"})
        temporary = archive_path.with_suffix(".zip.part")
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        temporary.replace(archive_path)
    archive_sha256 = _sha256(archive_path)

    image_root.mkdir(parents=True, exist_ok=True)
    class_counts: dict[str, int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        license_member = next((name for name in members if name.endswith("/LICENSE")), None)
        if license_member is None:
            raise RuntimeError("upstream archive has no LICENSE file")
        license_path = Path("licenses/DATASET_APACHE_2.0.txt")
        license_path.parent.mkdir(parents=True, exist_ok=True)
        license_path.write_bytes(archive.read(license_member))

        for member in members:
            parts = PurePosixPath(member).parts
            if "Dataset" not in parts:
                continue
            position = parts.index("Dataset")
            if len(parts) <= position + 2:
                continue
            class_name = parts[position + 1]
            suffix = Path(parts[-1]).suffix.lower()
            if class_name not in {str(i) for i in range(10)} or suffix not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue
            target = image_root / class_name / Path(*parts[position + 2 :])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    if len(class_counts) != 10 or sum(class_counts.values()) < 2000:
        raise RuntimeError(f"unexpected upstream dataset structure/counts: {class_counts}")
    manifest = {
        "repository": DATASET_REPO,
        "commit": DATASET_COMMIT,
        "archive_sha256": archive_sha256,
        "license": "Apache-2.0 (see licenses/DATASET_APACHE_2.0.txt)",
        "citation": DATASET_CITATION,
        "class_counts": dict(sorted(class_counts.items())),
        "image_count": sum(class_counts.values()),
    }
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return image_root


def image_manifest(image_root: str | Path) -> ImageManifest:
    """Read digit image paths/labels and exact-file hashes using ImageFolder-style folders."""
    root = Path(image_root)
    class_names = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if class_names != [str(index) for index in range(10)]:
        raise ValueError(f"expected digit class directories 0..9 under {root}; found {class_names}")
    class_to_id = {name: index for index, name in enumerate(class_names)}
    paths: list[Path] = []
    labels: list[int] = []
    groups: list[str] = []
    for class_name in class_names:
        for path in sorted((root / class_name).iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(path)
                labels.append(class_to_id[class_name])
                groups.append(_sha256(path))
    if len(paths) < 100:
        raise ValueError(f"too few digit images found under {root}: {len(paths)}")
    return ImageManifest(paths, np.asarray(labels, dtype=np.int64), class_names, groups)


def stratified_group_split(
    labels: np.ndarray, groups: list[str] | np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible 64/16/20 train/validation/test folds without exact-image overlap."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
        raise ValueError("labels and groups must be one-dimensional arrays of equal length")
    if len(labels) < 50 or len(np.unique(groups)) < 5:
        raise ValueError("need at least 50 images and five unique image groups")
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(np.zeros(len(labels)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(
        inner.split(
            np.zeros(len(train_val_idx)),
            labels[train_val_idx],
            groups[train_val_idx],
        )
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    partitions = (train_idx, val_idx, test_idx)
    group_sets = [set(groups[index]) for index in partitions]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("exact-image group leakage detected across partitions")
    all_labels = set(np.unique(labels))
    if any(set(np.unique(labels[index])) != all_labels for index in partitions):
        raise ValueError("each partition must contain all classes; adjust the seed or data")
    return partitions


class DigitImageDataset(Dataset):
    """PyTorch dataset wrapper for selected digit-image rows."""

    def __init__(self, paths: list[Path], labels: np.ndarray, indices: np.ndarray, transform=None):
        self.paths = paths
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        from PIL import Image

        position = int(self.indices[index])
        with Image.open(self.paths[position]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[position])


def ensure_dataset(data_dir: str | Path) -> Path:
    """Return the specified image root or download the pinned source to the default location."""
    path = Path(data_dir)
    if path.is_dir() and any(path.glob("[0-9]/*")):
        return path
    default_images = download_dataset()
    if Path(data_dir) == Path("data/sign-language-digits/images"):
        return default_images
    raise FileNotFoundError(f"digit image directory not found: {data_dir}")


def main() -> None:
    image_root = download_dataset()
    manifest = image_manifest(image_root)
    counts = {
        name: int(np.count_nonzero(manifest.labels == index))
        for index, name in enumerate(manifest.class_names)
    }
    print(
        json.dumps(
            {"images": len(manifest.paths), "class_counts": counts, "path": str(image_root)},
            indent=2,
        )
    )
