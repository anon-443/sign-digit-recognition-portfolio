"""Train an image classifier for the public 0–9 sign-language digit dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from gesturevision.digit_data import (
    DATASET_COMMIT,
    DATASET_REPO,
    DigitImageDataset,
    download_dataset,
    image_manifest,
    stratified_group_split,
)
from gesturevision.digit_models import SUPPORTED_IMAGE_MODELS, build_digit_model
from gesturevision.evaluation import calculate_metrics, save_diagnostics

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transforms_for(image_size: int):
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train, evaluate


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(logits.argmax(1).detach().cpu().tolist())
    return total_loss / len(loader.dataset), targets_all, predictions_all


def _latency_percentiles(model, device, image_size: int, repeats: int = 40) -> dict[str, float]:
    tensor = torch.zeros(1, 3, image_size, image_size, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(tensor)
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(tensor)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000)
    return {
        "p50_ms_per_image": float(np.percentile(durations, 50)),
        "p95_ms_per_image": float(np.percentile(durations, 95)),
        "fps_at_p50_model_only": float(1000 / np.percentile(durations, 50)),
        "warmup_runs": 5,
        "timed_runs": repeats,
    }


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir() or not any(data_dir.glob("[0-9]/*")):
        data_dir = download_dataset(args.data_root)
    manifest = image_manifest(data_dir)
    train_idx, validation_idx, test_idx = stratified_group_split(
        manifest.labels, manifest.sha256_groups, args.seed
    )
    train_transform, eval_transform = transforms_for(args.image_size)
    train_data = DigitImageDataset(manifest.paths, manifest.labels, train_idx, train_transform)
    validation_data = DigitImageDataset(
        manifest.paths, manifest.labels, validation_idx, eval_transform
    )
    test_data = DigitImageDataset(manifest.paths, manifest.labels, test_idx, eval_transform)

    train_labels = manifest.labels[train_idx]
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(manifest.class_names)), y=train_labels
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(class_weights[train_labels], dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device == "cuda",
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device == "cuda",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_digit_model(args.model, len(manifest.class_names), pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    bundle_path = output_dir / "model_bundle.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_val, pred_val = _run_epoch(model, validation_loader, criterion, device)
        val_f1 = float(
            f1_score(
                y_val,
                pred_val,
                labels=np.arange(len(manifest.class_names)),
                average="macro",
                zero_division=0,
            )
        )
        scheduler.step(val_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1, best_epoch, stale_epochs = val_f1, epoch, 0
            torch.save(
                {
                    "model_name": args.model,
                    "num_classes": len(manifest.class_names),
                    "class_names": manifest.class_names,
                    "image_size": args.image_size,
                    "mean": MEAN,
                    "std": STD,
                    "state_dict": model.state_dict(),
                    "best_validation_macro_f1": best_f1,
                    "best_epoch": best_epoch,
                },
                bundle_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break

    checkpoint = torch.load(bundle_path, map_location=device, weights_only=False)
    model = build_digit_model(checkpoint["model_name"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    test_loss, y_test, pred_test = _run_epoch(model, test_loader, criterion, device)
    metrics = calculate_metrics(y_test, pred_test, list(range(len(manifest.class_names))))
    latency = _latency_percentiles(model, device, args.image_size)
    metrics.update(
        {
            "model_name": args.model,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "test_loss": test_loss,
            "test_examples": len(test_idx),
            "train_examples": len(train_idx),
            "validation_examples": len(validation_idx),
            "class_count": len(manifest.class_names),
            "exact_duplicate_file_groups": len(manifest.paths) - len(set(manifest.sha256_groups)),
            "model_only_latency": latency,
            "split_protocol": (
                "nested StratifiedGroupKFold grouped by exact image SHA-256; "
                "source repository provides no participant IDs"
            ),
            "selection_protocol": "best validation macro-F1; test evaluated once after selection",
        }
    )
    save_diagnostics(
        metrics,
        manifest.class_names,
        history,
        output_dir,
        title="Sign-language digit held-out test confusion matrix",
    )
    configuration = vars(args).copy()
    configuration.update(
        {
            "resolved_device": device,
            "data_dir": str(data_dir),
            "dataset_repository": DATASET_REPO,
            "dataset_commit": DATASET_COMMIT,
            "class_names": manifest.class_names,
            "split_sizes": {
                "train": len(train_idx),
                "validation": len(validation_idx),
                "test": len(test_idx),
            },
            "exact_duplicate_file_groups": metrics["exact_duplicate_file_groups"],
            "participant_ids_available": False,
            "selection_metric": "validation macro-F1",
            "test_protocol": "one evaluation on seeded stratified exact-duplicate-grouped holdout",
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2)
    )
    return bundle_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_IMAGE_MODELS, default="mobilenet_v3_small")
    parser.add_argument("--data-dir", default="data/sign-language-digits/images")
    parser.add_argument("--data-root", default="data/sign-language-digits")
    parser.add_argument("--output-dir", default="artifacts/digits/run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.image_size < 64:
        raise SystemExit("epochs/patience/batch-size/threads must be positive and image-size >= 64")
    train(args)


if __name__ == "__main__":
    main()
