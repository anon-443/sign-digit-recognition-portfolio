"""Train and evaluate a temporal gesture classifier on local session recordings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from gesturevision.config import DEFAULT_SEQUENCE_LENGTH, FEATURE_DIM
from gesturevision.data import SequenceDataset, load_recordings, split_by_session
from gesturevision.evaluation import calculate_metrics, save_diagnostics
from gesturevision.model import GestureGRU


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all, preds_all = [], []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(sequences)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            targets_all.extend(labels.detach().cpu().tolist())
            preds_all.extend(logits.argmax(1).detach().cpu().tolist())
    return total_loss / len(loader.dataset), targets_all, preds_all


def measure_model_latency(model, sequences: np.ndarray, device: str, warmup: int = 10) -> dict:
    model.eval()
    sample = torch.as_tensor(sequences[:1], dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(min(warmup, 10)):
            model(sample)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        measurements = []
        for sequence in sequences:
            one = torch.as_tensor(sequence[None], dtype=torch.float32, device=device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = __import__("time").perf_counter()
            model(one)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            measurements.append((__import__("time").perf_counter() - start) * 1000)
    return {
        "classifier_window_latency_ms_p50": float(np.percentile(measurements, 50)),
        "classifier_window_latency_ms_p95": float(np.percentile(measurements, 95)),
        "latency_note": "GRU-only, one 30-frame feature window; excludes camera and landmark tracking",
    }


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    data = load_recordings(args.data_dir, args.sequence_length)
    train_idx, val_idx, test_idx = split_by_session(
        data.labels,
        data.sessions,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    label_to_id = {name: index for index, name in enumerate(data.class_names)}
    label_ids = np.asarray([label_to_id[label] for label in data.labels], dtype=np.int64)
    train_ids = label_ids[train_idx]
    counts = np.bincount(train_ids, minlength=len(data.class_names))
    sample_weights = 1.0 / np.maximum(counts[train_ids], 1)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_idx),
        replacement=True,
    )
    train_data = SequenceDataset(data.sequences[train_idx], train_ids, augment=True)
    val_data = SequenceDataset(data.sequences[val_idx], label_ids[val_idx])
    test_data = SequenceDataset(data.sequences[test_idx], label_ids[test_idx])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = GestureGRU(
        len(data.class_names),
        input_size=FEATURE_DIM,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    best_f1, best_epoch, stale_epochs = -1.0, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, y_val, pred_val = run_epoch(model, val_loader, loss_fn, device)
        val_f1 = float(
            f1_score(
                y_val,
                pred_val,
                labels=np.arange(len(data.class_names)),
                average="macro",
                zero_division=0,
            )
        )
        scheduler.step(val_f1)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_f1}
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1, best_epoch, stale_epochs = val_f1, epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": data.class_names,
                    "sequence_length": args.sequence_length,
                    "feature_dim": FEATURE_DIM,
                    "hidden_size": args.hidden_size,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "best_validation_macro_f1": best_f1,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GestureGRU(
        len(checkpoint["class_names"]),
        FEATURE_DIM,
        checkpoint["hidden_size"],
        checkpoint["layers"],
        checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, y_test, pred_test = run_epoch(model, test_loader, loss_fn, device)
    metrics = calculate_metrics(y_test, pred_test, list(range(len(data.class_names))))
    metrics.update(
        {
            "model": "2-layer GRU over normalized MediaPipe landmark windows",
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "test_loss": test_loss,
            "test_examples": len(test_idx),
            "train_examples": len(train_idx),
            "validation_examples": len(val_idx),
            "recording_sessions": len(np.unique(data.sessions)),
            "classes": data.class_names,
            "selection_protocol": "checkpoint selected by validation macro-F1; held-out session test scored once",
        }
    )
    metrics.update(measure_model_latency(model, data.sequences[test_idx], device))
    save_diagnostics(metrics, data.class_names, history, output_dir)
    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_device": device,
            "split_sizes": {
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
            },
            "class_names": data.class_names,
            "session_grouping": True,
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2)
    )
    return checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="artifacts/gru")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.sequence_length < 1:
        raise SystemExit("epochs, batch-size, and sequence-length must be positive")
    train(args)


if __name__ == "__main__":
    main()
