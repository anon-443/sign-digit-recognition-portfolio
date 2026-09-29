"""Metrics and diagnostic plots for multi-class sequence classification."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> dict:
    labels = list(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_diagnostics(
    metrics: dict,
    class_names: list[str],
    history: list[dict],
    output_dir: Path,
    title: str = "Session-held-out gesture test confusion matrix",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["class", "precision", "recall", "f1-score", "support"]
        )
        writer.writeheader()
        for index, name in enumerate(class_names):
            row = metrics["per_class"].get(str(index), {})
            writer.writerow(
                {
                    "class": name,
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "f1-score": row.get("f1-score", 0.0),
                    "support": row.get("support", 0),
                }
            )

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(
        figsize=(max(8, len(class_names) * 0.6), max(7, len(class_names) * 0.52))
    )
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    if history:
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
        axes[0].set(title="Cross-entropy loss", xlabel="Epoch")
        axes[0].legend()
        axes[1].plot(epochs, [row["val_macro_f1"] for row in history])
        axes[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(output_dir / "training_curves.png", dpi=180)
        plt.close(fig)
