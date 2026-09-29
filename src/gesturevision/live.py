"""Run the trained gesture classifier on a webcam stream."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from gesturevision.config import FEATURE_DIM
from gesturevision.model import GestureGRU
from gesturevision.tracker import HandLandmarkTracker


class LivePredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto", smoothing: int = 5):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.class_names = bundle["class_names"]
        self.sequence_length = int(bundle["sequence_length"])
        self.model = GestureGRU(
            len(self.class_names),
            FEATURE_DIM,
            bundle["hidden_size"],
            bundle["layers"],
            bundle["dropout"],
        ).to(self.device)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()
        self.windows: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.probabilities: deque[np.ndarray] = deque(maxlen=smoothing)

    def clear(self) -> None:
        self.windows.clear()
        self.probabilities.clear()

    def predict_window(self, features: np.ndarray) -> tuple[str, float, float] | None:
        self.windows.append(features.astype(np.float32))
        if len(self.windows) < self.sequence_length:
            return None
        sequence = torch.as_tensor(
            np.stack(self.windows)[None], dtype=torch.float32, device=self.device
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(self.model(sequence), dim=1)[0].cpu().numpy()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        self.probabilities.append(probs)
        smoothed = np.mean(self.probabilities, axis=0)
        index = int(np.argmax(smoothed))
        return self.class_names[index], float(smoothed[index]), latency_ms


def run_live(args: argparse.Namespace) -> None:
    predictor = LivePredictor(args.model, args.device, args.smoothing)
    tracker = HandLandmarkTracker(args.landmarker)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        tracker.close()
        raise RuntimeError(f"Could not open webcam device {args.camera}")
    started = time.monotonic()
    frames = 0
    current = ("waiting for hand", 0.0, 0.0)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, _result = tracker.extract(rgb, int(time.monotonic() * 1000))
            if features is None:
                predictor.clear()
                current = ("no hand", 0.0, 0.0)
            else:
                prediction = predictor.predict_window(features)
                if prediction is not None:
                    current = prediction
            label, confidence, latency_ms = current
            display_label = label if confidence >= args.threshold else "unknown"
            color = (60, 220, 80) if confidence >= args.threshold else (40, 180, 240)
            cv2.putText(
                frame,
                f"{display_label}  {confidence:.1%}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"GRU: {latency_ms:.1f} ms | windows {len(predictor.windows)}/{predictor.sequence_length}",
                (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )
            frames += 1
            fps = frames / max(time.monotonic() - started, 1e-6)
            cv2.putText(
                frame,
                f"camera FPS: {fps:.1f} | q quit | r reset",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Real-Time Gesture Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                predictor.clear()
                current = ("waiting for hand", 0.0, 0.0)
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--landmarker", default="models/hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--smoothing", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0 or args.smoothing < 1:
        raise SystemExit("threshold must be between 0 and 1; smoothing must be positive")
    run_live(args)


if __name__ == "__main__":
    main()
