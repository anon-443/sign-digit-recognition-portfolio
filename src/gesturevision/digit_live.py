"""Run trained static digit-handshape inference continuously on a local webcam."""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
from PIL import Image

from gesturevision.digit_inference import DigitImagePredictor


def run(args: argparse.Namespace) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Install the optional webcam dependencies with: pip install -e '.[webcam]'"
        ) from exc
    predictor = DigitImagePredictor(args.model, args.device)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam device {args.camera}")
    history: deque[np.ndarray] = deque(maxlen=args.smoothing)
    frame_count = 0
    started = time.monotonic()
    latency_ms = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = predictor.predict(Image.fromarray(rgb), top_k=1)
            history.append(np.asarray(prediction["probabilities"], dtype=np.float32))
            probabilities = np.mean(history, axis=0)
            index = int(np.argmax(probabilities))
            confidence = float(probabilities[index])
            digit = predictor.class_names[index] if confidence >= args.threshold else "unknown"
            latency_ms = float(prediction["latency_ms"])
            frame_count += 1
            fps = frame_count / max(time.monotonic() - started, 1e-6)
            color = (70, 220, 80) if digit != "unknown" else (40, 180, 240)
            cv2.putText(
                frame,
                f"digit: {digit}  confidence: {confidence:.1%}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"model: {latency_ms:.1f} ms | smoothed frames: {len(history)} | camera FPS: {fps:.1f}",
                (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Static 0-9 demo only | q quit | r reset smoothing",
                (20, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Sign-Language Digit Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                history.clear()
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--smoothing", type=int, default=5)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0 or args.smoothing < 1:
        raise SystemExit("threshold must be in [0, 1] and smoothing must be positive")
    run(args)


if __name__ == "__main__":
    main()
