"""Collect labeled landmark sequences from a webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from gesturevision.config import DEFAULT_GESTURES, DEFAULT_SEQUENCE_LENGTH
from gesturevision.tracker import HandLandmarkTracker


def collect(args: argparse.Namespace) -> None:
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam device {args.camera}")
    tracker = HandLandmarkTracker(args.model)
    labels = DEFAULT_GESTURES
    selected = 0
    recording = False
    buffer: list[np.ndarray] = []
    target = Path(args.output_dir) / args.session_id
    target.mkdir(parents=True, exist_ok=True)
    counts = {label: len(list((target / label).glob("*.npz"))) for label in labels}
    started = time.monotonic()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, _ = tracker.extract(rgb, int(time.monotonic() * 1000))
            if recording:
                if features is None:
                    buffer.clear()
                else:
                    buffer.append(features)
                if len(buffer) >= args.sequence_length:
                    label = labels[selected]
                    label_dir = target / label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    path = label_dir / f"{args.session_id}_{label}_{counts[label]:04d}.npz"
                    np.savez_compressed(
                        path,
                        features=np.stack(buffer[: args.sequence_length]).astype(np.float32),
                        label=np.asarray(label),
                        session_id=np.asarray(args.session_id),
                    )
                    counts[label] += 1
                    print(f"saved {path}")
                    buffer.clear()
                    recording = False

            color = (40, 220, 80) if features is not None else (30, 30, 240)
            cv2.putText(
                frame,
                f"Label [{selected + 1}]: {labels[selected]}",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
            )
            cv2.putText(
                frame,
                "  ".join(f"{i + 1}:{name} ({counts[name]})" for i, name in enumerate(labels)),
                (18, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 230, 230),
                1,
            )
            instruction = (
                f"Capturing {len(buffer)}/{args.sequence_length} - keep hand visible"
                if recording
                else f"Press 1-{len(labels)} to choose label; SPACE capture; q quit"
            )
            cv2.putText(
                frame,
                instruction,
                (18, frame.shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Gesture dataset collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if not recording and ord("1") <= key < ord("1") + len(labels):
                selected = key - ord("1")
            elif not recording and key == ord(" "):
                recording = True
                buffer.clear()
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
    print(f"collection ended after {(time.monotonic() - started) / 60:.1f} minutes; data: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--session-id", required=True, help="unique ID for this capture session/person"
    )
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sequence_length < 5:
        raise SystemExit("sequence-length must be at least 5")
    collect(args)


if __name__ == "__main__":
    main()
