"""Download the official MediaPipe Hand Landmarker task bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download(output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as file:
            if response.status != 200:
                raise RuntimeError(f"model download returned HTTP {response.status}")
            file.write(response.read())
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("downloaded model bundle is unexpectedly small")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded MediaPipe Hand Landmarker model to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/hand_landmarker.task")
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
