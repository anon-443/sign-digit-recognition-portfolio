"""Single-frame inference for trained sign-language digit classifiers."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from gesturevision.digit_models import build_digit_model


class DigitImagePredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cpu":
            torch.set_num_threads(max(1, int(os.environ.get("GESTURE_DIGITS_CPU_THREADS", "4"))))
        self.bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.class_names = self.bundle["class_names"]
        self.model = build_digit_model(
            self.bundle["model_name"], self.bundle["num_classes"], pretrained=False
        )
        self.model.load_state_dict(self.bundle["state_dict"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.bundle["image_size"], self.bundle["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.bundle["mean"], std=self.bundle["std"]),
            ]
        )
        warmup = torch.zeros(
            1,
            3,
            self.bundle["image_size"],
            self.bundle["image_size"],
            device=self.device,
        )
        with torch.inference_mode():
            self.model(warmup)

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        values, indices = torch.topk(probabilities, k=min(top_k, len(self.class_names)))
        ranked = [
            {"digit": self.class_names[int(index)], "confidence": float(value)}
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        return {
            "predictions": ranked,
            "probabilities": probabilities.detach().cpu().tolist(),
            "latency_ms": latency_ms,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to model_bundle.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    predictor = DigitImagePredictor(args.model, args.device)
    with Image.open(args.image) as image:
        result = predictor.predict(image, top_k=args.top_k)
    for item in result["predictions"]:
        print(f"{item['digit']}: {item['confidence']:.4f}")
    print(f"model inference latency: {result['latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
