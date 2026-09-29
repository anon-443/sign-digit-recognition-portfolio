# Real-Time Sign-Language Digit Recognition

A portfolio project for **static handshape classification of sign-language digits 0–9**. The measured workflow pins an open image dataset, compares two pretrained CNN backbones on the same duplicate-aware splits, selects checkpoints using validation macro-F1, reports class-wise diagnostics and CPU model latency, and provides upload/camera and local-webcam inference.

> **Scope:** The headline metrics below belong only to the static 10-class image classifier. This is not a full sign-language translator. The repository also retains an optional temporal MediaPipe/GRU track, but that extension needs user-recorded webcam sessions and has no trained checkpoint or measured result in this package.

## What is implemented

- Pinned dataset downloader for the Apache-2.0 [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset), with source revision and archive SHA-256 recorded locally.
- Duplicate-aware, nested `StratifiedGroupKFold` train/validation/test split grouped by exact image SHA-256; identical files cannot cross partitions.
- Transfer-learning comparison of **MobileNetV3-Small** and **EfficientNet-B0** with matched split, transforms, sampling, seed, and selection protocol.
- Validation-only checkpoint selection; held-out accuracy, balanced accuracy, macro/weighted F1, per-class CSV, confusion matrix, training curves, and model-only inference latency percentiles.
- Streamlit upload/browser-camera-photo demo with backbone selection, plus a local OpenCV webcam loop with temporal smoothing and a low-confidence `unknown` state.
- Retained 21-landmark, 30-frame GRU collection/training/inference tools for custom gesture classes, clearly separate from the measured digit-image results.

## Dataset and split protocol

The pinned upstream snapshot contains **2,062 RGB images** at 100×100 pixels across ten labels (0–9). The upstream repository describes images prepared by students at Turkey Ankara Ayrancı Anadolu High School and reports 218 participants and ten samples per student. This snapshot has per-class counts from 204 to 208. The images are not committed here; `data/` is ignored by Git, and the upstream Apache-2.0 license is included in `licenses/DATASET_APACHE_2.0.txt`.

The upstream repository does **not** provide participant IDs or an official participant-disjoint split. This project therefore uses nested, seeded `StratifiedGroupKFold`, grouping by exact file SHA-256. The split is **1,319 train / 330 validation / 413 test** with 41–42 test images per class and zero exact duplicate groups across partitions. The check prevents byte-identical-file leakage only; it cannot prevent different photos of the same participant from appearing in multiple splits. Accordingly, do not describe these results as participant-held-out or evidence of person-independent generalization. See [SOURCES.md](SOURCES.md) for the pinned commit, observed counts, license, and citation.

## Measured comparison

Both CPU runs used seed 42, 160×160 images, batch size 32, 8 maximum epochs, validation macro-F1 checkpoint selection, and the exact same held-out indices. The test set was scored once after checkpoint selection.

| Backbone | Best validation macro-F1 | Test accuracy | Test macro-F1 | Model-only CPU p50 | Model-only CPU p95 |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 0.9879 (epoch 6) | 0.9734 | 0.9733 | 4.95 ms/image | 8.25 ms/image |
| EfficientNet-B0 | 0.9970 (epoch 5) | 0.9927 | 0.9927 | 11.30 ms/image | 12.02 ms/image |

Latency was measured on the Sandbox CPU with four PyTorch threads after five warm-up forwards and 40 timed single-image forwards. It excludes image decoding, webcam capture, Streamlit, and network latency. **MobileNetV3-Small is initially selected in the demo**; choose EfficientNet-B0 in the model selector for the higher measured score. MobileNet is the lower-latency option. These metrics are a result on a small, repository-derived holdout—not an independent or person-disjoint benchmark.

## Setup, download, and training

Python 3.10+ is required. Install the base package and development tools, then download the pinned dataset:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
gesture-digits-download
```

Train both comparable backbones:

```bash
gesture-digits-train --model mobilenet_v3_small --data-dir data/sign-language-digits/images \\
  --epochs 8 --patience 3 --batch-size 32 --image-size 160 --workers 2 \\
  --threads 4 --seed 42 --device cpu --output-dir artifacts/digits/mobilenet

gesture-digits-train --model efficientnet_b0 --data-dir data/sign-language-digits/images \\
  --epochs 8 --patience 3 --batch-size 32 --image-size 160 --workers 2 \\
  --threads 4 --seed 42 --device cpu --output-dir artifacts/digits/efficientnet
```

Each output folder contains `model_bundle.pt`, `metrics.json`, `run_config.json`, `per_class_metrics.csv`, `confusion_matrix.png`, `training_curves.png`, and the training log. The pinned source archive, weights, and run artifacts stay out of the code repository.

## Inference and live demos

**Temporary public demo:** [Sign Digit Vision](https://8502-iob7qxpw0ti3uvx48v4ep-a7a4073f.us4.manus.computer/). The URL is health-checked, but is Sandbox-hosted and not a permanent deployment.

Run the browser upload/camera-photo demo from the project root after training:

```bash
streamlit run src/gesturevision/digit_app.py
```

Predict one image from the command line:

```bash
gesture-digits-predict --model artifacts/digits/efficientnet/model_bundle.pt \\
  --image path/to/handshape.jpg --top-k 3
```

For continuous **local** webcam inference, install the optional webcam dependencies and start the live loop:

```bash
python -m pip install -e '.[webcam]'
gesture-digits-live --model artifacts/digits/mobilenet/model_bundle.pt --camera 0 --threshold 0.45
```

Press `q` to quit or `r` to clear the five-frame smoothing buffer. This webcam path has not been camera-tested in the cloud Sandbox; validate it on the target device. Images submitted to the public demo are transmitted to the temporary Sandbox-hosted app for inference; the app code does not persist them. Run the app locally for images you do not want sent to that service.

## Optional temporal landmark/GRU extension

This separate seven-class track (`fist`, `open_palm`, `thumbs_up`, `peace`, `point`, `ok`, `wave`) uses MediaPipe 21-point landmarks and a 30-frame GRU. It needs your own labeled sessions; no raw camera recordings are included.

```bash
gesture-download-model --output models/hand_landmarker.task
gesture-collect --model models/hand_landmarker.task --session-id session_01 --output-dir data/raw
# Repeat every class across at least four distinct sessions
gesture-train --data-dir data/raw --epochs 60 --batch-size 64 --output-dir artifacts/gru
gesture-live --model artifacts/gru/best_model.pt --landmarker models/hand_landmarker.task --camera 0
```

The temporal extension's session split is designed to avoid session leakage, but no metrics should be claimed until it has been trained and measured on real, diverse sessions.

## Tests

```bash
pytest
ruff check src tests
```

Tests cover duplicate-group split isolation/reproducibility, class coverage, both model output heads, parsing, metrics, and core feature logic; they do not download pretrained weights, require a camera, or train models.

## Resume bullet

> Fine-tuned and compared MobileNetV3-Small and EfficientNet-B0 for 10-class sign-digit image recognition; selected EfficientNet-B0 by validation macro-F1 and achieved **99.27% macro-F1** on a 413-image exact-duplicate-grouped holdout, with **11.30 ms p50 model-only CPU inference**. Built reproducible evaluation artifacts and upload/camera/local-webcam inference paths.

**Interview caveat:** explicitly state that the dataset exposes no participant IDs, so the holdout is not person-independent; describe the measured scope as static digit handshapes rather than full sign-language translation.
