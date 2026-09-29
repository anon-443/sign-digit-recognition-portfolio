"""Browser-camera and upload demo for static sign-language digit recognition."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from gesturevision.digit_inference import DigitImagePredictor

st.set_page_config(page_title="Sign Digit Vision", page_icon="✋", layout="wide")
st.title("Sign Digit Vision")
st.caption("Static handshape classifier for ten sign-language digit classes (0–9)")
st.warning(
    "Portfolio research demo only. This dataset is small, uses centered 100×100 images, and has "
    "no public participant IDs; the holdout is not person-independent. This is not an ASL translator, "
    "accessibility service, or production model."
)

configured_model = os.environ.get("GESTURE_DIGITS_MODEL")
if configured_model:
    options = {Path(configured_model).parent.name: configured_model}
else:
    options = {
        "MobileNetV3-Small": "artifacts/digits/mobilenet/model_bundle.pt",
        "EfficientNet-B0": "artifacts/digits/efficientnet/model_bundle.pt",
    }
available = {name: path for name, path in options.items() if Path(path).is_file()}
if not available:
    st.info("No trained model found. Run the two training commands in the README first.")
    st.stop()


@st.cache_resource
def load_predictor(path: str) -> DigitImagePredictor:
    return DigitImagePredictor(path)


model_name = st.selectbox("Model backbone", list(available))
predictor = load_predictor(available[model_name])
st.caption(
    f"{model_name}. The displayed confidence is an uncalibrated model score; "
    "camera/background shift can reduce accuracy."
)
source_mode = st.radio("Frame source", ["Upload image", "Take a camera photo"], horizontal=True)
if source_mode == "Upload image":
    image_file = st.file_uploader("Upload a handshape image", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("Center one clear handshape in the frame")

if image_file:
    image = Image.open(image_file).convert("RGB")
    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input frame", use_container_width=True)
    with right:
        result = predictor.predict(image, top_k=3)
        st.subheader("Top predictions")
        for rank, item in enumerate(result["predictions"], start=1):
            st.write(f"**{rank}. Digit {item['digit']}** — {item['confidence']:.1%}")
        st.caption(f"Classifier-only CPU/GPU inference: {result['latency_ms']:.1f} ms")
