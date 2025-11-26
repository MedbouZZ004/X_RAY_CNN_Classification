"""
GUI for fast X-ray pneumonia screening using the pretrained CNN.

Features:
- Drag-and-drop or file picker upload via Gradio.
- Loads `best_covid_model.h5` once at startup for snappy responses.
- Outputs predicted class plus confidence chart.
"""

from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_covid_model.h5"
IMG_SIZE: Tuple[int, int] = (224, 224)
CLASS_NAMES = ("NORMAL", "PNEUMONIA")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH.name}' not found in {BASE_DIR}. "
        "Train the model first or copy the .h5 file next to this script."
    )

# Load once to avoid repeated initialization overhead.
MODEL = tf.keras.models.load_model(MODEL_PATH)


def _prepare_image(image: Image.Image) -> np.ndarray:
    """Resize, convert and normalize image for model inference."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)


def predict(image: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Return class label text + class probabilities."""
    if image is None:
        return "Please upload an X-ray image.", {cls: 0.0 for cls in CLASS_NAMES}

    pil_image = Image.fromarray(image.astype("uint8"), "RGB")
    prepared = _prepare_image(pil_image)
    probability = float(MODEL.predict(prepared, verbose=0)[0][0])

    pneu_score = probability
    normal_score = 1.0 - probability
    label = CLASS_NAMES[1] if pneu_score >= 0.5 else CLASS_NAMES[0]
    confidence = pneu_score if label == CLASS_NAMES[1] else normal_score

    summary = f"{label} ({confidence * 100:.1f}% confidence)"
    scores = {CLASS_NAMES[0]: normal_score, CLASS_NAMES[1]: pneu_score}
    return summary, scores


demo = gr.Interface(
    title="Chest X-ray Pneumonia Detector",
    fn=predict,
    inputs=gr.Image(type="numpy", label="Drop or upload a chest X-ray"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=2, label="Confidence"),
    ],
    description=(
        "The model reuses the pretrained MobileNetV2-based CNN from training. "
        "Images are analyzed instantly with no retraining required."
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()

