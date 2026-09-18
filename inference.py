"""
inference.py
------------
Loads the trained MobileNetV2 model and runs predictions on uploaded
retinal images. Mirrors the preprocessing used in training/train_model.py
so inference matches training distribution exactly.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import CLASS_NAMES, CLASS_INFO

_model = None


def get_model(model_path):
    """Lazy-load and cache the Keras model (avoids reloading per-request)."""
    global _model
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Run training/train_model.py first, or place a trained "
                ".h5 model at that path."
            )
        _model = tf.keras.models.load_model(model_path)
    return _model


def preprocess_image(path, img_size=224):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image — file may be corrupted or unsupported.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top:top + side, left:left + side]

    img = cv2.resize(img, (img_size, img_size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img


def predict(image_path, model_path, img_size=224):
    """
    Returns a dict:
        {
            "class": "Moderate",
            "risk": "Medium",
            "confidence": 87.32,
            "description": "...",
            "recommendation": "...",
            "all_scores": {"No DR": 2.1, "Mild": 5.4, ...}
        }
    """
    model = get_model(model_path)

    img = preprocess_image(image_path, img_size)
    img_batch = preprocess_input(img.astype(np.float32))
    img_batch = np.expand_dims(img_batch, axis=0)

    preds = model.predict(img_batch, verbose=0)[0]  # shape (5,)

    class_idx = int(np.argmax(preds))
    class_name = CLASS_NAMES[class_idx]
    confidence = float(preds[class_idx]) * 100

    info = CLASS_INFO[class_name]
    all_scores = {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASS_NAMES))}

    return {
        "class": class_name,
        "risk": info["risk"],
        "confidence": round(confidence, 2),
        "description": info["description"],
        "recommendation": info["recommendation"],
        "all_scores": all_scores,
    }
