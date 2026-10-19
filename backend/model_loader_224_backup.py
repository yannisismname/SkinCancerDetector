"""Model loading, preprocessing, prediction and Grad-CAM explanations.

Expected files (resolved relative to the repo root):
    model/model.h5      - trained Keras classifier (224x224 RGB input, softmax output)
    model/classes.json  - JSON list of class names matching the model's output order
"""
import base64
import io
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)  # (height, width) — must match training


class ModelLoader:
    def __init__(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = repo_root / "model" / "model.h5"
        classes_path = repo_root / "model" / "classes.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Train the model with training/skin_cancer_training.ipynb and "
                "place model.h5 and classes.json in the model/ folder."
            )
        if not classes_path.exists():
            raise FileNotFoundError(f"Class list not found at {classes_path}.")

        logger.info("Loading model from %s", model_path)
        self.model = load_model(str(model_path))

        with open(classes_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        # Make sure the model is built so .inputs / .output are available
        # (Keras 3 raises on .output for a never-called Sequential otherwise).
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

        output_units = self.model.output_shape[-1]
        if output_units != len(self.class_names):
            raise ValueError(
                f"Model outputs {output_units} classes but classes.json lists "
                f"{len(self.class_names)}. They must match — re-export classes.json "
                "from the training notebook."
            )
        logger.info("Model ready: %d classes -> %s", len(self.class_names), self.class_names)

    # ------------------------------------------------------------------ #
    def preprocess(self, img_bytes: bytes) -> np.ndarray:
        """Decode uploaded bytes -> model-ready batch (no temp files)."""
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
        arr = np.asarray(img, dtype=np.float32)
        arr = preprocess_input(arr)  # same preprocessing as training
        return np.expand_dims(arr, axis=0)

    def original_array(self, img_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
        return np.asarray(img, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _predict_proba(self, img_bytes: bytes) -> np.ndarray:
        """Average prediction over lossless augmented views (test-time
        augmentation). Free ~1-2% accuracy, no retraining needed."""
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
        views = [
            img,
            img.transpose(Image.FLIP_LEFT_RIGHT),
            img.transpose(Image.ROTATE_90),
            img.transpose(Image.ROTATE_270),
        ]
        batch = np.stack(
            [preprocess_input(np.asarray(v, dtype=np.float32)) for v in views],
            axis=0,
        )
        probs = np.asarray(self.model.predict(batch, verbose=0))
        return probs.mean(axis=0)

    def predict(self, img_bytes: bytes):
        probs = self._predict_proba(img_bytes).ravel()
        index = int(np.argmax(probs))
        if index >= len(self.class_names):
            # Should be impossible after the __init__ check; guard anyway.
            logger.error("Predicted index %d out of range (%d classes)", index, len(self.class_names))
            return f"class_{index}", index, float(probs[index])
        return self.class_names[index], index, float(probs[index])

    def predict_all(self, img_bytes: bytes):
        probs = self._predict_proba(img_bytes).ravel()
        order = np.argsort(probs)[::-1]
        return [
            {"label": self.class_names[i] if i < len(self.class_names) else f"class_{i}",
             "confidence": float(probs[i])}
            for i in order
        ]

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _find_last_conv(self, container):
        """Deepest Conv2D layer, looking inside nested models too
        (e.g. a MobileNetV2 submodel inside a Sequential)."""
        last = None
        for layer in container.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last = layer
            if hasattr(layer, "layers"):
                nested = self._find_last_conv(layer)
                if nested is not None:
                    last = nested
        return last

    def explain(self, img_bytes: bytes) -> str:
        """Grad-CAM heatmap overlaid on the original image, returned as a
        base64 data URI the frontend can display directly."""
        x = self.preprocess(img_bytes)

        # Case 1: transfer-learning shape — Sequential([Functional_base, head...]).
        # The conv tensor lives inside the Functional, so run Grad-CAM in two
        # stages: base -> conv features, then the head layers by hand.
        base, base_idx = None, None
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.Model) and not isinstance(layer, tf.keras.Sequential):
                base, base_idx = layer, i
                break

        with tf.GradientTape() as tape:
            if base is not None:
                last_conv = self._find_last_conv(base)
                if last_conv is None:
                    raise RuntimeError("No Conv2D layer found; cannot build Grad-CAM.")
                grad_base = tf.keras.Model(base.inputs, [last_conv.output, base.outputs[0]])
                conv_out, h = grad_base(x, training=False)
                for layer in self.model.layers[base_idx + 1:]:
                    h = layer(h, training=False)
                preds = h
            else:
                # Case 2: flat model — everything reachable from the top level.
                last_conv = self._find_last_conv(self.model)
                if last_conv is None:
                    raise RuntimeError("No Conv2D layer found; cannot build Grad-CAM.")
                grad_model = tf.keras.Model(
                    self.model.inputs, [last_conv.output, self.model.outputs[0]])
                conv_out, preds = grad_model(x, training=False)

            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE[1], IMG_SIZE[0]))
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        original = self.original_array(img_bytes)
        overlay = np.uint8(0.6 * original + 0.4 * heatmap)

        _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
