"""
utils/classifier.py
────────────────────
Wraps the EfficientNetV2B0 Keras model trained in
KYC_Identification_for_Aadhar__Pan_and_Passport.ipynb.

Model file expected: models/best_model.keras
Class order (alphabetical, as produced by image_dataset_from_directory):
    0 → Aadhaar
    1 → Negative
    2 → PAN
    3 → Passport
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

CLASS_NAMES: List[str] = ["Aadhaar", "Negative", "PAN", "Passport"]

# Path to the worker script sitting next to this file
_WORKER = str(Path(__file__).parent / "tflite_worker.py")


class DocumentClassifier:

    MODEL_FILENAME = "kyc_classifier.tflite"

    def __init__(self, models_dir: Path):
        self._model_path = models_dir / self.MODEL_FILENAME
        self._loaded = False
        self._load()

    def _load(self):
        if not self._model_path.exists():
            log.warning(f"Classifier not found at {self._model_path}")
            return
        # Verify the worker + model work by doing a tiny probe call
        try:
            dummy = Image.new("RGB", (4, 4), color=128)
            self._run_subprocess(dummy)
            self._loaded = True
            log.info(f"Classifier ready (subprocess mode) | {self._model_path.name}")
        except Exception as exc:
            log.error(f"Classifier probe failed: {exc}")

    def is_loaded(self) -> bool:
        return self._loaded

    def _run_subprocess(self, pil_img: Image.Image) -> List[float]:
        # Encode image as base64 so we can pass it through stdin
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = json.dumps({
            "model_path": str(self._model_path),
            "image_b64":  image_b64,
        }).encode()

        result = subprocess.run(
            [sys.executable, _WORKER],
            input=payload,
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"TFLite worker crashed:\n{result.stderr.decode()}"
            )

        response = json.loads(result.stdout.decode())

        if "error" in response:
            raise RuntimeError(f"TFLite worker error: {response['error']}")

        return response["probs"]

    def predict(
        self, pil_img: Image.Image
    ) -> Tuple[str, float, Dict[str, float]]:
        if not self.is_loaded():
            raise RuntimeError(f"'{self.MODEL_FILENAME}' not found in models/")

        probs = self._run_subprocess(pil_img)

        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        doc_type   = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"

        all_probs = {
            name: round(float(p), 6)
            for name, p in zip(CLASS_NAMES[:len(probs)], probs)
        }
        return doc_type, confidence, all_probs
