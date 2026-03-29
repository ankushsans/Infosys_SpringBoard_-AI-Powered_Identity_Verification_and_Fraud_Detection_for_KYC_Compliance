"""
utils/forgery_detector.py
──────────────────────────
HOG-based structural forgery detection.
Directly from Adhaar_Forgery_Detection.ipynb — extended to all 3 doc types.

Requires one genuine reference image per document type to create master templates.
Run: python create_master_templates.py --aadhaar ref.jpg --pan ref.jpg --passport ref.jpg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from skimage.feature import hog

log = logging.getLogger(__name__)

TARGET_SIZE = (250, 400)   # (height, width) — same as notebook

# Default thresholds per doc type.
# Override by editing models/<doc>_forgery_threshold.json after calibration.
DEFAULT_THRESHOLDS = {
    "Aadhaar":  0.32,
    "PAN":      0.32,
    "Passport": 0.35,
}

TEMPLATE_FILES = {
    "Aadhaar":  "aadhaar_master_hog.npy",
    "PAN":      "pan_master_hog.npy",
    "Passport": "passport_master_hog.npy",
}

THRESHOLD_FILES = {
    "Aadhaar":  "aadhaar_forgery_threshold.json",
    "PAN":      "pan_forgery_threshold.json",
    "Passport": "passport_forgery_threshold.json",
}


def extract_structural_fingerprint(image: np.ndarray) -> np.ndarray:
    """
    Extracts HOG feature vector from a BGR image.
    Identical to the notebook implementation.
    """
    resized = cv2.resize(image, (TARGET_SIZE[1], TARGET_SIZE[0]))   # cv2 wants (W, H)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys",
    )
    return features


def create_master_template(image_path: str, save_path: str) -> np.ndarray:
    """
    Creates and saves a HOG master template from a reference genuine image.
    Call once per document type via create_master_templates.py.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    vec = extract_structural_fingerprint(img)
    np.save(save_path, vec)
    log.info(f"Master template saved → {save_path}")
    return vec


class ForgeryDetector:
    """
    Loads master templates for all document types and scores
    uploaded images against them.
    """

    def __init__(self, models_dir: Path):
        self._models_dir = models_dir
        self._templates: Dict[str, np.ndarray] = {}
        self._thresholds: Dict[str, float]     = {}
        self._load_all()

    def _load_all(self):
        import json
        for doc_type, tpl_file in TEMPLATE_FILES.items():
            tpl_path = self._models_dir / tpl_file
            if tpl_path.exists():
                self._templates[doc_type] = np.load(str(tpl_path))
                log.info(f"ForgeryDetector: loaded template for {doc_type}")
            else:
                log.warning(
                    f"ForgeryDetector: no master template for {doc_type} "
                    f"({tpl_path}). Run create_master_templates.py first."
                )

            thresh_path = self._models_dir / THRESHOLD_FILES[doc_type]
            if thresh_path.exists():
                with open(thresh_path) as f:
                    self._thresholds[doc_type] = json.load(f).get(
                        "threshold", DEFAULT_THRESHOLDS[doc_type]
                    )
            else:
                self._thresholds[doc_type] = DEFAULT_THRESHOLDS[doc_type]

    def is_available(self, doc_type: str) -> bool:
        return doc_type in self._templates

    def analyze(self, image_path: str, doc_type: str) -> Dict:
        """
        Compares the uploaded image to the master template.

        Returns
        -------
        {
          "available":       bool   – False if no master template loaded
          "distance_score":  float  – 0.0 = identical, 1.0 = completely different
          "threshold":       float
          "is_forged":       bool
          "verdict":         str    – "AUTHENTIC" | "FORGED" | "UNAVAILABLE"
        }
        """
        if not self.is_available(doc_type):
            return {
                "available":      False,
                "distance_score": None,
                "threshold":      None,
                "is_forged":      None,
                "verdict":        "UNAVAILABLE",
                "message":        f"No master template for {doc_type}. "
                                  f"Run create_master_templates.py to enable forgery detection.",
            }

        img = cv2.imread(image_path)
        if img is None:
            return {
                "available":      True,
                "distance_score": None,
                "threshold":      None,
                "is_forged":      None,
                "verdict":        "ERROR",
                "message":        "Could not decode image for forgery analysis.",
            }

        test_vec    = extract_structural_fingerprint(img)
        master_vec  = self._templates[doc_type]
        distance    = float(cosine(master_vec, test_vec))
        threshold   = self._thresholds[doc_type]
        is_forged   = distance >= threshold

        return {
            "available":      True,
            "distance_score": round(distance, 6),
            "threshold":      round(threshold, 6),
            "is_forged":      is_forged,
            "verdict":        "FORGED" if is_forged else "AUTHENTIC",
        }
