"""
utils/aadhaar_model.py
────────────────────────
Aadhaar GNN model as described in Aadhaar_GNN_model.ipynb.

Feature vector (400 dims):
    name_embedding  (384)  – SentenceTransformer "all-MiniLM-L6-v2"
    gender           (1)   – Male=0, Female=1, Other=2, Unknown=3
    day              (1)
    month            (1)
    year             (1)
    aadhaar_digits  (12)   – each digit of the 12-digit Aadhaar number
    ─────────────────────
    TOTAL           400

OCR: openbharatocr.front_aadhaar(image_path)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict

import numpy as np

from .base_gnn import BaseGNN

log = logging.getLogger(__name__)

# SentenceTransformer is loaded lazily
_st_model = None

def _get_st():
    global _st_model
    if _st_model is None:
        from fastembed import TextEmbedding
        _st_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _st_model

class AadhaarGNN(BaseGNN):
    """GNN anomaly detector for Aadhaar cards."""

    MODEL_FILENAME      = "aadhaar_gnn_model.pth"
    SCALER_FILENAME     = "aadhaar_scaler.pkl"
    THRESHOLD_FILENAME  = "aadhaar_threshold.json"
    RECORDS_FILENAME    = "aadhaar_card_data.json"
    EMBEDDINGS_FILENAME = "aadhaar_train_embeddings.npy"
    FEATURE_DIM         = 400

    # ── OCR ─────────────────────────────────────────────────────────────────────
    def run_ocr(self, image_path: str) -> Dict[str, str]:
        from .ocr_runner import run_ocr_subprocess
        fields = run_ocr_subprocess("aadhaar", image_path)
        return {
            "Full Name":          fields.get("Full Name", ""),
            "Date/Year of Birth": fields.get("Date/Year of Birth", ""),
            "Gender":             fields.get("Gender", ""),
            "Aadhaar Number":     fields.get("Aadhaar Number", ""),
        }

    # ── Feature engineering ─────────────────────────────────────────────────────

    def _build_features(self, rec: Dict) -> np.ndarray:
        st   = _get_st()
        name = rec.get("Full Name") or ""

        # fastembed returns a generator — convert to list then stack
        name_vec = np.array(list(st.embed([name])))    # (1, 384)

        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        gender = gender_map.get(rec.get("Gender", ""), 3)

        day = month = year = 0
        dob = rec.get("Date/Year of Birth", "") or ""
        if "/" in dob:
            parts = dob.split("/")
            if len(parts) == 3:
                try:
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    pass
            elif len(parts) == 2:
                try:
                    month, year = int(parts[0]), int(parts[1])
                except ValueError:
                    pass
        elif re.fullmatch(r"\d{4}", dob):
            year = int(dob)

        aadhaar_digits = [0] * 12
        a = (rec.get("Aadhaar Number", "") or "").replace(" ", "")
        if a.isdigit() and len(a) == 12:
            aadhaar_digits = [int(d) for d in a]

        numeric = np.array(
            [[gender, day, month, year] + aadhaar_digits], dtype=float
        )
        return np.concatenate([name_vec, numeric], axis=1)   # (1, 400)
