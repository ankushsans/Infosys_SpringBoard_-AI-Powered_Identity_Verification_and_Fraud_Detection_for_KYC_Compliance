"""
utils/pan_model.py
────────────────────
PAN Card GNN model as described in Pan_Card_GNN_model.ipynb.

Feature vector (782 dims):
    name_embedding         (384)  – SentenceTransformer "all-MiniLM-L6-v2"
    parent_name_embedding  (384)  – SentenceTransformer "all-MiniLM-L6-v2"
    gender                  (1)   – always 3 (Unknown) for PAN cards
    day                     (1)
    month                   (1)
    year                    (1)
    pan_digits             (10)   – each char of 10-char PAN (digit→int, letter→1–26)
    ────────────────────────────
    TOTAL                  782

OCR: openbharatocr.pan(image_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .base_gnn import BaseGNN

log = logging.getLogger(__name__)

_st_model = None

def _get_st():
    global _st_model
    if _st_model is None:
        from fastembed import TextEmbedding
        _st_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _st_model

class PanGNN(BaseGNN):
    """GNN anomaly detector for PAN cards."""

    MODEL_FILENAME      = "pan_gnn_model.pth"
    SCALER_FILENAME     = "pan_scaler.pkl"
    THRESHOLD_FILENAME  = "pan_threshold.json"
    RECORDS_FILENAME    = "pan_card_data.json"
    EMBEDDINGS_FILENAME = "pan_train_embeddings.npy"
    FEATURE_DIM         = 782

    # ── OCR ─────────────────────────────────────────────────────────────────────
    def run_ocr(self, image_path: str) -> Dict[str, str]:
        from .ocr_runner import run_ocr_subprocess
        fields = run_ocr_subprocess("pan", image_path)
        return {
            "Name":          fields.get("Name", ""),
            "Parent's Name": fields.get("Parent's Name", ""),
            "Date of Birth": fields.get("Date of Birth", ""),
            "PAN Number":    fields.get("PAN Number", ""),
        }

    # ── Feature engineering ─────────────────────────────────────────────────────
    
    def _build_features(self, rec: Dict) -> np.ndarray:
        st          = _get_st()
        name        = rec.get("Name", "") or ""
        parent_name = rec.get("Parent's Name", "") or ""

        name_vec   = np.array(list(st.embed([name])))          # (1, 384)
        parent_vec = np.array(list(st.embed([parent_name])))   # (1, 384)

        gender = 3
        day = month = year = 0
        dob = rec.get("Date of Birth", "") or ""
        if "/" in dob:
            parts = dob.split("/")
            if len(parts) == 3:
                try:
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    pass
        elif len(dob) == 4 and dob.isdigit():
            year = int(dob)

        pan_digits = [0] * 10
        pan = (rec.get("PAN Number", "") or "").replace(" ", "")
        if pan:
            encoded = []
            for c in pan:
                if c.isdigit():
                    encoded.append(int(c))
                elif c.isalpha():
                    encoded.append(ord(c.upper()) - 64)
            pan_digits = (encoded + [0] * 10)[:10]

        numeric = np.array(
            [[gender, day, month, year] + pan_digits], dtype=float
        )
        return np.concatenate([name_vec, parent_vec, numeric], axis=1)  # (1, 782)
