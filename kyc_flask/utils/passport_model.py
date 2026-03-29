"""
utils/passport_model.py
─────────────────────────
Passport GNN model as described in passport_GNN_model.ipynb.

Feature vector (1157 dims):
    name_embedding         (384)  – SentenceTransformer "all-MiniLM-L6-v2"
    birth_place_embedding  (384)  – SentenceTransformer
    issue_place_embedding  (384)  – SentenceTransformer
    sex                     (1)   – M=1, F=2, Unknown=0
    nationality_num         (1)   – sum(ord(c)) % 1000
    day                     (1)
    month                   (1)
    year                    (1)
    ─────────────────────────────
    TOTAL                 1157

OCR: openbharatocr does not have a passport extractor, so we use
     pytesseract + a regex-based MRZ parser as a best-effort fallback.
     If mrz is installed (pip install mrz) we use it for higher accuracy.
"""

from __future__ import annotations

import logging
import re
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

# ── MRZ Parsing helpers ────────────────────────────────────────────────────────

def _parse_mrz_date(mrz_date: str):
    """Convert 6-digit MRZ date (YYMMDD) → (day, month, year)."""
    try:
        yy, mm, dd = int(mrz_date[:2]), int(mrz_date[2:4]), int(mrz_date[4:])
        # Heuristic: years >= 30 assumed 1900s, else 2000s
        year = 1900 + yy if yy >= 30 else 2000 + yy
        return dd, mm, year
    except Exception:
        return 0, 0, 0


def _ocr_with_tesseract(image_path: str) -> Dict[str, str]:
    """
    Attempt to read the MRZ from the passport image using pytesseract
    and extract structured fields.
    """
    result = {
        "given_name":     "",
        "surname":        "",
        "date_of_birth":  "",
        "sex":            "",
        "nationality":    "",
        "place_of_birth": "",
        "place_of_issue": "",
    }

    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        raw_text = pytesseract.image_to_string(img, config="--psm 6")
    except Exception as exc:
        log.warning(f"Tesseract OCR failed: {exc}")
        return result

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    # Look for MRZ lines (44-char lines of uppercase + < + digits)
    mrz_lines = [l for l in lines if re.fullmatch(r"[A-Z0-9<]{44}", l)]

    if len(mrz_lines) >= 2:
        line1, line2 = mrz_lines[0], mrz_lines[1]

        # Line 1: P<NATIONALITY<<SURNAME<<GIVEN_NAMES<...
        m = re.match(r"P[A-Z<]([A-Z]{3})<<([A-Z<]+)", line1)
        if m:
            result["nationality"] = m.group(1).replace("<", "")
            name_part = m.group(2)
            parts     = name_part.split("<<", 1)
            result["surname"]    = parts[0].replace("<", " ").strip()
            result["given_name"] = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""

        # Line 2: …DOB…SEX…
        if len(line2) >= 30:
            dob_str = line2[13:19]
            sex_chr = line2[20] if len(line2) > 20 else ""
            dd, mm, yyyy = _parse_mrz_date(dob_str)
            if yyyy:
                result["date_of_birth"] = f"{dd:02d} {list(MONTH_MAP.keys())[mm-1]} {yyyy}"
            result["sex"] = sex_chr if sex_chr in ("M", "F") else ""

    return result


def _ocr_with_mrz_library(image_path: str) -> Dict[str, str]:
    """Try the `mrz` library for structured MRZ extraction."""
    result = {
        "given_name":     "",
        "surname":        "",
        "date_of_birth":  "",
        "sex":            "",
        "nationality":    "",
        "place_of_birth": "",
        "place_of_issue": "",
    }
    try:
        from mrz.checker.td3 import TD3CodeChecker
        import pytesseract
        from PIL import Image

        img     = Image.open(image_path).convert("RGB")
        raw_txt = pytesseract.image_to_string(img, config="--psm 6")

        # Find two consecutive 44-char MRZ lines
        lines     = raw_txt.splitlines()
        mrz_lines = [l.strip() for l in lines if re.fullmatch(r"[A-Z0-9<]{44}", l.strip())]

        if len(mrz_lines) >= 2:
            checker = TD3CodeChecker(mrz_lines[0] + "\n" + mrz_lines[1], check_expiry=False)
            f       = checker.fields()

            given_name = f.name or ""
            surname    = f.surname or ""

            result.update({
                "given_name":    given_name,
                "surname":       surname,
                "date_of_birth": f.birth_date or "",
                "sex":           f.sex or "",
                "nationality":   f.nationality or "",
            })
    except Exception as exc:
        log.warning(f"mrz-library extraction failed: {exc}. Falling back to tesseract.")
        result = _ocr_with_tesseract(image_path)

    return result


class PassportGNN(BaseGNN):
    """GNN anomaly detector for passports."""

    MODEL_FILENAME      = "passport_gnn_model.pth"
    SCALER_FILENAME     = "passport_scaler.pkl"
    THRESHOLD_FILENAME  = "passport_threshold.json"
    RECORDS_FILENAME    = "passport_data.json"
    EMBEDDINGS_FILENAME = "passport_train_embeddings.npy"
    FEATURE_DIM         = 1157

    # ── OCR ─────────────────────────────────────────────────────────────────────
    def run_ocr(self, image_path: str) -> Dict[str, str]:
        from .ocr_runner import run_ocr_subprocess
        fields = run_ocr_subprocess("passport", image_path)
        return {
            "given_name":     fields.get("given_name", ""),
            "surname":        fields.get("surname", ""),
            "date_of_birth":  fields.get("date_of_birth", ""),
            "sex":            fields.get("sex", ""),
            "nationality":    fields.get("nationality", ""),
            "place_of_birth": fields.get("place_of_birth", ""),
            "place_of_issue": fields.get("place_of_issue", ""),
        }

    # ── Feature engineering ─────────────────────────────────────────────────────
    def _build_features(self, rec: Dict) -> np.ndarray:
        st = _get_st()

        given       = rec.get("given_name", "") or ""
        surname     = rec.get("surname", "") or ""
        full_name   = f"{given} {surname}".strip()
        birth_place = rec.get("place_of_birth", "") or ""
        issue_place = rec.get("place_of_issue", "") or ""

        name_vec        = np.array(list(st.embed([full_name])))      # (1, 384)
        birth_place_vec = np.array(list(st.embed([birth_place])))    # (1, 384)
        issue_place_vec = np.array(list(st.embed([issue_place])))    # (1, 384)

        sex_map = {"M": 1, "F": 2}
        sex = sex_map.get(rec.get("sex", ""), 0)

        nationality = rec.get("nationality", "") or ""
        nat_num = sum(ord(c) for c in nationality) % 1000 if nationality else 0

        day = month = year = 0
        dob = rec.get("date_of_birth", "") or ""
        if dob:
            try:
                parts = dob.split()
                if len(parts) == 3:
                    day   = int(parts[0])
                    month = MONTH_MAP.get(parts[1].upper(), 0)
                    year  = int(parts[2])
            except Exception:
                pass

        numeric = np.array([[sex, nat_num, day, month, year]], dtype=float)

        return np.concatenate(
            [name_vec, birth_place_vec, issue_place_vec, numeric], axis=1
        )   # (1, 1157)

