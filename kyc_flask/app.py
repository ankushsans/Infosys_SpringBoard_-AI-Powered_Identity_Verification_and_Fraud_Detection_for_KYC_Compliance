"""
KYC Compliance Flask Application
─────────────────────────────────
Integrates three GNN models (Aadhaar, PAN, Passport) with an
EfficientNetV2B0 document classifier.

Workflow:
  1. POST /api/kyc  →  Upload an image
  2. Classifier identifies document type (Aadhaar / PAN / Passport / Negative)
  3. OCR extracts structured fields from the document
  4. Appropriate GNN model scores anomaly & returns top-5 similar records
  5. Full result returned as JSON (and rendered on the UI)

Environment variables:
  KYC_MODELS_DIR  – path to models/  (default: ./models)
  KYC_UPLOAD_DIR  – path to uploads/ (default: ./uploads)
  KYC_MAX_MB      – max upload size in MB (default: 16)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]   = "-1"   # keep everything on CPU
import io
import logging
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

from utils.forgery_detector import ForgeryDetector
from utils.classifier import DocumentClassifier
from utils.aadhaar_model import AadhaarGNN
from utils.pan_model import PanGNN
from utils.passport_model import PassportGNN
# ── Configuration ──────────────────────────────────────────────────────────────
MODELS_DIR   = Path(os.getenv("KYC_MODELS_DIR",  "models"))
UPLOAD_DIR   = Path(os.getenv("KYC_UPLOAD_DIR",  "uploads"))
MAX_MB       = int(os.getenv("KYC_MAX_MB", "16"))
ALLOWED_EXTS = {"jpg", "jpeg", "png", "webp"}

MODELS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# ── Model loading (happens once at startup) ────────────────────────────────────
log.info("Loading models …")
classifier   = DocumentClassifier(models_dir=MODELS_DIR)
aadhaar_gnn  = AadhaarGNN(models_dir=MODELS_DIR)
pan_gnn      = PanGNN(models_dir=MODELS_DIR)
passport_gnn = PassportGNN(models_dir=MODELS_DIR)
forgery_detector = ForgeryDetector(models_dir=MODELS_DIR)

MODEL_MAP = {
    "Aadhaar":  aadhaar_gnn,
    "PAN":      pan_gnn,
    "Passport": passport_gnn,
}
log.info("All models loaded.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def _pil_from_upload(file_storage) -> Image.Image:
    return Image.open(io.BytesIO(file_storage.read())).convert("RGB")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/kyc", methods=["POST"])
def kyc_endpoint():
    """
    Multipart POST — field: 'image'

    Returns JSON:
    {
      "document_type": "Aadhaar" | "PAN" | "Passport" | "Negative",
      "confidence": float,
      "all_probs": {class: float, …},
      "ocr_fields": {field: value, …},
      "anomaly_score": float | null,
      "is_suspicious": bool | null,
      "threshold": float | null,
      "similar_records": [ {rank, node_id, similarity, record}, … ]
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' field in multipart form"}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not _allowed(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed extensions: {sorted(ALLOWED_EXTS)}"
        }), 415

    try:
        pil_img = _pil_from_upload(file)
    except Exception as exc:
        return jsonify({"error": f"Cannot decode image: {exc}"}), 400

    try:
        # ── Step 1 – Document classification ──────────────────────────────────
        doc_type, confidence, all_probs = classifier.predict(pil_img)
        log.info(f"Classified → {doc_type!r}  (conf={confidence:.3f})")

        if doc_type == "Negative":
            return jsonify({
                "document_type":   "Negative",
                "confidence":      round(confidence, 4),
                "all_probs":       all_probs,
                "message":         "Image does not appear to be a KYC document.",
                "ocr_fields":      {},
                "anomaly_score":   None,
                "is_suspicious":   None,
                "threshold":       None,
                "similar_records": [],
            })

        if doc_type not in MODEL_MAP:
            return jsonify({"error": f"Unrecognised document type: {doc_type}"}), 500

        tmp_path = UPLOAD_DIR / secure_filename(file.filename or "upload.jpg")
        pil_img.save(str(tmp_path))
        # ── Step 2 – OCR ───────────────────────────────────────────────────────

        gnn = MODEL_MAP[doc_type]

        try:
            ocr_fields = gnn.run_ocr(str(tmp_path))
            log.info(f"OCR fields: {ocr_fields}")
        except Exception as ocr_err:
            log.warning(f"OCR failed ({ocr_err}); using empty record.")
            ocr_fields = {}

# ── Step 3 – GNN anomaly detection ────────────────────────────────────
        log.info("Step 3: starting GNN inference …")
        try:
            score, is_suspicious, threshold, similar = gnn.infer(ocr_fields)
            log.info(f"Step 3 done: score={score:.5f}")
        except Exception as exc:
            log.error(f"Step 3 GNN inference crashed: {type(exc).__name__}: {exc}")
            import traceback; log.error(traceback.format_exc())
            score, is_suspicious, threshold, similar = 0.0, False, 0.5, []

        # ── Step 4 – Forgery detection ─────────────────────────────────────────
        log.info("Step 4: starting forgery detection …")
        try:
            forgery_result = forgery_detector.analyze(str(tmp_path), doc_type)
            log.info(f"Step 4 done: {forgery_result['verdict']}")
        except Exception as exc:
            log.error(f"Step 4 forgery crashed: {type(exc).__name__}: {exc}")
            forgery_result = {"available": False, "verdict": "ERROR",
                              "distance_score": None, "threshold": None, "is_forged": None}

        log.info("Returning response …")
        return jsonify({
            "document_type":   doc_type,
            "confidence":      round(confidence, 4),
            "all_probs":       all_probs,
            "ocr_fields":      ocr_fields,
            "anomaly_score":   round(score, 6),
            "is_suspicious":   is_suspicious,
            "threshold":       round(threshold, 6),
            "similar_records": similar,
            "forgery":         forgery_result,
        })


    except Exception:
        log.error(traceback.format_exc())
        return jsonify({"error": "Internal server error — check server logs"}), 500

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "classifier":   classifier.is_loaded(),
            "aadhaar_gnn":  aadhaar_gnn.is_loaded(),
            "pan_gnn":      pan_gnn.is_loaded(),
            "passport_gnn": passport_gnn.is_loaded(),
            "forgery": {
                "Aadhaar":  forgery_detector.is_available("Aadhaar"),
                "PAN":      forgery_detector.is_available("PAN"),
                "Passport": forgery_detector.is_available("Passport"),
            }
        },
        "config": {
            "models_dir":    str(MODELS_DIR.resolve()),
            "upload_dir":    str(UPLOAD_DIR.resolve()),
            "max_upload_mb": MAX_MB,
        }
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Maximum is {MAX_MB} MB."}), 413


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("KYC_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=debug)
