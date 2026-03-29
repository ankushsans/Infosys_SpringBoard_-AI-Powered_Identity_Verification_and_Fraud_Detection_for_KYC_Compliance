"""
wsgi.py
────────
Gunicorn / uWSGI entry point.

Usage:
    gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 4 --timeout 120

Environment variables (optional):
    KYC_MODELS_DIR   – path to models/ directory   (default: ./models)
    KYC_UPLOAD_DIR   – path to uploads/ directory  (default: ./uploads)
    KYC_MAX_MB       – max upload size in MB        (default: 16)
    KYC_DEBUG        – set to "1" to enable debug   (default: off)
"""

import os
from pathlib import Path

# Override directories via environment before the app module is imported
os.environ.setdefault("KYC_MODELS_DIR",  str(Path(__file__).parent / "models"))
os.environ.setdefault("KYC_UPLOAD_DIR",  str(Path(__file__).parent / "uploads"))

from app import app  # noqa: E402  (import after env setup)

if __name__ == "__main__":
    # For local dev only — use `python wsgi.py` as an alternative to `python app.py`
    debug = os.getenv("KYC_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=debug)
