"""
utils/ocr_runner.py
────────────────────
Calls ocr_worker.py in a subprocess so easyocr/openbharatocr never
share a process with the GNN's PyTorch session.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_WORKER = str(Path(__file__).parent / "ocr_worker.py")


def run_ocr_subprocess(doc_type: str, image_path: str) -> dict:
    """
    doc_type : "aadhaar" | "pan" | "passport"
    Returns the fields dict, or an empty dict on failure.
    """
    payload = json.dumps({
        "doc_type":   doc_type,
        "image_path": image_path,
    }).encode()

    try:
        result = subprocess.run(
            [sys.executable, _WORKER],
            input=payload,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            log.warning(f"OCR worker stderr: {result.stderr.decode()[:300]}")

        response = json.loads(result.stdout.decode())

        if "error" in response:
            log.warning(f"OCR worker error: {response['error']}")
            return {}

        fields = response.get("fields", {})
        if "_error" in fields:
            log.warning(f"OCR internal error: {fields['_error']}")
            fields.pop("_error")

        return fields

    except Exception as exc:
        log.warning(f"OCR subprocess failed: {type(exc).__name__}: {exc}")
        return {}
