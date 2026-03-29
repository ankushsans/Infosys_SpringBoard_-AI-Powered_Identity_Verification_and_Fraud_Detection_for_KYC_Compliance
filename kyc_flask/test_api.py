#!/usr/bin/env python3
"""
test_api.py
────────────
Quick smoke-test for the KYC Flask API.

Usage:
    # With a real image
    python test_api.py --image path/to/aadhaar.jpg

    # Health check only
    python test_api.py --health-only

    # Generate a synthetic test image (white rectangle — just to check endpoints)
    python test_api.py --synthetic

The script prints a formatted JSON summary of every response.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "http://localhost:5000"


# ── Helpers ────────────────────────────────────────────────────────────────────

def pprint(data: dict):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def check_health():
    print("── GET /api/health ──────────────────────────────")
    resp = requests.get(f"{BASE_URL}/api/health", timeout=10)
    pprint(resp.json())
    print()
    return resp.json().get("status") == "ok"


def analyse(image_path: str | None = None, synthetic: bool = False):
    print("── POST /api/kyc ────────────────────────────────")
    t0 = time.perf_counter()

    if synthetic or image_path is None:
        # Create a tiny white image with "AADHAAR" text so the classifier
        # has at least something to look at.
        img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((60, 100), "SYNTHETIC TEST", fill=(50, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        files = {"image": ("test.jpg", buf, "image/jpeg")}
    else:
        files = {"image": open(image_path, "rb")}

    try:
        resp = requests.post(f"{BASE_URL}/api/kyc", files=files, timeout=120)
        elapsed = time.perf_counter() - t0
        print(f"Status : {resp.status_code}  |  Time: {elapsed:.2f}s")
        pprint(resp.json())
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Is it running?")
        sys.exit(1)
    finally:
        if image_path:
            files["image"].close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KYC API smoke test")
    parser.add_argument("--url",          default=BASE_URL, help="Base URL of the API")
    parser.add_argument("--image",        default=None,     help="Path to image file")
    parser.add_argument("--synthetic",    action="store_true", help="Use a synthetic test image")
    parser.add_argument("--health-only",  action="store_true", help="Only run the health check")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url.rstrip("/")

    ok = check_health()
    if not ok:
        print("⚠  Health check failed — some models may not be loaded correctly.")

    if args.health_only:
        return

    analyse(image_path=args.image, synthetic=args.synthetic or (args.image is None))


if __name__ == "__main__":
    main()
