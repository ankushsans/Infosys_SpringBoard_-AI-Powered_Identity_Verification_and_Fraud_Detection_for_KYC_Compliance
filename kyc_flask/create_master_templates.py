"""
create_master_templates.py
───────────────────────────
Creates HOG master templates from genuine reference images.
Run ONCE before starting the server.

Usage:
    python create_master_templates.py \
        --aadhaar   path/to/genuine_aadhaar.jpg \
        --pan       path/to/genuine_pan.jpg \
        --passport  path/to/genuine_passport.jpg

You only need to provide the types you have reference images for.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

MODELS_DIR  = Path("models")
TARGET_SIZE = (250, 400)

TEMPLATE_FILES = {
    "aadhaar":  "aadhaar_master_hog.npy",
    "pan":      "pan_master_hog.npy",
    "passport": "passport_master_hog.npy",
}

THRESHOLD_FILES = {
    "aadhaar":  "aadhaar_forgery_threshold.json",
    "pan":      "pan_forgery_threshold.json",
    "passport": "passport_forgery_threshold.json",
}

DEFAULT_THRESHOLDS = {
    "aadhaar":  0.32,
    "pan":      0.32,
    "passport": 0.35,
}


def extract_hog(image: np.ndarray) -> np.ndarray:
    resized  = cv2.resize(image, (TARGET_SIZE[1], TARGET_SIZE[0]))
    gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys",
    )
    return features


def create_template(image_path: str, doc_type: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load image at {image_path}")
        sys.exit(1)

    vec      = extract_hog(img)
    out_path = MODELS_DIR / TEMPLATE_FILES[doc_type]
    np.save(str(out_path), vec)
    print(f"  Template saved → {out_path}  (vector dim: {vec.shape[0]})")

    # Save default threshold
    thresh_path = MODELS_DIR / THRESHOLD_FILES[doc_type]
    if not thresh_path.exists():
        with open(thresh_path, "w") as f:
            json.dump({"threshold": DEFAULT_THRESHOLDS[doc_type]}, f, indent=2)
        print(f"  Threshold saved → {thresh_path}  (default: {DEFAULT_THRESHOLDS[doc_type]})")
    else:
        print(f"  Threshold already exists at {thresh_path} — not overwritten")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aadhaar",  default=None, help="Path to genuine Aadhaar image")
    parser.add_argument("--pan",      default=None, help="Path to genuine PAN image")
    parser.add_argument("--passport", default=None, help="Path to genuine Passport image")
    args = parser.parse_args()

    if not any([args.aadhaar, args.pan, args.passport]):
        parser.print_help()
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    if args.aadhaar:
        print(f"Creating Aadhaar master template from {args.aadhaar} …")
        create_template(args.aadhaar, "aadhaar")

    if args.pan:
        print(f"Creating PAN master template from {args.pan} …")
        create_template(args.pan, "pan")

    if args.passport:
        print(f"Creating Passport master template from {args.passport} …")
        create_template(args.passport, "passport")

    print("\nDone. Restart the Flask server to load the new templates.")


if __name__ == "__main__":
    main()
