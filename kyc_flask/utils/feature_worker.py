"""
utils/feature_worker.py
────────────────────────
Subprocess worker: builds feature vectors using sentence-transformers
in an isolated process.
"""

import sys
import os
import json
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Point to local cached model — no network calls
LOCAL_ST_MODEL = str(Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2")


def main():
    try:
        req        = json.loads(sys.stdin.buffer.read())
        class_name = req["class_name"]
        rec        = req["rec"]

        models_dir = Path(__file__).parent.parent / "models"

        # Monkey-patch _get_st in each module to use local model
        import utils.aadhaar_model as am
        import utils.pan_model     as pm
        import utils.passport_model as psm

        def _get_st_local():
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(LOCAL_ST_MODEL)

        am._get_st  = _get_st_local
        pm._get_st  = _get_st_local
        psm._get_st = _get_st_local

        if class_name == "AadhaarGNN":
            from utils.aadhaar_model import AadhaarGNN
            wrapper = AadhaarGNN(models_dir=models_dir)
        elif class_name == "PanGNN":
            from utils.pan_model import PanGNN
            wrapper = PanGNN(models_dir=models_dir)
        elif class_name == "PassportGNN":
            from utils.passport_model import PassportGNN
            wrapper = PassportGNN(models_dir=models_dir)
        else:
            print(json.dumps({"error": f"Unknown class: {class_name}"}))
            sys.stdout.flush()
            return

        features = wrapper._build_features(rec)   # (1, FEATURE_DIM)
        print(json.dumps({"features": features.flatten().tolist()}))
        sys.stdout.flush()

    except Exception as exc:
        import traceback
        print(json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()
        }))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
