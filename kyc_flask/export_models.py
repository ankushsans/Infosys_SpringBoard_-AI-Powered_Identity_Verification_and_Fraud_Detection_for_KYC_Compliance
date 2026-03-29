"""
export_models.py
────────────────
Run this script ONCE after training (in the same environment as the notebooks,
or in Colab after the cells have executed).  It will:

  1. Load each trained GNN state-dict  (.pth already saved by notebooks)
  2. Save the sklearn StandardScaler   → models/<doc>_scaler.pkl
  3. Compute & save the threshold      → models/<doc>_threshold.json
  4. Pre-compute training embeddings   → models/<doc>_train_embeddings.npy
  5. Copy / reference the records JSON → models/<doc>_card_data.json

Usage (from the repo root, after all notebooks have been run):
    python export_models.py

The script expects the following files to already exist (produced by notebooks):
    aadhaar_gnn_model.pth   pan_gnn_model.pth   passport_gnn_model.pth
    aadhaar_card_data(414).json
    pan_card_data(536).json
    passport_data(200).json
    best_model.keras

It also expects Python objects in-memory:
    scaler              – the fitted StandardScaler from each notebook
    data.x              – the PyTorch feature tensor
    all_records         – the list of record dicts

Because each notebook uses a different set of variables, you should run this
script section-by-section by pasting the relevant blocks into the notebook
or running each function after the training cells.

Alternatively, run everything end-to-end by importing the re-runnable helpers
defined below.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Shared GNN class (same architecture used in all notebooks) ─────────────────
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────────────────────────────────────
# ── AADHAAR ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def build_aadhaar_features(all_records: List[Dict]) -> np.ndarray:
    """Replicate the feature engineering from Aadhaar_GNN_model.ipynb."""
    names       = [rec.get("Full Name") or "" for rec in all_records]
    name_vecs   = embed_model.encode(names)               # (N, 384)

    numeric = []
    for rec in all_records:
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        gender = gender_map.get(rec.get("Gender", ""), 3)

        day = month = year = 0
        dob = rec.get("Date/Year of Birth", "") or ""
        if "/" in dob:
            parts = dob.split("/")
            if len(parts) == 3:
                try:
                    day, month, year = map(int, parts)
                except ValueError:
                    pass
            elif len(parts) == 2:
                try:
                    month, year = map(int, parts)
                except ValueError:
                    pass
        elif re.fullmatch(r"\d{4}", dob):
            year = int(dob)

        aadhaar_digits = [0] * 12
        a = (rec.get("Aadhaar Number", "") or "").replace(" ", "")
        if a.isdigit() and len(a) == 12:
            aadhaar_digits = [int(d) for d in a]

        numeric.append([gender, day, month, year] + aadhaar_digits)

    return np.concatenate([name_vecs, np.array(numeric)], axis=1)   # (N, 400)


def export_aadhaar(
    records_json: str  = "aadhaar_card_data(414).json",
    weights_pth:  str  = "aadhaar_gnn_model.pth",
):
    log.info("── Exporting Aadhaar artifacts ──")

    # Load records
    with open(records_json) as f:
        all_records = json.load(f)
    shutil.copy(records_json, MODELS_DIR / "aadhaar_card_data.json")

    # Build features & fit scaler
    raw = build_aadhaar_features(all_records)         # (N, 400)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)

    # Save scaler
    with open(MODELS_DIR / "aadhaar_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info("  Scaler saved → models/aadhaar_scaler.pkl")

    # Load GNN & compute embeddings
    X = torch.tensor(scaled, dtype=torch.float)
    # We need edge_index from the training graph; build a dummy full-graph for export
    from torch_geometric.utils import from_networkx
    import networkx as nx
    from difflib import SequenceMatcher

    # Reconstruct the graph (same logic as notebook)
    G = nx.Graph()
    for i, rec in enumerate(all_records):
        G.add_node(i)

    def clean_aadhaar(p):
        return (p or "").replace(" ", "").strip()

    for i in range(len(all_records)):
        for j in range(i + 1, len(all_records)):
            r1, r2 = all_records[i], all_records[j]
            reasons = []
            a1 = clean_aadhaar(r1.get("Aadhaar Number"))
            a2 = clean_aadhaar(r2.get("Aadhaar Number"))
            if a1 and a1 == a2:
                reasons.append("same_aadhaar")
            if (r1.get("Full Name") and r2.get("Full Name") and
                    r1.get("Date/Year of Birth") and r2.get("Date/Year of Birth")):
                if (r1["Full Name"].lower() == r2["Full Name"].lower() and
                        r1["Date/Year of Birth"] == r2["Date/Year of Birth"]):
                    reasons.append("name_dob_match")
            if r1.get("Full Name") and r2.get("Full Name"):
                sim = SequenceMatcher(None,
                                      r1["Full Name"].lower(),
                                      r2["Full Name"].lower()).ratio()
                if sim >= 0.85 and r1.get("Date/Year of Birth") == r2.get("Date/Year of Birth"):
                    reasons.append("high_name_sim")
            if reasons:
                G.add_edge(i, j)

    data = from_networkx(G)
    data.x = X

    gnn = GNN(in_channels=400)
    gnn.load_state_dict(torch.load(weights_pth, map_location="cpu"))
    gnn.eval()

    with torch.no_grad():
        embeddings = gnn(data.x, data.edge_index)
        scores     = torch.norm(embeddings, dim=1).numpy()

    np.save(MODELS_DIR / "aadhaar_train_embeddings.npy", embeddings.numpy())
    log.info("  Embeddings saved → models/aadhaar_train_embeddings.npy")

    # Threshold
    mean, std = float(scores.mean()), float(scores.std())
    threshold = mean + 3 * std
    thresh_info = {"threshold": threshold, "mean": mean, "std": std}
    with open(MODELS_DIR / "aadhaar_threshold.json", "w") as f:
        json.dump(thresh_info, f, indent=2)
    log.info(f"  Threshold {threshold:.6f} → models/aadhaar_threshold.json")

    shutil.copy(weights_pth, MODELS_DIR / "aadhaar_gnn_model.pth")
    log.info("  Weights copied → models/aadhaar_gnn_model.pth")
    log.info("── Aadhaar export complete ──\n")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAN CARD ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def build_pan_features(all_records: List[Dict]) -> np.ndarray:
    """Replicate the feature engineering from Pan_Card_GNN_model.ipynb."""
    names        = [rec.get("Name") or ""          for rec in all_records]
    parent_names = [rec.get("Parent's Name") or "" for rec in all_records]

    name_vecs   = embed_model.encode(names)         # (N, 384)
    parent_vecs = embed_model.encode(parent_names)  # (N, 384)

    numeric = []
    for rec in all_records:
        gender = 3   # Unknown (PAN has no gender field)

        day = month = year = 0
        dob = rec.get("Date of Birth", "") or ""
        if "/" in dob:
            parts = dob.split("/")
            if len(parts) == 3:
                try:
                    day, month, year = map(int, parts)
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

        numeric.append([gender, day, month, year] + pan_digits)

    return np.concatenate(
        [name_vecs, parent_vecs, np.array(numeric)], axis=1
    )   # (N, 782)


def export_pan(
    records_json: str = "pan_card_data(536).json",
    weights_pth:  str = "pan_gnn_model.pth",
):
    log.info("── Exporting PAN artifacts ──")

    with open(records_json) as f:
        all_records = json.load(f)
    shutil.copy(records_json, MODELS_DIR / "pan_card_data.json")

    raw = build_pan_features(all_records)             # (N, 782)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)

    with open(MODELS_DIR / "pan_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info("  Scaler saved → models/pan_scaler.pkl")

    X = torch.tensor(scaled, dtype=torch.float)

    # Rebuild graph
    from torch_geometric.utils import from_networkx
    import networkx as nx
    from difflib import SequenceMatcher

    def clean_pan(p):
        return (p or "").replace(" ", "").strip()

    G = nx.Graph()
    for i in range(len(all_records)):
        G.add_node(i)

    for i in range(len(all_records)):
        for j in range(i + 1, len(all_records)):
            r1, r2 = all_records[i], all_records[j]
            reasons = []
            p1 = clean_pan(r1.get("PAN Number"))
            p2 = clean_pan(r2.get("PAN Number"))
            if p1 and p1 == p2:
                reasons.append("same_pan")
            for field in ("Name", "Parent's Name"):
                v1 = r1.get(field, "") or ""
                v2 = r2.get(field, "") or ""
                if v1 and v2:
                    if SequenceMatcher(None, v1.lower(), v2.lower()).ratio() > 0.7:
                        reasons.append(f"similar_{field.lower().replace(' ', '_')}")
            if (r1.get("Date of Birth") and
                    r1.get("Date of Birth") == r2.get("Date of Birth")):
                reasons.append("same_dob")
            if reasons:
                G.add_edge(i, j)

    in_dim = X.shape[1]
    data = from_networkx(G)
    data.x = X

    gnn = GNN(in_channels=in_dim)
    gnn.load_state_dict(torch.load(weights_pth, map_location="cpu"))
    gnn.eval()

    with torch.no_grad():
        embeddings = gnn(data.x, data.edge_index)
        scores     = torch.norm(embeddings, dim=1).numpy()

    np.save(MODELS_DIR / "pan_train_embeddings.npy", embeddings.numpy())
    log.info("  Embeddings saved → models/pan_train_embeddings.npy")

    mean, std = float(scores.mean()), float(scores.std())
    threshold = mean + 3 * std
    with open(MODELS_DIR / "pan_threshold.json", "w") as f:
        json.dump({"threshold": threshold, "mean": mean, "std": std}, f, indent=2)
    log.info(f"  Threshold {threshold:.6f} → models/pan_threshold.json")

    shutil.copy(weights_pth, MODELS_DIR / "pan_gnn_model.pth")
    log.info("── PAN export complete ──\n")


# ─────────────────────────────────────────────────────────────────────────────
# ── PASSPORT ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,  "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,"NOV": 11, "DEC": 12,
}


def build_passport_features(all_records: List[Dict]) -> np.ndarray:
    """Replicate the feature engineering from passport_GNN_model.ipynb."""
    names        = [
        f"{rec.get('given_name','')} {rec.get('surname','')}".strip()
        for rec in all_records
    ]
    birth_places = [rec.get("place_of_birth", "") or "" for rec in all_records]
    issue_places = [rec.get("place_of_issue", "") or "" for rec in all_records]

    name_vecs        = embed_model.encode(names)         # (N, 384)
    birth_place_vecs = embed_model.encode(birth_places)  # (N, 384)
    issue_place_vecs = embed_model.encode(issue_places)  # (N, 384)

    numeric = []
    for rec in all_records:
        sex_map = {"M": 1, "F": 2}
        sex = sex_map.get(rec.get("sex", ""), 0)

        nat = rec.get("nationality", "") or ""
        nat_num = sum(ord(c) for c in nat) % 1000 if nat else 0

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

        numeric.append([sex, nat_num, day, month, year])

    return np.concatenate(
        [name_vecs, birth_place_vecs, issue_place_vecs, np.array(numeric)], axis=1
    )   # (N, 1157)


def export_passport(
    records_json: str = "passport_data(200).json",
    weights_pth:  str = "passport_gnn_model.pth",
):
    log.info("── Exporting Passport artifacts ──")

    with open(records_json) as f:
        all_records = json.load(f)
    shutil.copy(records_json, MODELS_DIR / "passport_data.json")

    raw = build_passport_features(all_records)        # (N, 1157)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)

    with open(MODELS_DIR / "passport_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info("  Scaler saved → models/passport_scaler.pkl")

    X = torch.tensor(scaled, dtype=torch.float)

    # Rebuild graph
    from torch_geometric.utils import from_networkx
    import networkx as nx
    from difflib import SequenceMatcher

    full_names = [
        f"{rec.get('given_name','')} {rec.get('surname','')}".strip().lower()
        for rec in all_records
    ]

    G = nx.Graph()
    for i in range(len(all_records)):
        G.add_node(i)

    for i in range(len(all_records)):
        for j in range(i + 1, len(all_records)):
            r1, r2 = all_records[i], all_records[j]
            reasons = []
            name_sim = SequenceMatcher(None, full_names[i], full_names[j]).ratio()
            dob1 = r1.get("date_of_birth")
            dob2 = r2.get("date_of_birth")
            pob1 = r1.get("place_of_birth")
            pob2 = r2.get("place_of_birth")

            if full_names[i] == full_names[j] and dob1 and dob1 == dob2:
                reasons.append("name_dob_match")
            elif name_sim > 0.90 and dob1 and dob1 == dob2:
                reasons.append("similar_name_dob")
            elif name_sim > 0.92 and pob1 and pob1 == pob2:
                reasons.append("similar_name_pob")
            if reasons:
                G.add_edge(i, j)

    in_dim = X.shape[1]
    data = from_networkx(G)
    data.x = X

    gnn = GNN(in_channels=in_dim)
    gnn.load_state_dict(torch.load(weights_pth, map_location="cpu"))
    gnn.eval()

    with torch.no_grad():
        embeddings = gnn(data.x, data.edge_index)
        scores     = torch.norm(embeddings, dim=1).numpy()

    np.save(MODELS_DIR / "passport_train_embeddings.npy", embeddings.numpy())
    log.info("  Embeddings saved → models/passport_train_embeddings.npy")

    mean, std = float(scores.mean()), float(scores.std())
    threshold = mean + 3 * std
    with open(MODELS_DIR / "passport_threshold.json", "w") as f:
        json.dump({"threshold": threshold, "mean": mean, "std": std}, f, indent=2)
    log.info(f"  Threshold {threshold:.6f} → models/passport_threshold.json")

    shutil.copy(weights_pth, MODELS_DIR / "passport_gnn_model.pth")
    log.info("── Passport export complete ──\n")


# ─────────────────────────────────────────────────────────────────────────────
# ── Classifier model ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def export_classifier(source_path: str = "best_model.keras"):
    log.info("── Exporting Classifier ──")
    src = Path(source_path)
    dst = MODELS_DIR / "best_model.keras"
    if src.exists():
        shutil.copy(str(src), str(dst))
        log.info(f"  Copied {src} → {dst}")
    else:
        log.warning(f"  {src} not found — skipping. Place it in models/ manually.")
    log.info("── Classifier export complete ──\n")


# ─────────────────────────────────────────────────────────────────────────────
# ── Main ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export KYC model artifacts")
    parser.add_argument("--aadhaar-records",   default="aadhaar_card_data(414).json")
    parser.add_argument("--aadhaar-weights",   default="aadhaar_gnn_model.pth")
    parser.add_argument("--pan-records",       default="pan_card_data(536).json")
    parser.add_argument("--pan-weights",       default="pan_gnn_model.pth")
    parser.add_argument("--passport-records",  default="passport_data(200).json")
    parser.add_argument("--passport-weights",  default="passport_gnn_model.pth")
    parser.add_argument("--classifier",        default="best_model.keras")
    args = parser.parse_args()

    export_classifier(args.classifier)

    if Path(args.aadhaar_weights).exists():
        export_aadhaar(args.aadhaar_records, args.aadhaar_weights)
    else:
        log.warning(f"Aadhaar weights not found: {args.aadhaar_weights}")

    if Path(args.pan_weights).exists():
        export_pan(args.pan_records, args.pan_weights)
    else:
        log.warning(f"PAN weights not found: {args.pan_weights}")

    if Path(args.passport_weights).exists():
        export_passport(args.passport_records, args.passport_weights)
    else:
        log.warning(f"Passport weights not found: {args.passport_weights}")

    log.info("All exports done. Check the models/ directory.")
