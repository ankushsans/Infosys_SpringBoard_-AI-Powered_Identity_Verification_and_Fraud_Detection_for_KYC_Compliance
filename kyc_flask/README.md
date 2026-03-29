# AI KYC Compliance — Flask API

End-to-end KYC document analysis service integrating:
- An **EfficientNetV2B0** classifier (Aadhaar / PAN / Passport / Negative)
- Three **GNN anomaly-detection models** (one per document type)
- **OCR** via `openbharatocr` + `pytesseract`

---

## Architecture Overview

```
POST /api/kyc  (image)
        │
        ▼
┌─────────────────────┐
│ DocumentClassifier  │  EfficientNetV2B0 → best_model.keras
│ (TF / Keras)        │
└────────┬────────────┘
         │ doc_type ∈ {Aadhaar, PAN, Passport, Negative}
         ▼
┌─────────────────────┐
│  OCR Extraction     │  openbharatocr (Aadhaar/PAN)
│                     │  pytesseract + MRZ parser (Passport)
└────────┬────────────┘
         │ {name, dob, id_number, …}
         ▼
┌─────────────────────┐
│  GNN Inference      │  GCNConv(in→128) → GCNConv(128→64)
│  (PyTorch Geometric)│  Anomaly score = L2 norm of embedding
└────────┬────────────┘
         │ score, threshold, top-5 similar records
         ▼
       JSON Response
```

---

## File Layout

```
kyc_flask/
├── app.py                   # Flask application entry point
├── export_models.py         # ← Run ONCE after training to prepare models/
├── requirements.txt
├── models/                  # Place trained model files here
│   ├── best_model.keras          # EfficientNetV2B0 document classifier
│   │
│   ├── aadhaar_gnn_model.pth     # Aadhaar GNN weights
│   ├── aadhaar_scaler.pkl        # StandardScaler (produced by export_models.py)
│   ├── aadhaar_threshold.json    # {"threshold": …, "mean": …, "std": …}
│   ├── aadhaar_card_data.json    # Training records (for similarity search)
│   ├── aadhaar_train_embeddings.npy
│   │
│   ├── pan_gnn_model.pth
│   ├── pan_scaler.pkl
│   ├── pan_threshold.json
│   ├── pan_card_data.json
│   ├── pan_train_embeddings.npy
│   │
│   ├── passport_gnn_model.pth
│   ├── passport_scaler.pkl
│   ├── passport_threshold.json
│   ├── passport_data.json
│   └── passport_train_embeddings.npy
│
├── utils/
│   ├── __init__.py
│   ├── base_gnn.py          # Abstract base: loading, inference, similarity search
│   ├── classifier.py        # DocumentClassifier (EfficientNetV2B0 wrapper)
│   ├── aadhaar_model.py     # AadhaarGNN  (feature dim 400)
│   ├── pan_model.py         # PanGNN      (feature dim 782)
│   └── passport_model.py    # PassportGNN (feature dim 1157)
│
├── templates/
│   └── index.html           # Single-page UI
└── uploads/                 # Temp image storage (auto-created)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Also install the Tesseract binary (for passport MRZ parsing):
# Ubuntu / Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract
```

### 2. Download trained model files from GitHub

Download the following files from:
https://github.com/ankushsans/Infosys_AI_KYC_Compliance/tree/main/trained%20models

Place them in `models/`:
- `best_model.keras`
- `aadhaar_gnn_model.pth`
- `pan_gnn_model.pth`
- `passport_gnn_model.pth`

### 3. Export supplementary artifacts

The notebooks don't save the scaler, threshold, or training embeddings.
Run `export_models.py` once (in the same Python environment used for training,
after all notebook cells have executed):

```bash
python export_models.py \
  --aadhaar-records  aadhaar_card_data\(414\).json \
  --aadhaar-weights  aadhaar_gnn_model.pth \
  --pan-records      pan_card_data\(536\).json \
  --pan-weights      pan_gnn_model.pth \
  --passport-records passport_data\(200\).json \
  --passport-weights passport_gnn_model.pth \
  --classifier       best_model.keras
```

This produces all `*_scaler.pkl`, `*_threshold.json`,
`*_train_embeddings.npy`, and copies the JSON data files to `models/`.

### 4. Start the server

```bash
python app.py
# → http://localhost:5000
```

---

## API Reference

### `POST /api/kyc`

**Request:** `multipart/form-data` with field `image` (JPG / PNG / WEBP).

**Response (Aadhaar example):**

```json
{
  "document_type": "Aadhaar",
  "confidence": 0.9972,
  "all_probs": {
    "Aadhaar": 0.9972,
    "Negative": 0.0012,
    "PAN": 0.0008,
    "Passport": 0.0008
  },
  "ocr_fields": {
    "Full Name": "RAVI KUMAR",
    "Date/Year of Birth": "12/05/1985",
    "Gender": "Male",
    "Aadhaar Number": "9876 5432 1234"
  },
  "anomaly_score": 0.142381,
  "is_suspicious": false,
  "threshold": 0.387612,
  "similar_records": [
    {
      "rank": 1,
      "node_id": 47,
      "similarity": 0.9823,
      "record": {
        "Full Name": "RAVI KUMAR",
        "Date/Year of Birth": "12/05/1985",
        "Gender": "Male",
        "Aadhaar Number": "9876 5432 1234"
      }
    }
  ]
}
```

### `GET /api/health`

Returns model load status.

```json
{
  "status": "ok",
  "models": {
    "classifier":   true,
    "aadhaar_gnn":  true,
    "pan_gnn":      true,
    "passport_gnn": true
  }
}
```

---

## Feature Dimensions (from notebooks)

| Model    | Embedding                                          | Numeric                          | Total |
|----------|----------------------------------------------------|----------------------------------|-------|
| Aadhaar  | name (384)                                         | gender+day+month+year+digits(16) |   400 |
| PAN      | name (384) + parent_name (384)                     | gender+day+month+year+pan(14)    |   782 |
| Passport | name (384) + place_birth (384) + place_issue (384) | sex+nat+day+month+year (5)       |  1157 |

---

## Anomaly Threshold

Threshold = **μ + 3σ** of anomaly scores across training records (computed
by `export_models.py` and stored in `*_threshold.json`).  A record with a
score above this threshold is flagged as suspicious.  You can override the
threshold by editing the JSON file without retraining.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Model not found` at startup | Place `.pth` / `.keras` files in `models/` |
| OCR returns empty fields | `pip install openbharatocr` and ensure internet access (downloads OCR weights) |
| Passport OCR poor | Install Tesseract binary + `pip install pytesseract` |
| `scaler not found` warning | Run `export_models.py` |
| TF version mismatch on `.keras` load | Retrain or convert with matching TF version |
