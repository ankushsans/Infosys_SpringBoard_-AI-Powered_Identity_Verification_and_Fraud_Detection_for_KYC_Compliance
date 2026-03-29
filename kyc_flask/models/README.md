# models/

Place the following files here before starting the Flask server.

## Required files

### From the GitHub repo (trained models)
Download from:
https://github.com/ankushsans/Infosys_AI_KYC_Compliance/tree/main/trained%20models

| File | Size (approx.) | Description |
|------|----------------|-------------|
| `best_model.keras` | ~25 MB | EfficientNetV2B0 document classifier |
| `aadhaar_gnn_model.pth` | ~1 MB | Aadhaar GNN weights |
| `pan_gnn_model.pth` | ~1 MB | PAN card GNN weights |
| `passport_gnn_model.pth` | ~1 MB | Passport GNN weights |

### From `export_models.py` (generated once after training)
Run `python export_models.py` from the project root to produce:

| File | Description |
|------|-------------|
| `aadhaar_scaler.pkl` | sklearn StandardScaler fitted on 400-dim Aadhaar features |
| `aadhaar_threshold.json` | Anomaly threshold (μ + 3σ) |
| `aadhaar_train_embeddings.npy` | Pre-computed GNN embeddings for similarity search |
| `aadhaar_card_data.json` | Training records for similarity lookup |
| `pan_scaler.pkl` | StandardScaler fitted on 782-dim PAN features |
| `pan_threshold.json` | Anomaly threshold |
| `pan_train_embeddings.npy` | Pre-computed GNN embeddings |
| `pan_card_data.json` | Training records |
| `passport_scaler.pkl` | StandardScaler fitted on 1157-dim Passport features |
| `passport_threshold.json` | Anomaly threshold |
| `passport_train_embeddings.npy` | Pre-computed GNN embeddings |
| `passport_data.json` | Training records |

## Threshold file format

Each `*_threshold.json` looks like:
```json
{
  "threshold": 0.387612,
  "mean": 0.125430,
  "std": 0.087394
}
```

You can **manually edit the threshold** to tune sensitivity without retraining:
- Lower value → more records flagged as suspicious (higher recall)
- Higher value → fewer flags (higher precision)
