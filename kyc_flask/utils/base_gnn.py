"""
utils/base_gnn.py
──────────────────
Abstract base class shared by all three document GNN models.

Each subclass must implement:
    • run_ocr(image_path)          → dict of extracted fields
    • _build_features(rec)         → np.ndarray  (1, feature_dim)
    • MODEL_FILENAME               → str
    • SCALER_FILENAME              → str
    • THRESHOLD_FILENAME           → str
    • RECORDS_FILENAME             → str
    • FEATURE_DIM                  → int
"""
from __future__ import annotations

from pathlib import Path
import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Lazy imports (torch & torch_geometric are heavy) ───────────────────────────
_torch = None
_F     = None


def _import_torch():
    global _torch, _F
    if _torch is None:
        import torch
        import torch.nn.functional as F
        _torch = torch
        _F     = F
    return _torch, _F


class GCNConv(_import_torch()[0].nn.Module if False else object):
    pass  # defined properly below after torch is available


def _get_gcnconv_class():
    """
    Pure-PyTorch GCNConv — no torch_geometric needed.
    State-dict keys match torch_geometric exactly:
        conv.lin.weight  (out_channels, in_channels)
        conv.bias        (out_channels,)
    So all saved .pth files load without any modification.
    """
    torch, F = _import_torch()

    class GCNConv(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
            super().__init__()
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
            if bias:
                self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            else:
                self.register_parameter("bias", None)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            N   = x.size(0)
            out = self.lin(x)                        # (N, out_channels)

            # Single isolated node — no edges to aggregate
            if edge_index.size(1) == 0:
                return out + self.bias if self.bias is not None else out

            # Add self-loops
            self_loops = torch.arange(N, dtype=torch.long, device=x.device)
            self_loops = self_loops.unsqueeze(0).expand(2, -1)   # (2, N)
            ei = torch.cat([edge_index, self_loops], dim=1)

            row, col = ei[0], ei[1]

            # Degree vector
            deg = torch.zeros(N, dtype=x.dtype, device=x.device)
            deg.scatter_add_(0, row,
                             torch.ones(ei.size(1), dtype=x.dtype, device=x.device))

            # D^{-1/2}, guard against isolated nodes
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)

            # Per-edge normalisation coefficient
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]         # (E,)

            # Normalised aggregation
            src = norm.unsqueeze(1) * out[col]                   # (E, out_channels)
            agg = torch.zeros_like(out)
            agg.scatter_add_(0, row.unsqueeze(1).expand_as(src), src)

            return agg + self.bias if self.bias is not None else agg

    return GCNConv


def _build_gnn_class():
    torch, F = _import_torch()
    GCNConv  = _get_gcnconv_class()

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

    return GNN

# ── Base class ─────────────────────────────────────────────────────────────────
class BaseGNN(ABC):
    """
    Shared loading / inference logic.

    Files expected in models_dir:
        <MODEL_FILENAME>      – PyTorch state-dict (.pth)
        <SCALER_FILENAME>     – sklearn StandardScaler pickle (.pkl)
        <THRESHOLD_FILENAME>  – JSON with {"threshold": float,
                                           "mean": float, "std": float}
        <RECORDS_FILENAME>    – JSON list of training records (for similarity search)
        <EMBEDDINGS_FILENAME> – (optional) pre-computed training embeddings (.npy)
    """

    MODEL_FILENAME      = NotImplemented
    SCALER_FILENAME     = NotImplemented
    THRESHOLD_FILENAME  = NotImplemented
    RECORDS_FILENAME    = NotImplemented
    EMBEDDINGS_FILENAME = NotImplemented   # Optional — computed on first use if absent
    FEATURE_DIM         = NotImplemented   # Must be set by subclass

    def __init__(self, models_dir: Path):
        self._models_dir = models_dir
        self._model      = None
        self._scaler     = None
        self._threshold  = None
        self._records: List[Dict] = []
        self._train_embeddings: Any = None   # torch tensor

        self._load()

    # ── Loading ─────────────────────────────────────────────────────────────────
    def _load(self):
        torch, F  = _import_torch()
        GNN = _build_gnn_class()

        # 1. Model weights
        model_path = self._models_dir / self.MODEL_FILENAME
        if model_path.exists():
            try:
                gnn = GNN(in_channels=self.FEATURE_DIM)
                gnn.load_state_dict(
                    torch.load(str(model_path), map_location="cpu")
                )
                gnn.eval()
                self._model = gnn
                log.info(f"{self.__class__.__name__}: loaded weights from {model_path}")
            except Exception as exc:
                log.error(f"{self.__class__.__name__}: failed to load model: {exc}")
        else:
            log.warning(f"{self.__class__.__name__}: model file not found at {model_path}")

        # 2. Scaler
        scaler_path = self._models_dir / self.SCALER_FILENAME
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                log.info(f"{self.__class__.__name__}: loaded scaler from {scaler_path}")
            except Exception as exc:
                log.error(f"{self.__class__.__name__}: failed to load scaler: {exc}")
        else:
            log.warning(f"{self.__class__.__name__}: scaler not found at {scaler_path}. "
                        "Predictions will use unscaled features — retrain/export the scaler.")

        # 3. Threshold
        thresh_path = self._models_dir / self.THRESHOLD_FILENAME
        if thresh_path.exists():
            try:
                with open(thresh_path) as f:
                    info = json.load(f)
                self._threshold = info.get("threshold", 0.5)
                log.info(f"{self.__class__.__name__}: threshold = {self._threshold:.6f}")
            except Exception as exc:
                log.error(f"{self.__class__.__name__}: failed to load threshold: {exc}")
        else:
            log.warning(f"{self.__class__.__name__}: threshold file not found at {thresh_path}. "
                        "Using default threshold 0.5.")
            self._threshold = 0.5

        # 4. Training records (for similarity search)
        records_path = self._models_dir / self.RECORDS_FILENAME
        if records_path.exists():
            try:
                with open(records_path) as f:
                    self._records = json.load(f)
                log.info(f"{self.__class__.__name__}: {len(self._records)} training records loaded")
            except Exception as exc:
                log.error(f"{self.__class__.__name__}: failed to load records: {exc}")

        # 5. Pre-computed training embeddings (optional)
        emb_path = self._models_dir / self.EMBEDDINGS_FILENAME
        if emb_path.exists():
            try:
                arr = np.load(str(emb_path))
                self._train_embeddings = _torch.tensor(arr, dtype=_torch.float)
                log.info(f"{self.__class__.__name__}: loaded pre-computed embeddings {arr.shape}")
            except Exception as exc:
                log.error(f"{self.__class__.__name__}: failed to load embeddings: {exc}")

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── OCR (implemented by each subclass) ─────────────────────────────────────
    @abstractmethod
    def run_ocr(self, image_path: str) -> Dict[str, str]:
        """Run OCR on the given image and return a dict of extracted fields."""

    # ── Feature engineering (implemented by each subclass) ─────────────────────
    @abstractmethod
    def _build_features(self, rec: Dict) -> np.ndarray:
        """Convert a record dict → raw (1, FEATURE_DIM) numpy array (before scaling)."""

    # ── Inference ───────────────────────────────────────────────────────────────
    def infer(self, rec: Dict, top_k: int = 5) -> Tuple[float, bool, float, List[Dict]]:
        if not self.is_loaded():
            raise RuntimeError(f"{self.__class__.__name__} model not loaded.")

        torch, F = _import_torch()

        raw = self._build_features(rec)          # (1, FEATURE_DIM)

        if self._scaler is not None:
            raw = self._scaler.transform(raw)

        sample_x = torch.tensor(raw, dtype=torch.float).cpu()
        empty_ei = torch.empty((2, 0), dtype=torch.long)

        self._model.eval()
        with torch.no_grad():
            embedding = self._model(sample_x, empty_ei)
            score = float(torch.norm(embedding, dim=1).item())

        threshold     = self._threshold or 0.5
        is_suspicious = score > threshold
        similar       = self._find_similar(sample_x, embedding, top_k)

        return score, is_suspicious, threshold, similar   


    
    def _find_similar(
        self,
        sample_x: Any,
        sample_emb: Any,
        top_k: int,
    ) -> List[Dict]:
        """
        Cosine similarity between the query's GNN embedding and all
        training embeddings.  Falls back to feature-space cosine if no
        training embeddings are available.
        """
        torch, F = _import_torch()

        if not self._records:
            return []

        # Use pre-computed training embeddings when available
        if self._train_embeddings is not None:
            ref = self._train_embeddings                    # (N, 64)
            q   = sample_emb                                # (1, 64)
        else:
            # Fallback: raw feature cosine (less meaningful, but works)
            ref = None
            q   = sample_x

        if ref is None:
            return []

        q_norm   = F.normalize(q,   dim=1)
        ref_norm = F.normalize(ref, dim=1)
        sims     = torch.mm(q_norm, ref_norm.T).squeeze(0)  # (N,)

        k = min(top_k, len(self._records))
        values, indices = torch.topk(sims, k)

        results = []
        for rank, (node_id, sim_score) in enumerate(
            zip(indices.tolist(), values.tolist()), start=1
        ):
            results.append({
                "rank":       rank,
                "node_id":    node_id,
                "similarity": round(float(sim_score), 6),
                "record":     self._records[node_id],
            })

        return results
