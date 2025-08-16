# FAA_MVP.py
# Minimal, working FAA MVP implementations

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.stats import median_abs_deviation, zscore

CONFIG = {
    "WINDOW_LONG": 252,
    "WINDOW_SHORT": 63,
    "DELTA": 21,
    "PCA_KEEP_VAR": 0.90,
    "KNN_K": 10,
    "FRACTURE_ZV_THR": 2.5,
    "FRACTURE_dC_THR": -0.15,
    "PHASE_METHOD": "hilbert",
    "PHASE_BINS": ("Early", "Mid", "Late"),
}

# ---------------------------
# Utilities
# ---------------------------

def _safe_z(series: pd.Series) -> pd.Series:
    """Return z-score with constant-series protection."""
    if series.dropna().nunique() <= 1:
        return pd.Series(0.0, index=series.index)
    return pd.Series(zscore(series, nan_policy="omit"), index=series.index).fillna(0.0)

def _ensure_feature_cols(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> List[str]:
    if feature_cols is None:
        exclude = {"date", "firm", "sector"}
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    return feature_cols

# ---------------------------
# 1) Standardize & Winsorize
# ---------------------------

def standardize_winsorize(
    df: pd.DataFrame,
    cols: List[str],
    by: str = "date",
    pct: Tuple[float, float] = (1.0, 99.0),
    zclip: float = 3.0,
) -> pd.DataFrame:
    """Winsorize by group percentiles, then z-score per group, then clip extreme z."""
    df = df.copy()
    if by not in df.columns:
        raise ValueError(f"'{by}' column not found in DataFrame.")

    def _process(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        for c in cols:
            s = g[c].astype(float)
            if s.notna().sum() < 3:
                # too few to estimate percentiles; skip winsorize, but still z later
                s_w = s
            else:
                lo, hi = np.nanpercentile(s, pct)
                s_w = s.clip(lower=lo, upper=hi)
            g[c] = s_w
        # z-score per group
        g[cols] = g[cols].apply(_safe_z)
        # clip
        g[cols] = g[cols].clip(-zclip, zclip)
        return g

    out = df.groupby(by, group_keys=False).apply(_process)
    return out

# ---------------------------
# 2) PCA Embedding per snapshot
# ---------------------------

def embed_pca(
    X: pd.DataFrame,
    keep_var: float = CONFIG["PCA_KEEP_VAR"],
    k_min: int = 2,
    k_max: Optional[int] = None,
):
    """Fit PCA on standardized features; choose k to reach keep_var."""
    Xn = X.fillna(0.0).to_numpy(dtype=float)
    full_pca = PCA(svd_solver="full")
    full_pca.fit(Xn)
    cums = np.cumsum(full_pca.explained_variance_ratio_)
    k = int(np.searchsorted(cums, keep_var) + 1)
    k = max(k_min, k)
    if k_max is not None:
        k = min(k, k_max)
    k = min(k, Xn.shape[1])  # cannot exceed feature count

    pca = PCA(n_components=k, svd_solver="full")
    Y = pca.fit_transform(Xn)
    Y = pd.DataFrame(Y, index=X.index, columns=[f"pc{i+1}" for i in range(k)])
    return Y, pca

# ---------------------------
# 3) kNN Graph
# ---------------------------

def build_knn_graph(
    Y: pd.DataFrame,
    k: int = CONFIG["KNN_K"],
    weight: str = "rbf",
    sigma: str | float = "median",
) -> Dict:
    """Build symmetric kNN graph; weights via RBF or distances."""
    n = len(Y)
    if n < 2:
        W = csr_matrix((n, n))
        return {"W": W, "degree": np.zeros(n), "sigma": 1.0, "index": Y.index}

    k = max(2, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Y.values)
    distances, indices = nbrs.kneighbors(Y.values)
    # drop self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    rows = []
    cols = []
    vals = []

    if weight == "distance":
        vals = distances.flatten()
    else:
        # RBF weights
        if sigma == "median":
            sig = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
        else:
            sig = float(sigma) if isinstance(sigma, (int, float)) else 1.0
        vals = np.exp(-(distances ** 2) / (2.0 * (sig ** 2) + 1e-12)).flatten()

    for i in range(n):
        for j_idx, w in zip(indices[i], vals[i * k:(i + 1) * k]):
            rows.append(i)
            cols.append(j_idx)
            rows.append(j_idx)
            cols.append(i)
            # symmetric add
            # average when duplicates appear will be handled by csr_matrix sum
            rows.append(i)
            cols.append(j_idx)
    # Build with values (we added rows twice; instead, sum duplicates cleanly)
    # Rebuild cleanly:
    rows = []
    cols = []
    vals_list = []
    for i in range(n):
        for jpos in range(k):
            j = indices[i, jpos]
            if weight == "distance":
                w = distances[i, jpos]
                w = 1.0 / (w + 1e-9)  # invert so closer = larger
            else:
                if sigma == "median":
                    sig = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
                else:
                    sig = float(sigma) if isinstance(sigma, (int, float)) else 1.0
                w = np.exp(-(distances[i, jpos] ** 2) / (2.0 * (sig ** 2) + 1e-12))
            rows += [i, j]
            cols += [j, i]
            vals_list += [w, w]

    W = csr_matrix((vals_list, (rows, cols)), shape=(n, n))
    deg = np.array(W.sum(axis=1)).ravel()
    return {"W": W, "degree": deg, "sigma": float(sig if 'sig' in locals() else 1.0), "index": Y.index}

# ---------------------------
# 4) Coherence (Laplacian roughness → (0,1])
# ---------------------------

def compute_coherence(Y: pd.DataFrame, G: Dict) -> pd.Series:
    """Coherence via Laplacian smoothness: Ly = D*Y - W*Y."""
    W = G["W"]
    deg = G["degree"]
    if W.shape[0] != len(Y):
        raise ValueError("Graph and embedding size mismatch.")

    D = csr_matrix((deg, (np.arange(len(deg)), np.arange(len(deg)))), shape=W.shape)
    # Ly = D*Y - W*Y
    Yv = Y.values
    Ly = (D @ Yv) - (W @ Yv)
    rough = np.linalg.norm(Ly, axis=1)

    med = np.median(rough)
    mad = median_abs_deviation(rough, scale="normal") if np.any(rough != rough[0]) else 0.0
    if mad == 0:
        # all equal → perfect coherence
        return pd.Series(np.ones(len(rough)), index=Y.index)
    z = (rough - med) / (mad + 1e-12)
    z = np.maximum(0.0, z)  # only penalize above-median roughness
    coh = np.exp(-z)        # map to (0,1]
    return pd.Series(coh, index=Y.index)

# ---------------------------
# 5) Deformation Velocity (two snapshots)
# ---------------------------

def deformation_velocity(Y_t0: pd.DataFrame, Y_t1: pd.DataFrame, index: Optional[pd.Index] = None) -> pd.Series:
    if index is None:
        index = Y_t0.index.intersection(Y_t1.index)
    if len(index) == 0:
        return pd.Series([], dtype=float)
    A = Y_t0.loc[index].values
    B = Y_t1.loc[index].values
    v = np.linalg.norm(B - A, axis=1)  # per-period distance
    return pd.Series(v, index=index)

# ---------------------------
# 6) Out-of-Manifold Score (reconstruction error)
# ---------------------------

def out_of_manifold_score(X: pd.DataFrame, pca_model: PCA, index: Optional[pd.Index] = None) -> pd.Series:
    Z = pca_model.transform(X.fillna(0.0).to_numpy(dtype=float))
    Xhat = pca_model.inverse_transform(Z)
    err = ((X.fillna(0.0).to_numpy(dtype=float) - Xhat) ** 2).mean(axis=1)
    idx = index if index is not None else X.index
    return pd.Series(err, index=idx)

# ---------------------------
# 7) Aggregate Signals → Composite
# ---------------------------

def aggregate_signals(
    coherence: pd.Series,
    velocity: pd.Series,
    oom_score: pd.Series,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    idx = coherence.index.intersection(velocity.index).intersection(oom_score.index)
    c = _safe_z(coherence.loc[idx])
    v = _safe_z(velocity.loc[idx])
    o = _safe_z(oom_score.loc[idx])

    if weights is None:
        weights = {"coherence": 1.0, "velocity": 1.0, "oom": 1.0}

    comp = (weights["coherence"] * (-c) +
            weights["velocity"] * v +
            weights["oom"] * o)
    comp = _safe_z(comp)
    return comp

# ---------------------------
# 8) Orchestrator (simple MVP across time)
# ---------------------------

def faa_fit_transform(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    cfg: Optional[Dict] = None
) -> Dict[str, pd.Series]:
    """
    Minimal end-to-end:
      - standardize+winsorize per date on features
      - for each date, PCA + kNN graph + coherence
      - velocity between last two dates
      - OOM score on last date
      - composite on last date
    """
    if cfg is None:
        cfg = CONFIG
    fc = _ensure_feature_cols(df, feature_cols)
    df2 = df.copy()
    # ensure required columns
    for col in ["date", "firm"]:
        if col not in df2.columns:
            raise ValueError(f"Missing required column '{col}'.")

    # preprocess
    df2 = standardize_winsorize(df2, fc, by="date")

    # work per date
    dates = sorted(df2["date"].dropna().unique())
    if len(dates) < 1:
        raise ValueError("No dates found.")
    embeddings = {}
    pca_models = {}

    for d in dates:
        snap = df2[df2["date"] == d].set_index("firm")
        X = snap[fc]
        Y, pca = embed_pca(X, keep_var=cfg["PCA_KEEP_VAR"])
        G = build_knn_graph(Y, k=cfg["KNN_K"])
        C = compute_coherence(Y, G)
        embeddings[d] = {"Y": Y, "C": C}
        pca_models[d] = {"pca": pca, "X": X}

    # velocity between last two
    v = pd.Series(dtype=float)
    if len(dates) >= 2:
        t0, t1 = dates[-2], dates[-1]
        v = deformation_velocity(embeddings[t0]["Y"], embeddings[t1]["Y"])

    # OOM on last date
    last = dates[-1]
    pca = pca_models[last]["pca"]
    Xlast = pca_models[last]["X"]
    O = out_of_manifold_score(Xlast, pca)

    # Composite on last date (align indexes)
    C_last = embeddings[last]["C"]
    idx = C_last.index.intersection(v.index if len(v) else C_last.index).intersection(O.index)
    # if no velocity (only one date), create zeros to allow composite
    if len(v) == 0:
        v = pd.Series(0.0, index=idx)
    comp = aggregate_signals(C_last.loc[idx], v.loc[idx], O.loc[idx])

    return {
        "coherence": C_last,
        "velocity": v,
        "oom": O,
        "composite": comp,
        "last_date": pd.Series([last], index=["last_date"]),
    }

# ---------------------------
# 9) API entry-point for Flask
# ---------------------------

def run_analysis(payload: Dict) -> Dict:
    """
    Expects JSON:
      {
        "table": [ {"date":"YYYY-MM-DD","firm":"AAPL","sector":"Tech","feat1":..., ...}, ... ],
        "feature_cols": ["feat1","feat2", ...]   # optional
      }
    Returns per-firm series on last date.
    """
    if "table" not in payload or not isinstance(payload["table"], list):
        raise ValueError("Payload must include 'table' as a list of row dicts.")
    df = pd.DataFrame(payload["table"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    feature_cols = payload.get("feature_cols")
    out = faa_fit_transform(df, feature_cols)
    # Convert series to plain dicts
    resp = {
        "last_date": out["last_date"]["last_date"],
        "coherence": out["coherence"].to_dict(),
        "velocity": out["velocity"].to_dict(),
        "oom": out["oom"].to_dict(),
        "composite": out["composite"].to_dict(),
    }
    return resp

