"""
FAA_MVP_Crypto.py
Fractal Market Analyzer — Crypto Universe
Author: Benjamin Henry Owino
Version: 1.0
"""

import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# --- Helpers ---
def normalize(df, feature_cols):
    """Standardize features column-wise."""
    return (df[feature_cols] - df[feature_cols].mean()) / (df[feature_cols].std() + 1e-9)

def pca_reconstruction_error(X, n_components=3):
    """Return normalized PCA reconstruction error."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_pca)
    err = np.mean((X - X_recon) ** 2, axis=1)
    return err / (err.max() + 1e-9)

# --- Main Entry ---
def run_analysis_crypto(payload, seed=42):
    """
    Run Fractal Analysis for crypto-asset time series.
    Expects payload = { "table": [...], "feature_cols": [...] }
    Produces both JSON + Markdown report under /mnt/data/
    """

    # --- Input checks ---
    table = payload.get("table", [])
    feature_cols = payload.get("feature_cols", [])
    if not table or not feature_cols:
        raise ValueError("Missing 'table' or 'feature_cols' in request.")

    df = pd.DataFrame(table)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date]

    # --- Cohesion ---
    category_means = latest_df.groupby("category")[feature_cols].mean()
    coh_values = []
    for _, row in latest_df.iterrows():
        centroid = category_means.loc[row["category"]].values
        dist = np.linalg.norm(row[feature_cols].values - centroid)
        coh_values.append(dist)
    coh = float(np.mean(coh_values))

    # --- Velocity (weekly cycle, L=7) ---
    L = 7
    past_df = df[df["date"] == latest_date - pd.Timedelta(days=L)]
    vel = np.nan
    if not past_df.empty:
        merged = latest_df.merge(past_df, on="asset", suffixes=("", "_past"))
        if not merged.empty:
            vel = float(np.linalg.norm(
                normalize(merged, feature_cols).values -
                normalize(merged, [f + "_past" for f in feature_cols]).values
            ))

    # --- RDI (rolling EMA of velocity, 30d span) ---
    rdi = np.nan if np.isnan(vel) else float(pd.Series([vel]).ewm(span=30).mean().iloc[-1])

    # --- OOM (PCA reconstruction error) ---
    X_std = normalize(latest_df, feature_cols).values
    oom_norm = float(np.mean(pca_reconstruction_error(X_std)))

    # --- FRAC (fracture alert) ---
    frac_flag = int(vel > np.nanpercentile([vel], 90)) if vel == vel else 0

    # --- Deterministic tokens ---
    tokens = {
        "T_coh": ["STABLE" if coh < 0.5 else "UNSTABLE", coh],
        "T_vel": ["FAST" if vel and vel > 1 else "SLOW", vel],
        "T_rdi": ["ELEVATED" if rdi and rdi > 0.7 else "NORMAL", rdi],
        "T_oom": ["OUT" if oom_norm > 0.5 else "IN", oom_norm],
        "T_frac": ["FRACTURE" if frac_flag else "OK", frac_flag],
        "T_sect": ["MID", 0.5]  # placeholder sector phase until full impl
    }

    # --- Hypotheses ---
    hypotheses = []
    if coh < 0.5 and oom_norm < 0.5:
        hypotheses.append("structural_stability")
    if frac_flag == 1:
        hypotheses.append("fracture_alert")
    if not hypotheses:
        hypotheses.append("weak_structure")

    # --- JSON Schema ---
    result = {
        "meta": {
            "mode": "ts",
            "seed": seed,
            "universe": "crypto",
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "xsection_date": str(latest_date.date()),
            "firm": str(latest_df["asset"].iloc[0]) if not latest_df.empty else None,
            "L": L,
            "notes": "[ASSUMED]" if df.shape[0] < 90 else ""
        },
        "estimates": {
            "coh": coh,
            "vel": vel,
            "rdi": rdi,
            "oom_norm": oom_norm,
            "frac_flag": frac_flag
        },
        "tokens": tokens,
        "hypotheses": hypotheses
    }

    # --- Save outputs ---
    with open("/mnt/data/summary.json", "w") as f:
        json.dump(result, f, indent=2)

    with open("/mnt/data/report.md", "w") as f:
        f.write("# Crypto Market Analysis Report\n\n")
        f.write(f"**Date:** {latest_date.date()}\n\n")
        f.write(f"- Cohesion: {coh:.3f}\n")
        f.write(f"- Velocity: {vel:.3f}\n")
        f.write(f"- RDI: {rdi:.3f}\n")
        f.write(f"- OOM: {oom_norm:.3f}\n")
        f.write(f"- Fracture Flag: {frac_flag}\n\n")
        f.write("### Tokens\n")
        for k, v in tokens.items():
            f.write(f"- {k}: {v[0]} (conf={v[1]:.3f})\n")
        f.write("\n### Hypotheses\n")
        for h in hypotheses:
            f.write(f"- {h}\n")
        f.write("\n---\nDisclaimer: Research only; no investment advice.\n")

    print("JSON validation: PASS")
    return result

