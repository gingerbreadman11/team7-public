# -*- coding: utf-8 -*-
"""
Compute Typicality using cosine similarity (paper-style, no genre weighting)
============================================================================
- Uses kNN with cosine distance (k=100 by default)
- Normalizes audio features
- Outputs sample_with_typicality_cosine.csv in results_large/
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "results_large", "sample_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "results_large", "sample_with_typicality_cosine.csv")

AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "key",
    "mode",
    "time_signature",
]

K_NEIGHBORS = 50  # lowered for speed


def compute_typicality_cosine(df: pd.DataFrame, features, k=K_NEIGHBORS):
    """Typicality = mean cosine similarity to k nearest neighbors (cosine metric)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values).astype(np.float32)

    # NearestNeighbors with cosine uses cosine distance = 1 - cosine similarity
    # Use brute force + float32 for speed
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # exclude self
    distances = distances[:, 1:]
    cosine_sim = 1 - distances
    typicality = cosine_sim.mean(axis=1)

    # Normalize 0-1
    typicality_norm = (typicality - typicality.min()) / (typicality.max() - typicality.min())

    return typicality, typicality_norm


def main():
    print("=" * 60)
    print("  COMPUTING COSINE TYPICALITY (paper-style, no genre weighting)")
    print("=" * 60)

    if not os.path.exists(SAMPLE_PATH):
        print(f"ERROR: Sample not found at {SAMPLE_PATH}")
        return

    df = pd.read_csv(SAMPLE_PATH)
    print(f"Loaded sample: {len(df):,} rows")

    missing = [c for c in AUDIO_FEATURES if c not in df.columns]
    if missing:
        print(f"ERROR: missing audio features: {missing}")
        return

    df_clean = df.dropna(subset=AUDIO_FEATURES + ["track.popularity"]).copy()
    print(f"After dropping NaN: {len(df_clean):,} rows")

    # OPTIONAL: downsample for quick run; comment out to run full 500k
    if len(df_clean) > 300_000:
        df_clean = df_clean.sample(n=300_000, random_state=42)
        print(f"Downsampled to: {len(df_clean):,} rows for faster trial run")

    typicality, typicality_norm = compute_typicality_cosine(df_clean, AUDIO_FEATURES, k=K_NEIGHBORS)

    df_clean["typicality_cosine"] = typicality
    df_clean["typicality_cosine_norm"] = typicality_norm
    df_clean["typicality_cosine_norm_sq"] = typicality_norm ** 2

    print(f"Typicality cosine range: {typicality.min():.4f} - {typicality.max():.4f}")
    print(f"Typicality cosine mean: {typicality.mean():.4f} (std {typicality.std():.4f})")

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Columns now: {list(df_clean.columns)}")


if __name__ == "__main__":
    main()

