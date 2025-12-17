# -*- coding: utf-8 -*-
"""
Deduplicate tracks and compute cosine-based typicality (fast, isolated outputs).

Requirements:
- Does NOT modify existing data; writes everything to results_dedup/.
- Dedup rules:
    1) Collapse exact ISRC matches (keep max popularity, first for others).
    2) Collapse exact duplicate feature vectors (same 12 audio features);
       keep max popularity, count how many fused.
- Compute cosine typicality on the deduped set with k=25 (faster) and brute search.
- Limit to 150k rows for the typicality step to keep runtime reasonable.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_SAMPLE = os.path.join(BASE_DIR, "results_large", "sample_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results_dedup")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

K_NEIGHBORS = 25
MAX_ROWS_FOR_TYP = 150_000  # cap for the typicality computation to avoid long runtimes


def dedup_by_isrc(df: pd.DataFrame) -> (pd.DataFrame, int):
    """Collapse rows sharing the same ISRC. Keep max popularity and first for other cols."""
    has_isrc = df[df["isrc"].notna()].copy()
    no_isrc = df[df["isrc"].isna()].copy()

    removed = 0
    if not has_isrc.empty:
        agg = {col: "first" for col in has_isrc.columns}
        agg["track.popularity"] = "max"
        agg["isrc"] = "first"
        collapsed = has_isrc.groupby("isrc", as_index=False).agg(agg)
        removed = len(has_isrc) - len(collapsed)
    else:
        collapsed = has_isrc

    deduped = pd.concat([collapsed, no_isrc], ignore_index=True)
    return deduped, removed


def dedup_by_feature_vector(df: pd.DataFrame) -> (pd.DataFrame, int):
    """Collapse exact duplicates in feature space (all audio features identical)."""
    before = len(df)
    agg = {col: "first" for col in df.columns}
    agg["track.popularity"] = "max"

    deduped = df.groupby(AUDIO_FEATURES, as_index=False).agg(agg)
    removed = before - len(deduped)
    return deduped, removed


def compute_typicality_cosine(df: pd.DataFrame, k: int = K_NEIGHBORS):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[AUDIO_FEATURES].values).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df)), metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # drop self
    cos_sim = 1 - distances
    typicality = cos_sim.mean(axis=1)
    typ_norm = (typicality - typicality.min()) / (typicality.max() - typicality.min())
    return typicality, typ_norm


def main():
    print("=" * 60)
    print("  DEDUP + FAST COSINE TYPICALITY (safe outputs)")
    print("=" * 60)

    if not os.path.exists(INPUT_SAMPLE):
        print(f"ERROR: sample not found at {INPUT_SAMPLE}")
        return

    df = pd.read_csv(INPUT_SAMPLE)
    print(f"Loaded sample: {len(df):,} rows")

    # Basic cleaning
    df = df.dropna(subset=AUDIO_FEATURES + ["track.popularity"]).copy()
    print(f"After dropping NaN in features/popularity: {len(df):,}")

    # Dedup by ISRC
    df_iso, removed_isrc = dedup_by_isrc(df)
    print(f"After ISRC dedup: {len(df_iso):,} rows (removed {removed_isrc:,})")

    # Dedup by exact feature vector
    df_feat, removed_feat = dedup_by_feature_vector(df_iso)
    print(f"After feature-vector dedup: {len(df_feat):,} rows (removed {removed_feat:,})")

    # Save deduped sample
    dedup_path = os.path.join(OUTPUT_DIR, "sample_dedup.csv")
    df_feat.to_csv(dedup_path, index=False)
    print(f"Saved deduped sample to: {dedup_path}")

    # Prepare for typicality
    df_typ = df_feat.copy()
    if len(df_typ) > MAX_ROWS_FOR_TYP:
        df_typ = df_typ.sample(n=MAX_ROWS_FOR_TYP, random_state=42)
        print(f"Downsampled to {len(df_typ):,} rows for typicality speed.")

    print("Computing cosine typicality (k=25, brute)...")
    typicality, typ_norm = compute_typicality_cosine(df_typ, k=K_NEIGHBORS)
    df_typ["typicality_cosine"] = typicality
    df_typ["typicality_cosine_norm"] = typ_norm
    df_typ["typicality_cosine_norm_sq"] = typ_norm ** 2

    typ_path = os.path.join(OUTPUT_DIR, "sample_with_typicality_cosine_dedup.csv")
    df_typ.to_csv(typ_path, index=False)
    print(f"Saved typicality output to: {typ_path}")

    # Log stats
    stats = {
        "original_rows": len(df),
        "after_isrc_dedup": len(df_iso),
        "removed_isrc": removed_isrc,
        "after_feature_dedup": len(df_feat),
        "removed_feature_dups": removed_feat,
        "typicality_rows": len(df_typ),
        "k_neighbors": K_NEIGHBORS,
        "max_rows_for_typicality": MAX_ROWS_FOR_TYP,
    }
    stats_path = os.path.join(OUTPUT_DIR, "dedup_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()

