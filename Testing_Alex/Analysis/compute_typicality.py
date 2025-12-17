# -*- coding: utf-8 -*-
"""
Compute Typicality using kNN in Standardized Audio Feature Space
================================================================
For each track, typicality = mean similarity to k nearest neighbors.
Adds typicality and typicality_sq columns to sample data.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, 'results_large', 'sample_data.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'results_large', 'sample_with_typicality.csv')

# Audio features for typicality (continuous, meaningful for similarity)
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'valence',
    'instrumentalness', 'acousticness', 'speechiness', 'tempo'
]

K_NEIGHBORS = 100  # Number of neighbors for typicality calculation


def compute_typicality(df: pd.DataFrame, features: list, k: int = 100) -> np.ndarray:
    """
    Compute typicality as mean similarity to k nearest neighbors.
    
    Similarity = exp(-distance) so higher = more typical.
    Returns array of typicality scores (0-1 range approximately).
    """
    print(f"Standardizing {len(features)} features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)
    
    print(f"Building kNN index (k={k})...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn.fit(X)
    
    print("Querying nearest neighbors for all tracks...")
    distances, _ = nn.kneighbors(X)
    
    # Exclude self (first neighbor is always self with distance 0)
    distances = distances[:, 1:]  # shape: (n_samples, k)
    
    # Convert distances to similarities: exp(-d)
    # Mean similarity across k neighbors
    similarities = np.exp(-distances)
    typicality = similarities.mean(axis=1)
    
    print(f"Typicality range: [{typicality.min():.4f}, {typicality.max():.4f}]")
    print(f"Typicality mean: {typicality.mean():.4f}, std: {typicality.std():.4f}")
    
    return typicality


def main():
    print("="*60)
    print("  COMPUTING TYPICALITY (kNN-based)")
    print("="*60)
    
    if not os.path.exists(SAMPLE_PATH):
        print(f"ERROR: Sample not found at {SAMPLE_PATH}")
        return
    
    print(f"Loading sample: {SAMPLE_PATH}")
    df = pd.read_csv(SAMPLE_PATH)
    print(f"  Rows: {len(df):,}")
    
    # Check for required features
    missing = [f for f in AUDIO_FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}")
        return
    
    # Drop rows with NaN in audio features
    df_clean = df.dropna(subset=AUDIO_FEATURES + ['track.popularity']).copy()
    print(f"  After dropping NaN: {len(df_clean):,} rows")
    
    # Compute typicality
    typicality = compute_typicality(df_clean, AUDIO_FEATURES, k=K_NEIGHBORS)
    
    # Add columns
    df_clean['typicality'] = typicality
    df_clean['typicality_sq'] = typicality ** 2
    
    # Normalize typicality to 0-1 for easier interpretation
    min_t, max_t = df_clean['typicality'].min(), df_clean['typicality'].max()
    df_clean['typicality_norm'] = (df_clean['typicality'] - min_t) / (max_t - min_t)
    df_clean['typicality_norm_sq'] = df_clean['typicality_norm'] ** 2
    
    print(f"\nTypicality (normalized) stats:")
    print(df_clean['typicality_norm'].describe())
    
    # Save
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Columns: {list(df_clean.columns)}")


if __name__ == "__main__":
    main()

