# -*- coding: utf-8 -*-
"""
Typicality k Comparison (k=25 vs k=100)
=======================================
Uses feature-deduped sample (max pop).
Plots both curves on same figure.

Usage:
    python3 run_typicality_k_comparison.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'results_feature_dedup', 'sample_feature_dedup.csv')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
OUT_DIR = os.path.join(BASE_DIR, 'results_feature_dedup')
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
    'loudness', 'speechiness', 'tempo', 'valence', 'key', 'mode', 'time_signature'
]

# Colors
COLOR_K5 = "#2E86AB"    # teal/blue
COLOR_K25 = "#67129A"   # purple
COLOR_K100 = "#D20D52"  # red


def compute_typicality_cosine(X, k):
    """Compute cosine typicality with given k."""
    print(f"  Computing k={k}...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # exclude self
    cos_sim = 1 - distances
    typ = cos_sim.mean(axis=1)
    typ_norm = (typ - typ.min()) / (typ.max() - typ.min())
    return typ_norm


def compute_binned_stats(typicality, popularity, n_bins=20):
    """Bin typicality and compute mean popularity per bin."""
    valid = ~np.isnan(typicality) & ~np.isnan(popularity)
    typ = typicality[valid]
    pop = popularity[valid]
    
    # Filter to typicality >= 0.5 for cleaner plot
    mask = typ >= 0.5
    typ = typ[mask]
    pop = pop[mask]
    
    bin_edges = np.percentile(typ, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 0.001
    bin_idx = np.digitize(typ, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    stats = []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            stats.append({
                'typ_mean': typ[m].mean(),
                'pop_mean': pop[m].mean(),
                'pop_std': pop[m].std(),
                'count': m.sum()
            })
    return pd.DataFrame(stats)


def main():
    print("="*60)
    print("  TYPICALITY k COMPARISON (k=25 vs k=100)")
    print("="*60)
    
    # Load feature-deduped data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run feature dedup first.")
        return
    
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + ['track.popularity']).copy()
    print(f"Loaded: {len(df):,} rows")
    
    # Standardize features
    print("Standardizing features...")
    X = StandardScaler().fit_transform(df[FEATURES].values).astype(np.float32)
    popularity = df['track.popularity'].values
    
    # Compute typicality for k=5, k=25, and k=100
    typ_k5 = compute_typicality_cosine(X, k=5)
    typ_k25 = compute_typicality_cosine(X, k=25)
    typ_k100 = compute_typicality_cosine(X, k=100)
    
    # Save to dataframe
    df['typicality_k5'] = typ_k5
    df['typicality_k25'] = typ_k25
    df['typicality_k100'] = typ_k100
    df.to_csv(os.path.join(OUT_DIR, 'sample_with_typicality_k5_k25_k100.csv'), index=False)
    print(f"Saved: {OUT_DIR}/sample_with_typicality_k5_k25_k100.csv")
    
    # Compute binned stats
    stats_k5 = compute_binned_stats(typ_k5, popularity, n_bins=20)
    stats_k25 = compute_binned_stats(typ_k25, popularity, n_bins=20)
    stats_k100 = compute_binned_stats(typ_k100, popularity, n_bins=20)
    
    print("\nBinned stats (k=5):")
    print(stats_k5[['typ_mean', 'pop_mean']].round(3))
    print("\nBinned stats (k=25):")
    print(stats_k25[['typ_mean', 'pop_mean']].round(3))
    print("\nBinned stats (k=100):")
    print(stats_k100[['typ_mean', 'pop_mean']].round(3))
    
    # Plot all three curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # k=5
    se_k5 = stats_k5['pop_std'] / np.sqrt(stats_k5['count'])
    ax.fill_between(stats_k5['typ_mean'], 
                    stats_k5['pop_mean'] - 1.96*se_k5,
                    stats_k5['pop_mean'] + 1.96*se_k5,
                    alpha=0.15, color=COLOR_K5)
    ax.plot(stats_k5['typ_mean'], stats_k5['pop_mean'], 
            '^-', color=COLOR_K5, linewidth=2.5, markersize=6, label='k = 5')
    
    # k=25
    se_k25 = stats_k25['pop_std'] / np.sqrt(stats_k25['count'])
    ax.fill_between(stats_k25['typ_mean'], 
                    stats_k25['pop_mean'] - 1.96*se_k25,
                    stats_k25['pop_mean'] + 1.96*se_k25,
                    alpha=0.15, color=COLOR_K25)
    ax.plot(stats_k25['typ_mean'], stats_k25['pop_mean'], 
            'o-', color=COLOR_K25, linewidth=2.5, markersize=6, label='k = 25')
    
    # k=100
    se_k100 = stats_k100['pop_std'] / np.sqrt(stats_k100['count'])
    ax.fill_between(stats_k100['typ_mean'], 
                    stats_k100['pop_mean'] - 1.96*se_k100,
                    stats_k100['pop_mean'] + 1.96*se_k100,
                    alpha=0.15, color=COLOR_K100)
    ax.plot(stats_k100['typ_mean'], stats_k100['pop_mean'], 
            's-', color=COLOR_K100, linewidth=2.5, markersize=6, label='k = 100')
    
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(35, 40)
    ax.set_xlabel('Typicality (cosine, 0 = distinctive, 1 = typical)', fontsize=12)
    ax.set_ylabel('Mean Popularity (0–100)', fontsize=12)
    ax.set_title('Typicality vs Popularity', fontsize=14, fontweight='bold')
    ax.legend(title='Neighbors', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    fig_path = os.path.join(FIG_DIR, 'typicality_k5_k25_k100.png')
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_path.replace('.png', '.pdf'))
    plt.close()
    print(f"\nSaved: {fig_path}")
    
    print("\n" + "="*60)
    print("  DONE")
    print("="*60)


if __name__ == "__main__":
    main()

