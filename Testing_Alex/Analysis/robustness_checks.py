# -*- coding: utf-8 -*-
"""
Robustness Checks for Typicality Analysis
==========================================
1. Vary k (25, 50, 100, 200) and compare curves
2. Exclude same-artist neighbors
3. Exclude hits from reference set
4. Generate poster-ready comparison figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, 'results_large', 'sample_data.csv')
TRACK_FEATURES_DIR = os.path.join(BASE_DIR, '..', 'Data', 'album and track files', 'tracks', 'csvs for paper sample')
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_large')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
pbStart = "#67129A"
pbMid = "#9A2578"
pbEnd = "#D20D52"
COLORS = [pbStart, pbMid, pbEnd, "#2E86AB"]

AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'valence',
    'instrumentalness', 'acousticness', 'speechiness', 'tempo'
]


def load_sample_with_artist():
    """Load sample and merge artist.id from track_features."""
    print("Loading sample data...")
    df = pd.read_csv(SAMPLE_PATH)
    print(f"  Sample rows: {len(df):,}")
    
    # Load track features to get artist.id
    print("Loading track features for artist.id...")
    csv_files = sorted(glob.glob(os.path.join(TRACK_FEATURES_DIR, 'track_features_*.csv')))
    
    artist_map = {}
    for f in csv_files:
        tf = pd.read_csv(f, usecols=['track.id', 'artist.id'])
        for _, row in tf.iterrows():
            if pd.notna(row['track.id']) and pd.notna(row['artist.id']):
                artist_map[row['track.id']] = row['artist.id']
    
    df['artist.id'] = df['track.id'].map(artist_map)
    matched = df['artist.id'].notna().sum()
    print(f"  Matched artist.id: {matched:,} / {len(df):,} ({matched/len(df)*100:.1f}%)")
    
    return df


def compute_typicality_varied_k(df, k_values=[25, 50, 100, 200]):
    """Compute typicality for multiple k values."""
    df = df.dropna(subset=AUDIO_FEATURES + ['track.popularity']).copy()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[AUDIO_FEATURES].values)
    
    results = {}
    for k in k_values:
        print(f"  Computing typicality with k={k}...")
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        distances = distances[:, 1:]  # exclude self
        similarities = np.exp(-distances)
        typicality = similarities.mean(axis=1)
        
        # Normalize to 0-1
        typicality_norm = (typicality - typicality.min()) / (typicality.max() - typicality.min())
        results[k] = typicality_norm
    
    return df, results


def compute_typicality_no_same_artist(df):
    """Compute typicality excluding same-artist neighbors."""
    df = df.dropna(subset=AUDIO_FEATURES + ['track.popularity', 'artist.id']).copy()
    print(f"  Rows with artist.id: {len(df):,}")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[AUDIO_FEATURES].values)
    artist_ids = df['artist.id'].values
    
    k = 100
    # Use more neighbors initially, then filter
    nn = NearestNeighbors(n_neighbors=min(500, len(df)-1), metric='euclidean', n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    typicality = []
    for i in range(len(df)):
        focal_artist = artist_ids[i]
        # Get neighbors excluding same artist
        neighbor_mask = artist_ids[indices[i, 1:]] != focal_artist
        valid_distances = distances[i, 1:][neighbor_mask][:k]
        
        if len(valid_distances) < 10:
            typicality.append(np.nan)
        else:
            similarities = np.exp(-valid_distances)
            typicality.append(similarities.mean())
    
    typicality = np.array(typicality)
    valid_mask = ~np.isnan(typicality)
    typicality_norm = np.full_like(typicality, np.nan)
    typicality_norm[valid_mask] = (typicality[valid_mask] - np.nanmin(typicality)) / (np.nanmax(typicality) - np.nanmin(typicality))
    
    return df, typicality_norm


def compute_typicality_no_hits_reference(df, hit_threshold=80):
    """Compute typicality using only non-hits as reference set."""
    df = df.dropna(subset=AUDIO_FEATURES + ['track.popularity']).copy()
    
    # Split into hits and non-hits
    is_hit = df['track.popularity'] >= hit_threshold
    non_hits = df[~is_hit].copy()
    print(f"  Reference set (non-hits <{hit_threshold}): {len(non_hits):,}")
    print(f"  Query set (all tracks): {len(df):,}")
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[AUDIO_FEATURES].values)
    X_ref = scaler.transform(non_hits[AUDIO_FEATURES].values)
    
    k = min(100, len(non_hits) - 1)
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(X_ref)
    distances, _ = nn.kneighbors(X_all)
    
    similarities = np.exp(-distances)
    typicality = similarities.mean(axis=1)
    typicality_norm = (typicality - typicality.min()) / (typicality.max() - typicality.min())
    
    return df, typicality_norm


def compute_binned_stats(typicality, popularity, n_bins=20):
    """Compute binned statistics for plotting."""
    valid = ~np.isnan(typicality) & ~np.isnan(popularity)
    typ = typicality[valid]
    pop = popularity[valid]
    
    # Create bins
    bin_edges = np.percentile(typ, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 0.001  # ensure last value included
    bin_idx = np.digitize(typ, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    stats = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            stats.append({
                'bin': b,
                'typicality_mean': typ[mask].mean(),
                'pop_mean': pop[mask].mean(),
                'pop_std': pop[mask].std(),
                'count': mask.sum(),
                'hit_rate': (pop[mask] >= 70).mean()
            })
    
    return pd.DataFrame(stats)


def plot_k_comparison(df, k_results):
    """Plot typicality vs popularity for different k values."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    popularity = df['track.popularity'].values
    
    for i, (k, typicality) in enumerate(k_results.items()):
        stats = compute_binned_stats(typicality, popularity, n_bins=15)
        ax.plot(stats['typicality_mean'], stats['pop_mean'], 
                'o-', color=COLORS[i % len(COLORS)], linewidth=2, markersize=5,
                label=f'k = {k}')
    
    ax.set_xlabel('Typicality (0 = Distinctive, 1 = Typical)', fontsize=11)
    ax.set_ylabel('Mean Popularity', fontsize=11)
    ax.set_title('Robustness Check: Typicality vs Popularity\nacross different k values', fontsize=12, fontweight='bold')
    ax.legend(title='Neighbors (k)')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(25, 65)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'robustness_k_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved: {out_path}")


def plot_robustness_comparison(df, typ_standard, typ_no_artist, typ_no_hits):
    """Compare standard vs no-same-artist vs no-hits-reference."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    popularity = df['track.popularity'].values
    
    configs = [
        (typ_standard, 'Standard (k=100)', pbStart),
        (typ_no_artist, 'Exclude same-artist', pbMid),
        (typ_no_hits, 'Non-hits reference', pbEnd),
    ]
    
    for typicality, label, color in configs:
        if typicality is not None:
            stats = compute_binned_stats(typicality, popularity, n_bins=15)
            ax.plot(stats['typicality_mean'], stats['pop_mean'], 
                    'o-', color=color, linewidth=2, markersize=5, label=label)
    
    ax.set_xlabel('Typicality (0 = Distinctive, 1 = Typical)', fontsize=11)
    ax.set_ylabel('Mean Popularity', fontsize=11)
    ax.set_title('Robustness Check: Different Reference Sets', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(25, 65)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'robustness_reference_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved: {out_path}")


def plot_hit_density(df, typicality, hit_threshold=70):
    """Density plot of typicality for hits vs non-hits."""
    df = df.copy()
    df['typicality'] = typicality
    df = df.dropna(subset=['typicality', 'track.popularity'])
    
    df['is_hit'] = df['track.popularity'] >= hit_threshold
    hits = df[df['is_hit']]['typicality']
    non_hits = df[~df['is_hit']]['typicality']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.kdeplot(non_hits, ax=ax, fill=True, alpha=0.5, color=COLORS[0], label=f'Non-hits (<{hit_threshold})')
    sns.kdeplot(hits, ax=ax, fill=True, alpha=0.5, color=pbEnd, label=f'Hits (≥{hit_threshold})')
    
    # Add vertical lines for means
    ax.axvline(non_hits.mean(), color=COLORS[0], linestyle='--', linewidth=2)
    ax.axvline(hits.mean(), color=pbEnd, linestyle='--', linewidth=2)
    
    ax.set_xlabel('Typicality (0 = Distinctive, 1 = Typical)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of Typicality:\nHits vs Non-Hits', fontsize=12, fontweight='bold')
    ax.legend()
    
    # Add stats text
    stats_text = f'Non-hits mean: {non_hits.mean():.3f}\nHits mean: {hits.mean():.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'typicality_density_hits_vs_others.png')
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved: {out_path}")
    
    print(f"\n  Hit density stats:")
    print(f"    Non-hits ({len(non_hits):,}): mean={non_hits.mean():.3f}, median={non_hits.median():.3f}")
    print(f"    Hits ({len(hits):,}): mean={hits.mean():.3f}, median={hits.median():.3f}")


def main():
    print("="*60)
    print("  ROBUSTNESS CHECKS FOR TYPICALITY ANALYSIS")
    print("="*60)
    
    # Load data with artist info
    df = load_sample_with_artist()
    
    # 1. Vary k
    print("\n--- 1. Varying k ---")
    df_clean, k_results = compute_typicality_varied_k(df, k_values=[25, 50, 100, 200])
    plot_k_comparison(df_clean, k_results)
    
    # Standard k=100 for other comparisons
    typ_standard = k_results[100]
    
    # 2. Exclude same-artist
    print("\n--- 2. Exclude same-artist neighbors ---")
    df_artist, typ_no_artist = compute_typicality_no_same_artist(df)
    
    # 3. Non-hits reference
    print("\n--- 3. Non-hits reference set ---")
    df_nohits, typ_no_hits = compute_typicality_no_hits_reference(df_clean, hit_threshold=80)
    
    # Plot comparison
    print("\n--- Generating comparison plots ---")
    plot_robustness_comparison(df_clean, typ_standard, typ_no_artist, typ_no_hits)
    
    # 4. Hit density plot
    print("\n--- 4. Hit vs non-hit density ---")
    plot_hit_density(df_clean, typ_standard, hit_threshold=70)
    
    print("\n" + "="*60)
    print("  ROBUSTNESS CHECKS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

