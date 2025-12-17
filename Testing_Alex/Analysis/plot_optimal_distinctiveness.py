# -*- coding: utf-8 -*-
"""
Plot Optimal Distinctiveness Results
====================================
Main figures for the poster:
1. Typicality vs Mean Popularity (binned curve) - the inverted-U test
2. Regression summary with typicality and typicality_sq
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'results_large', 'sample_with_typicality.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
pbStart = "#67129A"
pbMid = "#9A2578"
pbEnd = "#D20D52"

# Features for regression controls
AUDIO_CONTROLS = [
    'danceability', 'energy', 'loudness', 'valence',
    'instrumentalness', 'acousticness', 'speechiness', 'tempo'
]


def plot_typicality_vs_popularity(df, n_bins=20):
    """
    Main result plot: binned typicality vs mean popularity.
    Looking for inverted-U (optimal distinctiveness).
    """
    # Use normalized typicality for x-axis
    df = df.dropna(subset=['typicality_norm', 'track.popularity']).copy()
    
    # Create quantile bins for typicality
    df['typicality_bin'] = pd.qcut(df['typicality_norm'], q=n_bins, labels=False, duplicates='drop')
    
    # Aggregate by bin
    binned = df.groupby('typicality_bin').agg({
        'typicality_norm': 'mean',
        'track.popularity': ['mean', 'median', 'std', 'count']
    }).reset_index()
    
    binned.columns = ['bin', 'typicality_mean', 'pop_mean', 'pop_median', 'pop_std', 'count']
    
    # Also compute share of "hits" (popularity >= 70)
    df['is_hit'] = (df['track.popularity'] >= 70).astype(int)
    hit_rate = df.groupby('typicality_bin')['is_hit'].mean().reset_index()
    hit_rate.columns = ['bin', 'hit_rate']
    binned = binned.merge(hit_rate, on='bin')
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Main line: mean popularity
    ax1.plot(binned['typicality_mean'], binned['pop_mean'], 
             'o-', color=pbMid, linewidth=2, markersize=6, label='Mean Popularity')
    ax1.fill_between(binned['typicality_mean'],
                     binned['pop_mean'] - binned['pop_std']/np.sqrt(binned['count']),
                     binned['pop_mean'] + binned['pop_std']/np.sqrt(binned['count']),
                     alpha=0.2, color=pbMid)
    
    ax1.set_xlabel('Typicality (0 = Distinctive, 1 = Typical)', fontsize=11)
    ax1.set_ylabel('Mean Popularity (0–100)', fontsize=11, color=pbMid)
    ax1.tick_params(axis='y', labelcolor=pbMid)
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Secondary axis: hit rate
    ax2 = ax1.twinx()
    ax2.plot(binned['typicality_mean'], binned['hit_rate'] * 100, 
             's--', color=pbEnd, linewidth=1.5, markersize=5, alpha=0.7, label='Hit Rate (≥70)')
    ax2.set_ylabel('Hit Rate % (popularity ≥ 70)', fontsize=11, color=pbEnd)
    ax2.tick_params(axis='y', labelcolor=pbEnd)
    ax2.set_ylim(0, 50)
    
    # Title
    plt.title('Typicality vs Popularity: Testing Optimal Distinctiveness', fontsize=12, fontweight='bold')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save
    out_png = os.path.join(OUTPUT_DIR, 'typicality_vs_popularity.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'typicality_vs_popularity.pdf')
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_png}")
    
    return binned


def run_regression(df):
    """
    Run OLS regression: popularity ~ typicality + typicality_sq + audio controls.
    Test for inverted-U: typicality positive, typicality_sq negative.
    """
    df = df.dropna(subset=['typicality_norm', 'typicality_norm_sq', 'track.popularity'] + AUDIO_CONTROLS).copy()
    
    # Prepare features
    X_cols = ['typicality_norm', 'typicality_norm_sq'] + AUDIO_CONTROLS
    X = df[X_cols].copy()
    X = sm.add_constant(X)
    y = df['track.popularity']
    
    print("\n" + "="*60)
    print("  REGRESSION: popularity ~ typicality + typicality² + controls")
    print("="*60)
    
    model = sm.OLS(y, X).fit()
    
    print(model.summary())
    
    # Extract key coefficients
    print("\n--- KEY COEFFICIENTS (Optimal Distinctiveness Test) ---")
    print(f"  typicality:    coef = {model.params['typicality_norm']:.4f}, p = {model.pvalues['typicality_norm']:.4f}")
    print(f"  typicality_sq: coef = {model.params['typicality_norm_sq']:.4f}, p = {model.pvalues['typicality_norm_sq']:.4f}")
    
    # Interpretation
    typ_coef = model.params['typicality_norm']
    typ_sq_coef = model.params['typicality_norm_sq']
    
    if typ_coef > 0 and typ_sq_coef < 0:
        print("\n  ✓ INVERTED-U PATTERN DETECTED")
        print("    → Typicality has a positive linear effect, but diminishing returns (negative quadratic).")
        print("    → Optimal distinctiveness hypothesis SUPPORTED.")
    elif typ_coef < 0 and typ_sq_coef > 0:
        print("\n  ⚠ U-SHAPED PATTERN (opposite of inverted-U)")
        print("    → Extreme values (very typical OR very distinctive) perform better.")
    elif typ_coef > 0 and typ_sq_coef >= 0:
        print("\n  → MONOTONIC POSITIVE: More typical = more popular (no optimum).")
    elif typ_coef < 0 and typ_sq_coef <= 0:
        print("\n  → MONOTONIC NEGATIVE: More distinctive = more popular (no optimum).")
    else:
        print("\n  → FLAT or inconclusive pattern.")
    
    # Save summary to text file
    summary_path = os.path.join(BASE_DIR, 'results_large', 'regression_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(model.summary().as_text())
    print(f"\nSaved regression summary to: {summary_path}")
    
    return model


def main():
    print("="*60)
    print("  OPTIMAL DISTINCTIVENESS ANALYSIS")
    print("="*60)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Run compute_typicality.py first.")
        return
    
    print(f"Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df):,}")
    
    # 1. Main plot: typicality vs popularity
    print("\n--- Generating typicality vs popularity plot ---")
    binned = plot_typicality_vs_popularity(df, n_bins=20)
    print("\nBinned statistics:")
    print(binned[['typicality_mean', 'pop_mean', 'hit_rate']].round(3))
    
    # 2. Regression analysis
    model = run_regression(df)
    
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

