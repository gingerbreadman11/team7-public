# -*- coding: utf-8 -*-
"""
Main Poster Figure: Billboard vs Spotify
========================================
Side-by-side comparison of optimal distinctiveness patterns.
Left: Billboard era (Askin & Mauskapf's inverted-U) - schematic
Right: Spotify era (our finding) - actual data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'results_large', 'sample_with_typicality.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
pbStart = "#67129A"
pbMid = "#9A2578"
pbEnd = "#D20D52"


def create_billboard_schematic(ax):
    """Draw schematic inverted-U for Billboard era."""
    # Inverted-U curve
    x = np.linspace(0, 1, 100)
    # Quadratic with peak around 0.5
    y = -4 * (x - 0.5)**2 + 1
    y = y * 40 + 30  # Scale to ~30-70 range
    
    ax.plot(x, y, color=pbStart, linewidth=3, label='Billboard (Askin & Mauskapf)')
    ax.fill_between(x, 25, y, alpha=0.2, color=pbStart)
    
    # Mark optimal point
    ax.scatter([0.5], [70], s=100, color=pbStart, zorder=5, edgecolors='black', linewidths=1.5)
    ax.annotate('Optimal\nDistinctiveness', xy=(0.5, 70), xytext=(0.5, 78),
                ha='center', fontsize=10, fontweight='bold', color=pbStart)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(25, 85)
    ax.set_xlabel('Typicality →', fontsize=11)
    ax.set_ylabel('Success', fontsize=11)
    ax.set_title('Billboard Era (1958–2016)\n"Just Different Enough"', fontsize=13, fontweight='bold', pad=10)
    
    # Add arrows
    ax.annotate('', xy=(0.15, 28), xytext=(0.02, 28),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.08, 26, 'Distinctive', fontsize=9, ha='center', color='gray')
    
    ax.annotate('', xy=(0.98, 28), xytext=(0.85, 28),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.92, 26, 'Typical', fontsize=9, ha='center', color='gray')
    
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks([])


def create_spotify_plot(ax, df):
    """Plot actual Spotify data showing monotonic/U-shape pattern."""
    df = df.dropna(subset=['typicality_norm', 'track.popularity']).copy()
    
    # Bin typicality
    n_bins = 20
    df['bin'] = pd.qcut(df['typicality_norm'], q=n_bins, labels=False, duplicates='drop')
    
    binned = df.groupby('bin').agg({
        'typicality_norm': 'mean',
        'track.popularity': ['mean', 'std', 'count']
    }).reset_index()
    binned.columns = ['bin', 'typ', 'pop_mean', 'pop_std', 'count']
    
    # Compute standard error
    binned['se'] = binned['pop_std'] / np.sqrt(binned['count'])
    
    # Plot with confidence band
    ax.fill_between(binned['typ'], 
                    binned['pop_mean'] - 1.96*binned['se'],
                    binned['pop_mean'] + 1.96*binned['se'],
                    alpha=0.3, color=pbEnd)
    ax.plot(binned['typ'], binned['pop_mean'], 'o-', color=pbEnd, linewidth=3, 
            markersize=6, label='Spotify (2020–2022)')
    
    # Mark the high typicality advantage
    max_idx = binned['pop_mean'].idxmax()
    max_typ = binned.loc[max_idx, 'typ']
    max_pop = binned.loc[max_idx, 'pop_mean']
    ax.scatter([max_typ], [max_pop], s=100, color=pbEnd, zorder=5, edgecolors='black', linewidths=1.5)
    ax.annotate('Highest\nSuccess', xy=(max_typ, max_pop), xytext=(max_typ - 0.15, max_pop + 5),
                ha='center', fontsize=10, fontweight='bold', color=pbEnd,
                arrowprops=dict(arrowstyle='->', color=pbEnd))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(25, 85)
    ax.set_xlabel('Typicality →', fontsize=11)
    ax.set_ylabel('Popularity (0–100)', fontsize=11)
    ax.set_title('Spotify Era (2020–2022)\n"As Typical As Possible"', fontsize=13, fontweight='bold', pad=10)
    
    # Add arrows
    ax.annotate('', xy=(0.15, 28), xytext=(0.02, 28),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.08, 26, 'Distinctive', fontsize=9, ha='center', color='gray')
    
    ax.annotate('', xy=(0.98, 28), xytext=(0.85, 28),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.92, 26, 'Typical', fontsize=9, ha='center', color='gray')
    
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks([])


def main():
    print("="*60)
    print("  CREATING MAIN POSTER FIGURE")
    print("="*60)
    
    # Load data
    print("Loading Spotify data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df):,}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Billboard schematic
    create_billboard_schematic(ax1)
    
    # Right: Spotify actual data
    create_spotify_plot(ax2, df)
    
    # Overall title
    fig.suptitle('Optimal Distinctiveness: Then vs Now', fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    out_png = os.path.join(OUTPUT_DIR, 'billboard_vs_spotify.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'billboard_vs_spotify.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    
    # Also create a summary stats printout
    print("\n--- KEY STATISTICS ---")
    df_clean = df.dropna(subset=['typicality_norm', 'track.popularity'])
    
    # Split into quintiles
    df_clean['quintile'] = pd.qcut(df_clean['typicality_norm'], q=5, labels=['Q1 (Distinctive)', 'Q2', 'Q3', 'Q4', 'Q5 (Typical)'])
    
    summary = df_clean.groupby('quintile')['track.popularity'].agg(['mean', 'median', 'count'])
    print("\nPopularity by Typicality Quintile:")
    print(summary.round(2))
    
    # Hit rate by quintile
    df_clean['is_hit'] = df_clean['track.popularity'] >= 70
    hit_summary = df_clean.groupby('quintile')['is_hit'].mean() * 100
    print("\nHit Rate (≥70) by Typicality Quintile:")
    print(hit_summary.round(2))


if __name__ == "__main__":
    main()

