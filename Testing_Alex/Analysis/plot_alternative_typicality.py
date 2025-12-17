# -*- coding: utf-8 -*-
"""
Plot Alternative Typicality Results
===================================
Compare:
- Standard typicality (cosine kNN)
- Popularity-weighted typicality
- Temporal window typicality

No titles, PDF only, large axis labels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Font sizes
SCALE = 1.5625
LABEL_SIZE = int(12 * SCALE)
TICK_SIZE = int(10 * SCALE)
LEGEND_SIZE = int(9 * SCALE)

# Colors
COLOR_STANDARD = '#3498db'
COLOR_POP_WEIGHTED = '#e74c3c'
COLOR_TEMPORAL = '#27ae60'


def load_data(results_path: str) -> pd.DataFrame:
    """Load the sample with all typicality measures."""
    path = os.path.join(results_path, 'alternative_typicality', 'sample_with_all_typicality.csv')
    df = pd.read_csv(path)
    df['track.popularity'] = pd.to_numeric(df['track.popularity'], errors='coerce')
    print(f"Loaded {len(df):,} rows with all typicality measures")
    return df


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def plot_typicality_comparison_scatter(df: pd.DataFrame, output_dir: str):
    """Scatter plots comparing typicality measures."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    comparisons = [
        ('typicality_cosine_norm', 'typicality_pop_weighted', 'Standard', 'Pop-Weighted'),
        ('typicality_cosine_norm', 'typicality_temporal', 'Standard', 'Temporal'),
        ('typicality_pop_weighted', 'typicality_temporal', 'Pop-Weighted', 'Temporal'),
    ]
    
    for ax, (col1, col2, name1, name2) in zip(axes, comparisons):
        x = df[col1].dropna()
        y = df[col2].dropna()
        common = x.index.intersection(y.index)
        x, y = x.loc[common].values, y.loc[common].values
        
        # Hexbin
        colors_gb = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
        cmap = mcolors.LinearSegmentedColormap.from_list('blues', colors_gb)
        
        ax.hexbin(x, y, gridsize=40, cmap=cmap, mincnt=1)
        
        # Correlation
        corr, _ = scipy_stats.pearsonr(x, y)
        ax.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=TICK_SIZE,
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel(f'{name1} Typicality', fontsize=TICK_SIZE, fontweight='bold')
        ax.set_ylabel(f'{name2} Typicality', fontsize=TICK_SIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_SIZE * 0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_comparison_scatter.pdf'))
    plt.close()
    print("Saved: typicality_comparison_scatter.pdf")


def plot_typicality_distributions_overlay(df: pd.DataFrame, output_dir: str):
    """KDE overlay of all typicality measures."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    measures = [
        ('typicality_cosine_norm', 'Standard (Cosine kNN)', COLOR_STANDARD),
        ('typicality_pop_weighted', 'Popularity-Weighted', COLOR_POP_WEIGHTED),
        ('typicality_temporal', 'Temporal Window', COLOR_TEMPORAL),
    ]
    
    for col, label, color in measures:
        data = df[col].dropna()
        if len(data) > 100:
            data.plot.kde(ax=ax, color=color, linewidth=3, label=label)
    
    ax.set_xlabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Density', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper left')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_distributions_overlay.pdf'))
    plt.close()
    print("Saved: typicality_distributions_overlay.pdf")


def plot_popularity_relationship_comparison(df: pd.DataFrame, output_dir: str):
    """Compare typicality-popularity relationship for each measure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    measures = [
        ('typicality_cosine_norm', 'Standard (Cosine kNN)', COLOR_STANDARD),
        ('typicality_pop_weighted', 'Popularity-Weighted', COLOR_POP_WEIGHTED),
        ('typicality_temporal', 'Temporal Window', COLOR_TEMPORAL),
    ]
    
    colors_gb = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#084081']
    cmap_gb = mcolors.LinearSegmentedColormap.from_list('green_blue', colors_gb)
    
    for ax, (col, label, color) in zip(axes, measures):
        data = df[[col, 'track.popularity']].dropna()
        x = data[col].values
        y = data['track.popularity'].values
        
        if len(x) < 100:
            continue
        
        # Hexbin
        ax.hexbin(x, y, gridsize=35, cmap=cmap_gb, mincnt=1)
        
        # Quadratic fit
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_line = np.linspace(0.05, 0.95, 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=3)
        
        # Stats
        corr, _ = scipy_stats.pearsonr(x, y)
        ax.text(0.02, 0.98, f'r = {corr:.3f}\nn = {len(x):,}', transform=ax.transAxes, 
                fontsize=TICK_SIZE * 0.9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Label in corner
        ax.text(0.98, 0.02, label, transform=ax.transAxes, fontsize=TICK_SIZE * 0.9,
                ha='right', va='bottom', fontweight='bold', color=color)
        
        ax.set_xlabel('Typicality', fontsize=TICK_SIZE, fontweight='bold')
        ax.set_ylabel('Popularity' if ax == axes[0] else '', fontsize=TICK_SIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_SIZE * 0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'popularity_relationship_comparison.pdf'))
    plt.close()
    print("Saved: popularity_relationship_comparison.pdf")


def plot_trend_lines_overlay(df: pd.DataFrame, output_dir: str):
    """Overlay trend lines for all typicality measures."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    measures = [
        ('typicality_cosine_norm', 'Standard (Cosine kNN)', COLOR_STANDARD, '-'),
        ('typicality_pop_weighted', 'Popularity-Weighted', COLOR_POP_WEIGHTED, '--'),
        ('typicality_temporal', 'Temporal Window', COLOR_TEMPORAL, ':'),
    ]
    
    for col, label, color, linestyle in measures:
        data = df[[col, 'track.popularity']].dropna()
        x = data[col].values
        y = data['track.popularity'].values
        
        if len(x) < 100:
            continue
        
        # Compute binned means
        n_bins = 20
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []
        bin_sems = []
        
        for i in range(n_bins):
            mask = (x >= bins[i]) & (x < bins[i+1])
            if mask.sum() > 5:
                bin_means.append(y[mask].mean())
                bin_sems.append(y[mask].std() / np.sqrt(mask.sum()) * 1.96)
            else:
                bin_means.append(np.nan)
                bin_sems.append(np.nan)
        
        bin_means = np.array(bin_means)
        bin_sems = np.array(bin_sems)
        
        valid = ~np.isnan(bin_means)
        bc = bin_centers[valid]
        bm = bin_means[valid]
        bs = bin_sems[valid]
        
        # Smooth
        bm_smooth = gaussian_filter1d(bm, sigma=1)
        
        ax.plot(bc, bm_smooth, color=color, linewidth=3, linestyle=linestyle, label=label)
        ax.fill_between(bc, bm_smooth - bs, bm_smooth + bs, color=color, alpha=0.15)
    
    ax.set_xlabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Popularity (± 95% CI)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='best')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_lines_overlay.pdf'))
    plt.close()
    print("Saved: trend_lines_overlay.pdf")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Correlation heatmap of all typicality measures + popularity."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    cols = ['typicality_cosine_norm', 'typicality_pop_weighted', 'typicality_temporal', 'track.popularity']
    labels = ['Standard\nTypicality', 'Pop-Weighted\nTypicality', 'Temporal\nTypicality', 'Popularity']
    
    corr_matrix = df[cols].corr()
    
    # Custom colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap=cmap, center=0,
                square=True, linewidths=1, cbar_kws={'shrink': 0.8},
                annot_kws={'size': TICK_SIZE}, ax=ax,
                xticklabels=labels, yticklabels=labels)
    
    ax.tick_params(axis='both', labelsize=TICK_SIZE * 0.85)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.pdf'))
    plt.close()
    print("Saved: correlation_heatmap.pdf")


def plot_boxplot_comparison(df: pd.DataFrame, output_dir: str):
    """Boxplot comparing typicality distributions."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    cols = ['typicality_cosine_norm', 'typicality_pop_weighted', 'typicality_temporal']
    labels = ['Standard\n(Cosine kNN)', 'Popularity-\nWeighted', 'Temporal\nWindow']
    colors = [COLOR_STANDARD, COLOR_POP_WEIGHTED, COLOR_TEMPORAL]
    
    data = [df[col].dropna().values for col in cols]
    
    bp = ax.boxplot(data, positions=range(len(cols)), widths=0.6, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_ylabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_boxplot_comparison.pdf'))
    plt.close()
    print("Saved: typicality_boxplot_comparison.pdf")


def plot_popularity_bins_comparison(df: pd.DataFrame, output_dir: str):
    """Mean typicality by popularity bins for each measure."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    measures = [
        ('typicality_cosine_norm', 'Standard', COLOR_STANDARD, 'o'),
        ('typicality_pop_weighted', 'Pop-Weighted', COLOR_POP_WEIGHTED, 's'),
        ('typicality_temporal', 'Temporal', COLOR_TEMPORAL, '^'),
    ]
    
    # Bin by popularity
    df['pop_bin'] = pd.cut(df['track.popularity'], bins=10, labels=False)
    
    for col, label, color, marker in measures:
        binned = df.groupby('pop_bin')[col].agg(['mean', 'std', 'count'])
        binned['sem'] = binned['std'] / np.sqrt(binned['count']) * 1.96
        
        valid = binned['count'] > 10
        x = binned.index[valid] * 10 + 5  # Bin centers (0-10, 10-20, etc.)
        y = binned['mean'][valid].values
        yerr = binned['sem'][valid].values
        
        ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, markersize=10,
                   linewidth=2, capsize=4, label=label, markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_by_popularity_bins.pdf'))
    plt.close()
    print("Saved: typicality_by_popularity_bins.pdf")


def plot_summary_stats(df: pd.DataFrame, output_dir: str):
    """Summary statistics bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cols = ['typicality_cosine_norm', 'typicality_pop_weighted', 'typicality_temporal']
    labels = ['Standard', 'Pop-Weighted', 'Temporal']
    colors = [COLOR_STANDARD, COLOR_POP_WEIGHTED, COLOR_TEMPORAL]
    
    # Correlation with popularity
    ax1 = axes[0]
    corrs = [df[col].corr(df['track.popularity']) for col in cols]
    bars = ax1.bar(range(len(cols)), corrs, color=colors, edgecolor='white', linewidth=2)
    ax1.set_xticks(range(len(cols)))
    ax1.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax1.set_ylabel('Correlation with Popularity', fontsize=TICK_SIZE, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=TICK_SIZE)
    ax1.axhline(0, color='black', linewidth=0.5)
    
    # Add values on bars
    for bar, val in zip(bars, corrs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005 * np.sign(val), 
                f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=TICK_SIZE)
    
    # Mean and std
    ax2 = axes[1]
    means = [df[col].mean() for col in cols]
    stds = [df[col].std() for col in cols]
    
    x = np.arange(len(cols))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, means, width, label='Mean', color=colors, alpha=0.9)
    bars2 = ax2.bar(x + width/2, stds, width, label='Std Dev', color=colors, alpha=0.5, hatch='//')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax2.set_ylabel('Value', fontsize=TICK_SIZE, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=TICK_SIZE)
    ax2.legend(fontsize=LEGEND_SIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.pdf'))
    plt.close()
    print("Saved: summary_statistics.pdf")


def main():
    """Generate all alternative typicality plots."""
    print("="*60)
    print("  ALTERNATIVE TYPICALITY PLOTS")
    print("="*60)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, 'results_temporal')
    output_dir = os.path.join(results_path, 'plots_alt_typicality')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    df = load_data(results_path)
    
    # Generate plots
    print("\n" + "-"*40)
    print("Generating comparison plots...")
    plot_typicality_comparison_scatter(df, output_dir)
    plot_typicality_distributions_overlay(df, output_dir)
    plot_popularity_relationship_comparison(df, output_dir)
    plot_trend_lines_overlay(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_boxplot_comparison(df, output_dir)
    plot_popularity_bins_comparison(df, output_dir)
    plot_summary_stats(df, output_dir)
    
    print("\n" + "="*60)
    print("  ALL PLOTS COMPLETE")
    print("="*60)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

