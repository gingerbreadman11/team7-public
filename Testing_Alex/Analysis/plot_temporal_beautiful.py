# -*- coding: utf-8 -*-
"""
Beautiful Temporal Plots
========================
Generate publication-quality plots for temporal typicality analysis.

- Combined heatmap (green-blue colormap)
- Multiple trend line styles (GP-smooth, polynomial, zigzag)
- No titles, larger axis labels (25% bigger)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Period configuration
PERIOD_NAMES = [
    "2020-03_2020-08",
    "2020-09_2021-02",
    "2021-03_2021-08",
    "2021-09_2022-02",
    "2022-03_2022-08",
    "2022-09_2023-01",
]

PERIOD_LABELS = [
    "Mar-Aug 2020",
    "Sep 2020 - Feb 2021",
    "Mar-Aug 2021",
    "Sep 2021 - Feb 2022",
    "Mar-Aug 2022",
    "Sep 2022 - Jan 2023",
]

# Color palette for period lines
PERIOD_COLORS = [
    '#1a5276',  # Dark blue
    '#2874a6',  # Medium blue
    '#3498db',  # Light blue
    '#48c9b0',  # Teal
    '#27ae60',  # Green
    '#82e0aa',  # Light green
]

# Base font sizes (scaled 56% larger = 1.25 * 1.25)
BASE_LABEL_SIZE = 12
BASE_TICK_SIZE = 10
SCALE = 1.5625  # 25% bigger twice

LABEL_SIZE = int(BASE_LABEL_SIZE * SCALE)
TICK_SIZE = int(BASE_TICK_SIZE * SCALE)


def load_all_period_data(results_path: str) -> dict:
    """Load all period data from cache."""
    all_data = {}
    for period_name in PERIOD_NAMES:
        period_dir = os.path.join(results_path, f"period_{period_name}")
        typ_path = os.path.join(period_dir, 'sample_with_typicality.csv')
        if os.path.exists(typ_path):
            df = pd.read_csv(typ_path)
            all_data[period_name] = df
            print(f"Loaded {period_name}: {len(df):,} rows")
    return all_data


def combine_all_data(all_data: dict) -> pd.DataFrame:
    """Combine all period data into one DataFrame."""
    dfs = []
    for period_name, df in all_data.items():
        df_copy = df.copy()
        df_copy['period'] = period_name
        dfs.append(df_copy)
    return pd.concat(dfs, ignore_index=True)


def plot_combined_heatmap_green_blue(all_data: dict, output_dir: str):
    """
    Create a combined heatmap of all periods with green-blue colormap.
    No title, larger axis labels.
    """
    # Combine all data
    combined = combine_all_data(all_data)
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    x = combined[typ_col].dropna()
    y = pd.to_numeric(combined[pop_col], errors='coerce').dropna()
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx].values
    y = y.loc[common_idx].values
    
    print(f"Combined data: {len(x):,} points")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom green-blue colormap
    colors_gb = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#084081']
    cmap_gb = mcolors.LinearSegmentedColormap.from_list('green_blue', colors_gb)
    
    # Hexbin plot
    hb = ax.hexbin(x, y, gridsize=60, cmap=cmap_gb, mincnt=1)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Count', fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    
    # Axis labels - no title
    ax.set_xlabel('Typicality (normalized)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_combined_green_blue.pdf'))
    plt.close()
    print("Saved: heatmap_combined_green_blue.pdf")


def plot_combined_heatmap_with_trend(all_data: dict, output_dir: str):
    """
    Heatmap with quadratic trend line overlay.
    """
    combined = combine_all_data(all_data)
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    x = combined[typ_col].dropna()
    y = pd.to_numeric(combined[pop_col], errors='coerce').dropna()
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx].values
    y = y.loc[common_idx].values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom green-blue colormap
    colors_gb = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#084081']
    cmap_gb = mcolors.LinearSegmentedColormap.from_list('green_blue', colors_gb)
    
    hb = ax.hexbin(x, y, gridsize=60, cmap=cmap_gb, mincnt=1, alpha=0.9)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Count', fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    
    # Add quadratic fit
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), color='#e74c3c', linewidth=3, linestyle='-', label='Quadratic fit')
    
    ax.set_xlabel('Typicality (normalized)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc='upper right')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_combined_with_trend.pdf'))
    plt.close()
    print("Saved: heatmap_combined_with_trend.pdf")


def compute_binned_means(x: np.ndarray, y: np.ndarray, n_bins: int = 20):
    """Compute mean y values in x bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(y[mask].mean())
            bin_stds.append(y[mask].std())
            bin_counts.append(mask.sum())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)
    
    return bin_centers, np.array(bin_means), np.array(bin_stds), np.array(bin_counts)


def plot_trends_smooth_gp_style(all_data: dict, output_dir: str):
    """
    Smooth GP-like curves for each period.
    Uses Gaussian smoothing on binned means.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    for period_name, period_label, color in zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS):
        if period_name not in all_data:
            continue
        
        df = all_data[period_name]
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx].values
        y = y.loc[common_idx].values
        
        if len(x) < 50:
            continue
        
        # Compute binned means
        bin_centers, bin_means, bin_stds, _ = compute_binned_means(x, y, n_bins=25)
        
        # Remove NaN
        valid = ~np.isnan(bin_means)
        bc = bin_centers[valid]
        bm = bin_means[valid]
        bs = bin_stds[valid]
        
        if len(bc) < 5:
            continue
        
        # Smooth with Gaussian filter (GP-like appearance)
        smooth_means = gaussian_filter1d(bm, sigma=2)
        
        # Plot smooth line
        ax.plot(bc, smooth_means, color=color, linewidth=3, label=period_label)
        
        # Add confidence band
        ax.fill_between(bc, smooth_means - bs*0.3, smooth_means + bs*0.3, 
                       color=color, alpha=0.15)
    
    ax.set_xlabel('Typicality (normalized)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc='best', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trends_smooth_gp_style.pdf'))
    plt.close()
    print("Saved: trends_smooth_gp_style.pdf")


def plot_trends_polynomial(all_data: dict, output_dir: str):
    """
    Polynomial (degree 4) fits for each period.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    for period_name, period_label, color in zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS):
        if period_name not in all_data:
            continue
        
        df = all_data[period_name]
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx].values
        y = y.loc[common_idx].values
        
        if len(x) < 50:
            continue
        
        # Polynomial fit (degree 4 for more flexibility)
        z = np.polyfit(x, y, 4)
        p = np.poly1d(z)
        x_line = np.linspace(0.05, 0.95, 100)
        y_line = p(x_line)
        
        ax.plot(x_line, y_line, color=color, linewidth=3, label=period_label)
    
    ax.set_xlabel('Typicality (normalized)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Predicted Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc='best', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trends_polynomial.pdf'))
    plt.close()
    print("Saved: trends_polynomial.pdf")


def plot_trends_zigzag_with_points(all_data: dict, output_dir: str):
    """
    Zigzag/connected line plot showing binned means with points.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    for period_name, period_label, color in zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS):
        if period_name not in all_data:
            continue
        
        df = all_data[period_name]
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx].values
        y = y.loc[common_idx].values
        
        if len(x) < 50:
            continue
        
        # Compute binned means
        bin_centers, bin_means, bin_stds, bin_counts = compute_binned_means(x, y, n_bins=15)
        
        # Remove NaN
        valid = ~np.isnan(bin_means)
        bc = bin_centers[valid]
        bm = bin_means[valid]
        
        if len(bc) < 3:
            continue
        
        # Plot connected line (zigzag)
        ax.plot(bc, bm, color=color, linewidth=2.5, linestyle='-', marker='o', 
                markersize=8, markerfacecolor='white', markeredgewidth=2, 
                label=period_label)
    
    ax.set_xlabel('Typicality (normalized)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc='best', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trends_zigzag_with_points.pdf'))
    plt.close()
    print("Saved: trends_zigzag_with_points.pdf")


def plot_trends_mean_with_sem(all_data: dict, output_dir: str):
    """
    Mean with Standard Error of Mean (SEM) intervals.
    SEM = std / sqrt(n), so more data = smaller interval.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    for period_name, period_label, color in zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS):
        if period_name not in all_data:
            continue
        
        df = all_data[period_name]
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx].values
        y = y.loc[common_idx].values
        
        if len(x) < 50:
            continue
        
        # Compute binned statistics
        n_bins = 20
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []
        bin_sems = []  # Standard Error of Mean
        
        for i in range(n_bins):
            mask = (x >= bins[i]) & (x < bins[i+1])
            n_in_bin = mask.sum()
            if n_in_bin > 2:
                bin_mean = y[mask].mean()
                bin_std = y[mask].std()
                bin_sem = bin_std / np.sqrt(n_in_bin)  # SEM shrinks with more data
                bin_means.append(bin_mean)
                bin_sems.append(bin_sem * 1.96)  # 95% CI
            else:
                bin_means.append(np.nan)
                bin_sems.append(np.nan)
        
        bin_means = np.array(bin_means)
        bin_sems = np.array(bin_sems)
        
        # Remove NaN
        valid = ~np.isnan(bin_means)
        bc = bin_centers[valid]
        bm = bin_means[valid]
        bs = bin_sems[valid]
        
        if len(bc) < 3:
            continue
        
        # Smooth the mean line slightly
        bm_smooth = gaussian_filter1d(bm, sigma=1)
        
        # Plot mean line
        ax.plot(bc, bm_smooth, color=color, linewidth=2.5, label=period_label)
        
        # Plot SEM confidence band
        ax.fill_between(bc, bm_smooth - bs, bm_smooth + bs, color=color, alpha=0.2)
    
    ax.set_xlabel('Typicality (normalized)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Popularity (± 95% CI)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc='best', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trends_mean_with_sem.pdf'))
    plt.close()
    print("Saved: trends_mean_with_sem.pdf")


def plot_per_period_heatmaps(all_data: dict, output_dir: str):
    """
    Individual period heatmaps with green-blue colormap.
    """
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    colors_gb = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#084081']
    cmap_gb = mcolors.LinearSegmentedColormap.from_list('green_blue', colors_gb)
    
    for period_name, period_label, color in zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS):
        if period_name not in all_data:
            continue
        
        df = all_data[period_name]
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx].values
        y = y.loc[common_idx].values
        
        if len(x) < 50:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        hb = ax.hexbin(x, y, gridsize=40, cmap=cmap_gb, mincnt=1)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Count', fontsize=LABEL_SIZE)
        cb.ax.tick_params(labelsize=TICK_SIZE)
        
        # Add trend line
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), color='#e74c3c', linewidth=3)
        
        # Stats
        corr, _ = scipy_stats.pearsonr(x, y)
        ax.text(0.02, 0.98, f"N = {len(x):,}\nr = {corr:.3f}", 
                transform=ax.transAxes, fontsize=TICK_SIZE,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Typicality (normalized)', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Popularity', fontsize=LABEL_SIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        
        # Add period label in corner instead of title
        ax.text(0.98, 0.02, period_label, transform=ax.transAxes, fontsize=LABEL_SIZE,
                ha='right', va='bottom', fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{period_name}.pdf'))
        plt.close()
        print(f"Saved: heatmap_{period_name}.pdf")


def plot_six_panel_heatmaps(all_data: dict, output_dir: str):
    """
    6-panel combined heatmap plot with green-blue colormap.
    """
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.25, wspace=0.25)
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    colors_gb = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#084081']
    cmap_gb = mcolors.LinearSegmentedColormap.from_list('green_blue', colors_gb)
    
    for i, (period_name, period_label, color) in enumerate(zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS)):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        
        if period_name not in all_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=LABEL_SIZE)
            continue
        
        df = all_data[period_name]
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx].values
        y = y.loc[common_idx].values
        
        if len(x) < 50:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=LABEL_SIZE)
            continue
        
        hb = ax.hexbin(x, y, gridsize=30, cmap=cmap_gb, mincnt=1, alpha=0.9)
        
        # Quadratic fit
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_line = np.linspace(0.05, 0.95, 100)
        ax.plot(x_line, p(x_line), color='#c0392b', linewidth=2.5)
        
        # Correlation
        corr, _ = scipy_stats.pearsonr(x, y)
        ax.text(0.02, 0.98, f"N={len(x):,}\nr={corr:.3f}", transform=ax.transAxes,
                fontsize=TICK_SIZE * 0.9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Period label instead of title
        ax.text(0.98, 0.02, period_label, transform=ax.transAxes, fontsize=TICK_SIZE,
                ha='right', va='bottom', fontweight='bold', color=color)
        
        ax.set_xlabel('Typicality' if i >= 3 else '', fontsize=TICK_SIZE)
        ax.set_ylabel('Popularity' if i % 3 == 0 else '', fontsize=TICK_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE * 0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps_6panel.pdf'))
    plt.close()
    print("Saved: heatmaps_6panel.pdf")


def main():
    """Generate all beautiful plots."""
    print("="*60)
    print("  GENERATING BEAUTIFUL TEMPORAL PLOTS")
    print("="*60)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, 'results_temporal')
    output_dir = os.path.join(results_path, 'plots_beautiful')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading cached data...")
    all_data = load_all_period_data(results_path)
    
    if not all_data:
        print("ERROR: No period data found!")
        return
    
    # Generate plots
    print("\n" + "-"*40)
    print("Generating heatmaps...")
    plot_combined_heatmap_green_blue(all_data, output_dir)
    plot_combined_heatmap_with_trend(all_data, output_dir)
    plot_per_period_heatmaps(all_data, output_dir)
    plot_six_panel_heatmaps(all_data, output_dir)
    
    print("\n" + "-"*40)
    print("Generating trend plots...")
    plot_trends_smooth_gp_style(all_data, output_dir)
    plot_trends_polynomial(all_data, output_dir)
    plot_trends_zigzag_with_points(all_data, output_dir)
    plot_trends_mean_with_sem(all_data, output_dir)
    
    print("\n" + "="*60)
    print("  ALL PLOTS COMPLETE")
    print("="*60)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

