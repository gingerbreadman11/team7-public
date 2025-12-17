# -*- coding: utf-8 -*-
"""
Temporal Typicality Visualization
=================================
Generate plots comparing typicality vs popularity across 6-month periods.

Outputs:
- Individual period scatter/hexbin plots
- 6-panel temporal evolution plot
- Overlaid comparison plot
- Summary statistics table
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional
from scipy import stats as scipy_stats

# Custom color palette - deep purple to red gradient
PERIOD_COLORS = [
    '#67129A',  # Period 1 - deep purple
    '#7B1A8E',  # Period 2
    '#9A2578',  # Period 3 - magenta-pink
    '#B52060',  # Period 4
    '#D20D52',  # Period 5 - rich red
    '#E8184A',  # Period 6 - bright red
]

# Period labels for display
PERIOD_LABELS = [
    "Mar-Aug 2020",
    "Sep 2020 - Feb 2021",
    "Mar-Aug 2021",
    "Sep 2021 - Feb 2022",
    "Mar-Aug 2022",
    "Sep 2022 - Jan 2023",
]

# Period name mapping
PERIOD_NAMES = [
    "2020-03_2020-08",
    "2020-09_2021-02",
    "2021-03_2021-08",
    "2021-09_2022-02",
    "2022-03_2022-08",
    "2022-09_2023-01",
]


def load_period_data(output_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all period data from output directory.
    
    Args:
        output_path: Base output directory with period subfolders
    
    Returns:
        Dict mapping period_name -> DataFrame with typicality
    """
    results = {}
    
    for period_name in PERIOD_NAMES:
        period_dir = os.path.join(output_path, f"period_{period_name}")
        typ_path = os.path.join(period_dir, 'sample_with_typicality.csv')
        
        if os.path.exists(typ_path):
            df = pd.read_csv(typ_path)
            results[period_name] = df
            print(f"Loaded {period_name}: {len(df):,} rows")
        else:
            print(f"Warning: {period_name} not found at {typ_path}")
    
    return results


def plot_period_hexbin(
    df: pd.DataFrame,
    period_name: str,
    period_label: str,
    output_path: str,
    color: str
):
    """
    Create hexbin plot of typicality vs popularity for a single period.
    
    Args:
        df: DataFrame with typicality and popularity
        period_name: Internal period name
        period_label: Display label
        output_path: Path to save figure
        color: Color for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get data
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    if typ_col not in df.columns or pop_col not in df.columns:
        print(f"Warning: Required columns not found for {period_name}")
        return
    
    x = df[typ_col].dropna()
    y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
    
    # Align indices
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]
    
    if len(x) < 10:
        print(f"Warning: Not enough data for {period_name}")
        return
    
    # Hexbin plot
    hb = ax.hexbin(x, y, gridsize=50, cmap='YlOrRd', mincnt=1)
    cb = plt.colorbar(hb, ax=ax, label='Count')
    
    # Add trend line
    z = np.polyfit(x, y, 2)  # Quadratic fit
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color=color, linewidth=3, linestyle='--', label='Quadratic Fit')
    
    # Calculate correlation
    corr, p_val = scipy_stats.pearsonr(x, y)
    
    # Stats box
    stats_text = f"N = {len(x):,}\nr = {corr:.3f} (p={p_val:.2e})"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Typicality (normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Popularity', fontsize=12, fontweight='bold')
    ax.set_title(f'Typicality vs Popularity: {period_label}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'typicality_hexbin_{period_name}.png'), dpi=150)
    plt.savefig(os.path.join(output_path, f'typicality_hexbin_{period_name}.pdf'))
    plt.close()


def plot_six_panel_evolution(
    all_results: Dict[str, pd.DataFrame],
    output_path: str
):
    """
    Create 6-panel plot showing typicality vs popularity evolution.
    
    Args:
        all_results: Dict of period_name -> DataFrame
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.25)
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    for i, (period_name, period_label, color) in enumerate(zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS)):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        
        if period_name not in all_results:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(period_label, fontsize=12, fontweight='bold')
            continue
        
        df = all_results[period_name]
        
        if typ_col not in df.columns or pop_col not in df.columns:
            ax.text(0.5, 0.5, 'Missing columns', ha='center', va='center', fontsize=14)
            ax.set_title(period_label, fontsize=12, fontweight='bold')
            continue
        
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(x) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=14)
            ax.set_title(period_label, fontsize=12, fontweight='bold')
            continue
        
        # 2D histogram / hexbin
        hb = ax.hexbin(x, y, gridsize=30, cmap='YlOrRd', mincnt=1, alpha=0.9)
        
        # Quadratic fit
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='-')
        
        # Correlation
        corr, _ = scipy_stats.pearsonr(x, y)
        
        ax.text(0.02, 0.98, f"N={len(x):,}\nr={corr:.3f}", transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Typicality' if i >= 3 else '', fontsize=10)
        ax.set_ylabel('Popularity' if i % 3 == 0 else '', fontsize=10)
        ax.set_title(period_label, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
    
    fig.suptitle('Temporal Evolution: Typicality vs Popularity (2020-2023)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'temporal_evolution_6panel.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'temporal_evolution_6panel.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved 6-panel evolution plot")


def plot_overlaid_trends(
    all_results: Dict[str, pd.DataFrame],
    output_path: str
):
    """
    Create overlaid plot showing quadratic trends for all periods.
    
    Args:
        all_results: Dict of period_name -> DataFrame
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    for period_name, period_label, color in zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS):
        if period_name not in all_results:
            continue
        
        df = all_results[period_name]
        
        if typ_col not in df.columns or pop_col not in df.columns:
            continue
        
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx].values
        y = y.loc[common_idx].values
        
        if len(x) < 10:
            continue
        
        # Quadratic fit
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        
        # Plot trend line
        ax.plot(x_line, p(x_line), color=color, linewidth=3, label=period_label)
        
        # Add confidence band using bootstrap (simplified)
        # Just show the line for clarity
    
    ax.set_xlabel('Typicality (normalized)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Popularity', fontsize=13, fontweight='bold')
    ax.set_title('Optimal Distinctiveness Over Time\n(Quadratic Trend Lines)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'temporal_trends_overlaid.png'), dpi=200)
    plt.savefig(os.path.join(output_path, 'temporal_trends_overlaid.pdf'))
    plt.close()
    
    print(f"Saved overlaid trends plot")


def plot_summary_metrics(
    all_results: Dict[str, pd.DataFrame],
    output_path: str
):
    """
    Create bar charts showing summary metrics across periods.
    
    Args:
        all_results: Dict of period_name -> DataFrame
        output_path: Path to save figure
    """
    typ_col = 'typicality_cosine_norm'
    pop_col = 'track.popularity'
    
    # Collect metrics
    metrics = {
        'period': [],
        'n_songs': [],
        'mean_typicality': [],
        'mean_popularity': [],
        'correlation': [],
        'optimal_typicality': []
    }
    
    for period_name, period_label in zip(PERIOD_NAMES, PERIOD_LABELS):
        metrics['period'].append(period_label)
        
        if period_name not in all_results:
            metrics['n_songs'].append(0)
            metrics['mean_typicality'].append(np.nan)
            metrics['mean_popularity'].append(np.nan)
            metrics['correlation'].append(np.nan)
            metrics['optimal_typicality'].append(np.nan)
            continue
        
        df = all_results[period_name]
        
        x = df[typ_col].dropna()
        y = pd.to_numeric(df[pop_col], errors='coerce').dropna()
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        metrics['n_songs'].append(len(x))
        metrics['mean_typicality'].append(x.mean())
        metrics['mean_popularity'].append(y.mean())
        
        if len(x) > 10:
            corr, _ = scipy_stats.pearsonr(x, y)
            metrics['correlation'].append(corr)
            
            # Find optimal typicality (vertex of quadratic fit)
            z = np.polyfit(x, y, 2)
            if z[0] != 0:  # Parabola
                optimal_x = -z[1] / (2 * z[0])
                metrics['optimal_typicality'].append(optimal_x)
            else:
                metrics['optimal_typicality'].append(np.nan)
        else:
            metrics['correlation'].append(np.nan)
            metrics['optimal_typicality'].append(np.nan)
    
    df_metrics = pd.DataFrame(metrics)
    
    # Save as CSV
    df_metrics.to_csv(os.path.join(output_path, 'temporal_summary_metrics.csv'), index=False)
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x_pos = range(len(PERIOD_LABELS))
    short_labels = ['Mar-Aug\n2020', 'Sep 20-\nFeb 21', 'Mar-Aug\n2021', 
                    'Sep 21-\nFeb 22', 'Mar-Aug\n2022', 'Sep 22-\nJan 23']
    
    # 1. Number of songs
    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, df_metrics['n_songs'], color=PERIOD_COLORS, alpha=0.8)
    ax1.set_ylabel('Number of Unique Songs', fontweight='bold')
    ax1.set_title('Sample Size per Period', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_labels, fontsize=9)
    for bar, val in zip(bars, df_metrics['n_songs']):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:,}', ha='center', va='bottom', fontsize=8)
    
    # 2. Mean Typicality
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, df_metrics['mean_typicality'], color=PERIOD_COLORS, alpha=0.8)
    ax2.set_ylabel('Mean Typicality', fontweight='bold')
    ax2.set_title('Average Typicality per Period', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_labels, fontsize=9)
    ax2.set_ylim(0, 1)
    
    # 3. Correlation
    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, df_metrics['correlation'], color=PERIOD_COLORS, alpha=0.8)
    ax3.set_ylabel('Pearson Correlation (r)', fontweight='bold')
    ax3.set_title('Typicality-Popularity Correlation', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(short_labels, fontsize=9)
    ax3.axhline(0, color='black', linewidth=0.5)
    
    # 4. Optimal Typicality
    ax4 = axes[1, 1]
    bars = ax4.bar(x_pos, df_metrics['optimal_typicality'], color=PERIOD_COLORS, alpha=0.8)
    ax4.set_ylabel('Optimal Typicality', fontweight='bold')
    ax4.set_title('Optimal Typicality (Quadratic Vertex)', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(short_labels, fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Mid-point')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'temporal_summary_metrics.png'), dpi=200)
    plt.savefig(os.path.join(output_path, 'temporal_summary_metrics.pdf'))
    plt.close()
    
    print(f"Saved summary metrics plot and CSV")


def generate_temporal_plots(
    output_path: str,
    all_results: Optional[Dict[str, pd.DataFrame]] = None
):
    """
    Generate all temporal analysis plots.
    
    Args:
        output_path: Base output directory
        all_results: Optional dict of pre-loaded results
    """
    print("="*60)
    print("  GENERATING TEMPORAL PLOTS")
    print("="*60)
    
    # Load data if not provided
    if all_results is None:
        all_results = load_period_data(output_path)
    
    if not all_results:
        print("ERROR: No period data found")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Individual hexbin plots for each period
    print("\nGenerating individual period plots...")
    for period_name, period_label, color in zip(PERIOD_NAMES, PERIOD_LABELS, PERIOD_COLORS):
        if period_name in all_results:
            plot_period_hexbin(
                all_results[period_name],
                period_name,
                period_label,
                plots_dir,
                color
            )
    
    # 2. Six-panel evolution plot
    print("\nGenerating 6-panel evolution plot...")
    plot_six_panel_evolution(all_results, plots_dir)
    
    # 3. Overlaid trends plot
    print("\nGenerating overlaid trends plot...")
    plot_overlaid_trends(all_results, plots_dir)
    
    # 4. Summary metrics
    print("\nGenerating summary metrics...")
    plot_summary_metrics(all_results, plots_dir)
    
    print("\n" + "="*60)
    print("  PLOTS COMPLETE")
    print("="*60)
    print(f"All plots saved to: {plots_dir}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Temporal Typicality Plots')
    parser.add_argument('--output-path', type=str, default='results_temporal',
                       help='Path to temporal analysis results directory')
    
    args = parser.parse_args()
    
    # Resolve path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(base_dir, args.output_path)
    
    generate_temporal_plots(args.output_path)

