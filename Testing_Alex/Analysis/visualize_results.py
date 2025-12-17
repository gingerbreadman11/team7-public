# -*- coding: utf-8 -*-
"""
Results Visualization Module
============================
Generate plots for correlation analysis and model results.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: str,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (12, 10)
):
    """
    Plot correlation matrix as heatmap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle (optional - shows full matrix)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation heatmap to: {output_path}")


def plot_correlation_bars(
    correlations: pd.Series,
    output_path: str,
    title: str = "Feature Correlations with Popularity",
    figsize: tuple = (10, 8)
):
    """
    Plot correlations as horizontal bar chart.
    
    Args:
        correlations: Series with feature names as index
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by absolute correlation
    sorted_corr = correlations.sort_values(key=abs, ascending=True)
    
    # Colors based on sign
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in sorted_corr.values]
    
    # Plot
    bars = ax.barh(range(len(sorted_corr)), sorted_corr.values, color=colors)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_corr.values)):
        x_pos = val + 0.01 if val >= 0 else val - 0.01
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=9)
    
    ax.set_yticks(range(len(sorted_corr)))
    ax.set_yticklabels(sorted_corr.index)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-0.3, 0.3)  # Adjust based on data
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Positive'),
        Patch(facecolor='#e74c3c', label='Negative')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation bars to: {output_path}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str,
    title: str = "Feature Importance",
    figsize: tuple = (10, 8)
):
    """
    Plot feature importance from model.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by importance
    sorted_df = importance_df.sort_values('importance', ascending=True)
    
    # Plot
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_df)))
    bars = ax.barh(range(len(sorted_df)), sorted_df['importance'], color=colors)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_df['importance'])):
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)
    
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['feature'])
    ax.set_xlabel('Importance Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature importance to: {output_path}")


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    output_path: str,
    title: str = "Actual vs Predicted Popularity",
    figsize: tuple = (10, 8)
):
    """
    Plot actual vs predicted values scatter plot.
    
    Args:
        y_actual: Actual target values
        y_predicted: Predicted values
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot with density coloring
    from scipy.stats import gaussian_kde
    
    # Calculate point density for coloring
    xy = np.vstack([y_actual, y_predicted])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        y_actual_sorted = y_actual[idx]
        y_predicted_sorted = y_predicted[idx]
        z_sorted = z[idx]
        
        scatter = ax.scatter(
            y_actual_sorted, y_predicted_sorted, 
            c=z_sorted, s=10, alpha=0.5, cmap='viridis'
        )
        plt.colorbar(scatter, ax=ax, label='Density')
    except:
        # Fallback without density coloring
        ax.scatter(y_actual, y_predicted, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Metrics text
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.2f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Actual Popularity')
    ax.set_ylabel('Predicted Popularity')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved actual vs predicted to: {output_path}")


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    output_path: str,
    title: str = "Feature Distributions",
    figsize: tuple = (15, 12)
):
    """
    Plot distributions of all features.
    
    Args:
        df: DataFrame with features
        features: List of feature columns
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if feature in df.columns:
            data = pd.to_numeric(df[feature], errors='coerce').dropna()
            axes[i].hist(data, bins=50, edgecolor='black', alpha=0.7)
            axes[i].set_title(feature, fontsize=10)
            axes[i].set_xlabel('')
            
            # Add stats
            mean_val = data.mean()
            std_val = data.std()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'μ={mean_val:.2f}')
            axes[i].legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature distributions to: {output_path}")


def plot_popularity_distribution(
    df: pd.DataFrame,
    output_path: str,
    target_col: str = 'track.popularity',
    figsize: tuple = (10, 6)
):
    """
    Plot distribution of target popularity.
    
    Args:
        df: DataFrame with target column
        output_path: Path to save figure
        target_col: Target column name
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    data = pd.to_numeric(df[target_col], errors='coerce').dropna()
    
    ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(data.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {data.mean():.1f}')
    ax.axvline(data.median(), color='green', linestyle='--', lw=2, label=f'Median: {data.median():.1f}')
    
    ax.set_xlabel('Popularity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Track Popularity', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add stats box
    stats_text = f'N = {len(data):,}\nMean = {data.mean():.1f}\nStd = {data.std():.1f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved popularity distribution to: {output_path}")


def generate_all_plots(
    results_dir: str,
    output_dir: Optional[str] = None
):
    """
    Generate all plots from results directory.
    
    Args:
        results_dir: Path to results directory with CSVs/JSONs
        output_dir: Path to save plots (default: same as results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Correlation matrix
    corr_matrix_path = os.path.join(results_dir, 'correlation_matrix.csv')
    if os.path.exists(corr_matrix_path):
        corr_matrix = pd.read_csv(corr_matrix_path, index_col=0)
        plot_correlation_heatmap(
            corr_matrix,
            os.path.join(output_dir, 'correlation_heatmap.png')
        )
    
    # 2. Correlation bars
    corr_path = os.path.join(results_dir, 'correlations.csv')
    if os.path.exists(corr_path):
        correlations = pd.read_csv(corr_path, index_col=0).squeeze()
        plot_correlation_bars(
            correlations,
            os.path.join(output_dir, 'correlation_bars.png')
        )
    
    # 3. Feature importance
    importance_path = os.path.join(results_dir, 'best_model_feature_importance.csv')
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
        plot_feature_importance(
            importance_df,
            os.path.join(output_dir, 'feature_importance.png')
        )
    
    # 4. Sample data distributions
    sample_path = os.path.join(results_dir, 'sample_data.csv')
    if os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        
        # Feature distributions
        from track_features_lookup import AUDIO_FEATURES
        available_features = [f for f in AUDIO_FEATURES if f in sample_df.columns]
        if available_features:
            plot_feature_distributions(
                sample_df,
                available_features,
                os.path.join(output_dir, 'feature_distributions.png')
            )
        
        # Popularity distribution
        if 'track.popularity' in sample_df.columns:
            plot_popularity_distribution(
                sample_df,
                os.path.join(output_dir, 'popularity_distribution.png')
            )
    
    print("\n" + "="*60)
    print("  VISUALIZATION COMPLETE")
    print("="*60)
    print(f"All plots saved to: {output_dir}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Result Visualizations')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Path to save plots (default: same as results)')
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.results_dir):
        args.results_dir = os.path.join(base_dir, args.results_dir)
    
    if args.output_dir and not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
    
    generate_all_plots(args.results_dir, args.output_dir)

