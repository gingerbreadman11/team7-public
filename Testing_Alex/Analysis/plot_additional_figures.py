# -*- coding: utf-8 -*-
"""
Additional Figures for Paper
============================
- Typicality distribution (illustration)
- Feature importance INCLUDING typicality
- Multiple variations

No titles, PDF only, large axis labels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Font sizes (56% larger than base)
SCALE = 1.5625
LABEL_SIZE = int(12 * SCALE)
TICK_SIZE = int(10 * SCALE)
LEGEND_SIZE = int(9 * SCALE)

# Period names
PERIOD_NAMES = [
    "2020-03_2020-08", "2020-09_2021-02", "2021-03_2021-08",
    "2021-09_2022-02", "2022-03_2022-08", "2022-09_2023-01",
]

# Audio features
AUDIO_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
]

# Colors
PURPLE_START = '#67129A'
PURPLE_MID = '#9A2578'
RED_END = '#D20D52'


def load_all_data(results_path: str) -> pd.DataFrame:
    """Load all period data."""
    dfs = []
    for period_name in PERIOD_NAMES:
        period_dir = os.path.join(results_path, f"period_{period_name}")
        typ_path = os.path.join(period_dir, 'sample_with_typicality.csv')
        if os.path.exists(typ_path):
            df = pd.read_csv(typ_path)
            dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['track.popularity'] = pd.to_numeric(combined['track.popularity'], errors='coerce')
    print(f"Loaded {len(combined):,} total rows")
    return combined


# =============================================================================
# TYPICALITY DISTRIBUTION PLOTS
# =============================================================================

def plot_typicality_distribution_simple(df: pd.DataFrame, output_dir: str):
    """Simple histogram of typicality."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    data = df['typicality_cosine_norm'].dropna()
    
    ax.hist(data, bins=50, color='#3498db', edgecolor='white', alpha=0.8, linewidth=0.5)
    
    # Add mean and median lines
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color=RED_END, linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color=PURPLE_START, linestyle='--', linewidth=2.5, label=f'Median: {median_val:.3f}')
    
    ax.set_xlabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper left')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_distribution_histogram.pdf'))
    plt.close()
    print("Saved: typicality_distribution_histogram.pdf")


def plot_typicality_distribution_kde(df: pd.DataFrame, output_dir: str):
    """KDE plot of typicality distribution."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    data = df['typicality_cosine_norm'].dropna()
    
    # KDE
    data.plot.kde(ax=ax, color='#2c3e50', linewidth=3)
    
    # Fill under curve
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 1, 200)
    ax.fill_between(x_range, kde(x_range), color='#3498db', alpha=0.3)
    
    # Stats
    ax.axvline(data.mean(), color=RED_END, linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
    ax.axvline(data.median(), color=PURPLE_START, linestyle=':', linewidth=2, label=f'Median: {data.median():.3f}')
    
    ax.set_xlabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Density', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_distribution_kde.pdf'))
    plt.close()
    print("Saved: typicality_distribution_kde.pdf")


def plot_typicality_vs_popularity_joint(df: pd.DataFrame, output_dir: str):
    """Joint distribution plot showing both marginals."""
    # Subsample for speed
    df_sample = df.dropna(subset=['typicality_cosine_norm', 'track.popularity']).sample(n=min(50000, len(df)), random_state=42)
    
    g = sns.JointGrid(data=df_sample, x='typicality_cosine_norm', y='track.popularity', height=9)
    
    # Main plot - hexbin
    colors_gb = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#084081']
    cmap_gb = mcolors.LinearSegmentedColormap.from_list('green_blue', colors_gb)
    
    g.ax_joint.hexbin(df_sample['typicality_cosine_norm'], df_sample['track.popularity'], 
                      gridsize=40, cmap=cmap_gb, mincnt=1)
    
    # Marginal histograms
    g.ax_marg_x.hist(df_sample['typicality_cosine_norm'], bins=40, color='#3498db', alpha=0.7, edgecolor='white')
    g.ax_marg_y.hist(df_sample['track.popularity'], bins=40, color='#3498db', alpha=0.7, 
                     edgecolor='white', orientation='horizontal')
    
    g.ax_joint.set_xlabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    g.ax_joint.set_ylabel('Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    g.ax_joint.tick_params(axis='both', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_popularity_joint.pdf'))
    plt.close()
    print("Saved: typicality_popularity_joint.pdf")


# =============================================================================
# FEATURE IMPORTANCE WITH TYPICALITY
# =============================================================================

def compute_feature_importance_with_typicality(df: pd.DataFrame):
    """Train model and get feature importances including typicality."""
    
    # Features including typicality
    features_with_typ = AUDIO_FEATURES + ['typicality_cosine_norm']
    
    # Prepare data
    df_clean = df.dropna(subset=features_with_typ + ['track.popularity'])
    
    X = df_clean[features_with_typ].values
    y = df_clean['track.popularity'].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    rf_importance = rf.feature_importances_
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_score = gb.score(X_test, y_test)
    gb_importance = gb.feature_importances_
    
    print(f"Random Forest R²: {rf_score:.4f}")
    print(f"Gradient Boosting R²: {gb_score:.4f}")
    
    return {
        'features': features_with_typ,
        'rf_importance': rf_importance,
        'gb_importance': gb_importance,
        'rf_score': rf_score,
        'gb_score': gb_score
    }


def plot_feature_importance_horizontal(importance_data: dict, output_dir: str):
    """Horizontal bar chart of feature importance (including typicality)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features = importance_data['features']
    importance = importance_data['rf_importance']
    
    # Sort by importance
    idx = np.argsort(importance)
    features_sorted = [features[i] for i in idx]
    importance_sorted = importance[idx]
    
    # Highlight typicality
    colors = ['#c0392b' if 'typicality' in f else '#3498db' for f in features_sorted]
    
    bars = ax.barh(range(len(features_sorted)), importance_sorted, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars, importance_sorted)):
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=TICK_SIZE * 0.9)
    
    # Clean feature names
    clean_names = [f.replace('typicality_cosine_norm', 'TYPICALITY').replace('_', ' ').title() for f in features_sorted]
    
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(clean_names, fontsize=TICK_SIZE)
    ax.set_xlabel('Feature Importance', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    
    # Add R² annotation
    ax.text(0.98, 0.02, f"R² = {importance_data['rf_score']:.4f}", transform=ax.transAxes,
            fontsize=TICK_SIZE, ha='right', va='bottom', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_with_typicality_rf.pdf'))
    plt.close()
    print("Saved: feature_importance_with_typicality_rf.pdf")


def plot_feature_importance_comparison(importance_data: dict, output_dir: str):
    """Compare RF vs GB importance side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    features = importance_data['features']
    
    for ax, (imp, score, name) in zip(axes, [
        (importance_data['rf_importance'], importance_data['rf_score'], 'Random Forest'),
        (importance_data['gb_importance'], importance_data['gb_score'], 'Gradient Boosting')
    ]):
        idx = np.argsort(imp)
        features_sorted = [features[i] for i in idx]
        imp_sorted = imp[idx]
        
        colors = ['#c0392b' if 'typicality' in f else '#2ecc71' for f in features_sorted]
        
        ax.barh(range(len(features_sorted)), imp_sorted, color=colors, edgecolor='white')
        
        clean_names = [f.replace('typicality_cosine_norm', 'TYPICALITY').replace('_', ' ').title() for f in features_sorted]
        ax.set_yticks(range(len(features_sorted)))
        ax.set_yticklabels(clean_names, fontsize=TICK_SIZE * 0.9)
        ax.set_xlabel('Importance', fontsize=TICK_SIZE, fontweight='bold')
        ax.tick_params(axis='x', labelsize=TICK_SIZE * 0.9)
        
        ax.text(0.98, 0.02, f"{name}\nR² = {score:.4f}", transform=ax.transAxes,
                fontsize=TICK_SIZE * 0.9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.pdf'))
    plt.close()
    print("Saved: feature_importance_comparison.pdf")


def plot_feature_importance_clean(importance_data: dict, output_dir: str):
    """Clean, minimal feature importance plot."""
    fig, ax = plt.subplots(figsize=(8, 9))
    
    features = importance_data['features']
    importance = importance_data['rf_importance']
    
    # Sort
    idx = np.argsort(importance)[::-1]  # Descending
    features_sorted = [features[i] for i in idx]
    importance_sorted = importance[idx]
    
    # Colors - gradient with typicality highlighted
    n = len(features_sorted)
    colors = []
    for f in features_sorted:
        if 'typicality' in f:
            colors.append('#e74c3c')  # Red for typicality
        else:
            colors.append('#34495e')  # Dark gray for others
    
    y_pos = range(len(features_sorted))
    ax.barh(y_pos, importance_sorted, color=colors, height=0.7)
    
    # Clean names
    clean_names = []
    for f in features_sorted:
        if 'typicality' in f:
            clean_names.append('Typicality')
        else:
            clean_names.append(f.replace('_', ' ').capitalize())
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=TICK_SIZE)
    ax.set_xlabel('Feature Importance', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Typicality'),
        Patch(facecolor='#34495e', label='Audio Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=LEGEND_SIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_clean.pdf'))
    plt.close()
    print("Saved: feature_importance_clean.pdf")


def plot_feature_importance_poster_style(importance_data: dict, output_dir: str):
    """Poster-style feature importance with both models."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = importance_data['features']
    rf_imp = importance_data['rf_importance']
    gb_imp = importance_data['gb_importance']
    
    # Average importance
    avg_imp = (rf_imp + gb_imp) / 2
    
    # Sort by average
    idx = np.argsort(avg_imp)
    features_sorted = [features[i] for i in idx]
    rf_sorted = rf_imp[idx]
    gb_sorted = gb_imp[idx]
    
    y_pos = np.arange(len(features_sorted))
    width = 0.35
    
    # Highlight typicality
    rf_colors = ['#c0392b' if 'typicality' in f else '#3498db' for f in features_sorted]
    gb_colors = ['#e74c3c' if 'typicality' in f else '#5dade2' for f in features_sorted]
    
    bars1 = ax.barh(y_pos - width/2, rf_sorted, width, label='Random Forest', color=rf_colors, alpha=0.9)
    bars2 = ax.barh(y_pos + width/2, gb_sorted, width, label='Gradient Boosting', color=gb_colors, alpha=0.7)
    
    # Clean names
    clean_names = [f.replace('typicality_cosine_norm', '→ TYPICALITY').replace('_', ' ').title() for f in features_sorted]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=TICK_SIZE)
    ax.set_xlabel('Feature Importance', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_poster_style.pdf'))
    plt.close()
    print("Saved: feature_importance_poster_style.pdf")


# =============================================================================
# ADDITIONAL DISTRIBUTION PLOTS
# =============================================================================

def plot_popularity_distribution(df: pd.DataFrame, output_dir: str):
    """Distribution of popularity scores."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    data = df['track.popularity'].dropna()
    
    ax.hist(data, bins=50, color='#9b59b6', edgecolor='white', alpha=0.8)
    ax.axvline(data.mean(), color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Mean: {data.mean():.1f}')
    ax.axvline(data.median(), color='#2c3e50', linestyle='--', linewidth=2.5, label=f'Median: {data.median():.1f}')
    
    ax.set_xlabel('Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'popularity_distribution.pdf'))
    plt.close()
    print("Saved: popularity_distribution.pdf")


def plot_features_correlation_with_popularity(df: pd.DataFrame, output_dir: str):
    """Correlation of each feature (including typicality) with popularity."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features = AUDIO_FEATURES + ['typicality_cosine_norm']
    correlations = []
    
    for f in features:
        if f in df.columns:
            corr = df[f].corr(df['track.popularity'])
            correlations.append((f, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]))
    
    features_sorted = [c[0] for c in correlations]
    corr_sorted = [c[1] for c in correlations]
    
    # Colors
    colors = ['#c0392b' if 'typicality' in f else ('#27ae60' if c > 0 else '#e74c3c') 
              for f, c in correlations]
    
    ax.barh(range(len(features_sorted)), corr_sorted, color=colors, edgecolor='white')
    
    # Clean names
    clean_names = [f.replace('typicality_cosine_norm', 'TYPICALITY').replace('_', ' ').title() for f in features_sorted]
    
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(clean_names, fontsize=TICK_SIZE)
    ax.set_xlabel('Correlation with Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlations_with_popularity.pdf'))
    plt.close()
    print("Saved: feature_correlations_with_popularity.pdf")


def main():
    """Generate all additional figures."""
    print("="*60)
    print("  ADDITIONAL FIGURES FOR PAPER")
    print("="*60)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, 'results_temporal')
    output_dir = os.path.join(results_path, 'plots_additional')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    df = load_all_data(results_path)
    
    # Typicality distributions
    print("\n" + "-"*40)
    print("1. Typicality Distribution Plots")
    plot_typicality_distribution_simple(df, output_dir)
    plot_typicality_distribution_kde(df, output_dir)
    plot_typicality_vs_popularity_joint(df, output_dir)
    
    # Feature importance with typicality
    print("\n" + "-"*40)
    print("2. Feature Importance (with Typicality)")
    importance_data = compute_feature_importance_with_typicality(df)
    plot_feature_importance_horizontal(importance_data, output_dir)
    plot_feature_importance_comparison(importance_data, output_dir)
    plot_feature_importance_clean(importance_data, output_dir)
    plot_feature_importance_poster_style(importance_data, output_dir)
    
    # Additional distributions
    print("\n" + "-"*40)
    print("3. Additional Distributions")
    plot_popularity_distribution(df, output_dir)
    plot_features_correlation_with_popularity(df, output_dir)
    
    print("\n" + "="*60)
    print("  ALL FIGURES COMPLETE")
    print("="*60)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

