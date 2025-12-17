# -*- coding: utf-8 -*-
"""
Full Analysis on Feature-Deduped Data (~60k tracks)
====================================================
Re-runs:
1. Popularity distribution (histogram + density)
2. RF vs GBM vs Bagging model comparison
3. Feature importance

Usage:
    python3 run_all_dedup_analysis.py
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'results_feature_dedup', 'sample_feature_dedup.csv')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
OUT_DIR = os.path.join(BASE_DIR, 'results_feature_dedup')
os.makedirs(FIG_DIR, exist_ok=True)

# Features
AUDIO_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
    'loudness', 'speechiness', 'tempo', 'valence'
]
TARGET = 'track.popularity'

# Colors
pbStart = '#67129A'  # purple
pbMid = '#9A2578'    # magenta
pbEnd = '#D20D52'    # red


def plot_popularity_distribution(df):
    """Plot popularity histogram and density."""
    print("\n[1/3] Plotting popularity distribution...")
    pop = df[TARGET].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(pop, bins=50, color=pbMid, edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Popularity Score', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Popularity Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3, linestyle='--')
    
    # Density
    sns.kdeplot(pop, ax=axes[1], fill=True, color=pbEnd, alpha=0.7)
    axes[1].set_xlabel('Popularity Score', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Popularity Density', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'popularity_distribution_dedup.png')
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"  Saved: {path}")
    
    # Stats
    print(f"  Mean: {pop.mean():.1f}, Median: {pop.median():.1f}, Std: {pop.std():.1f}")


def train_models(df):
    """Train RF, GBM, Bagging and compare."""
    print("\n[2/3] Training models (RF, GBM, Bagging)...")
    
    # Prepare data
    df_clean = df.dropna(subset=AUDIO_FEATURES + [TARGET]).copy()
    X = df_clean[AUDIO_FEATURES].values
    y = df_clean[TARGET].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Bagging': BaggingRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    }
    
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {'R2': r2, 'MAE': mae}
        print(f"    R²={r2:.4f}, MAE={mae:.2f}")
        
        # Feature importance (RF and GBM have feature_importances_)
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = dict(zip(AUDIO_FEATURES, model.feature_importances_))
    
    # Save metrics
    metrics_path = os.path.join(OUT_DIR, 'model_metrics_dedup.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")
    
    return results, feature_importances


def plot_model_comparison(results):
    """Plot R² and MAE comparison bar chart."""
    print("\n[3/3] Plotting model comparison...")
    
    models = list(results.keys())
    r2_scores = [results[m]['R2'] for m in models]
    mae_scores = [results[m]['MAE'] for m in models]
    
    colors = [pbStart, pbMid, pbEnd]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² bars
    bars1 = axes[0].bar(models, r2_scores, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Model Comparison: R²', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, max(r2_scores) * 1.2)
    axes[0].grid(alpha=0.3, linestyle='--', axis='y')
    for bar, val in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # MAE bars
    bars2 = axes[1].bar(models, mae_scores, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Model Comparison: MAE', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, max(mae_scores) * 1.2)
    axes[1].grid(alpha=0.3, linestyle='--', axis='y')
    for bar, val in zip(bars2, mae_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                     f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'model_comparison_dedup.png')
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(feature_importances):
    """Plot feature importance for RF."""
    if 'Random Forest' not in feature_importances:
        return
    
    print("\n  Plotting feature importance (RF)...")
    imp = feature_importances['Random Forest']
    
    # Sort by importance
    sorted_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Horizontal bar
    colors = plt.cm.RdPu(np.linspace(0.4, 0.9, len(features)))[::-1]
    bars = ax.barh(features[::-1], values[::-1], color=colors, edgecolor='black', alpha=0.85)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', axis='x')
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'feature_importance_dedup.png')
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("="*60)
    print("  FULL ANALYSIS ON DEDUPED DATA")
    print("="*60)
    
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found!")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df):,} tracks (deduped)")
    
    # 1. Popularity distribution
    plot_popularity_distribution(df)
    
    # 2. Train models
    results, feature_importances = train_models(df)
    
    # 3. Plot model comparison
    plot_model_comparison(results)
    
    # 4. Feature importance
    plot_feature_importance(feature_importances)
    
    print("\n" + "="*60)
    print("  ALL DONE!")
    print("="*60)


if __name__ == "__main__":
    main()

