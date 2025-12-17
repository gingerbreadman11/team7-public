# -*- coding: utf-8 -*-
"""
Plot Comparison Module
======================
Generates poster-ready figures for Model Comparison and Feature Importance.

Figures:
1. Audio Feature Importance (Horizontal Bar Chart)
2. Model Performance Comparison (Grouped Bar Chart)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, 'results_test', 'model_comparison_metrics.csv')
# Prefer large metrics if present
METRICS_PATH_LARGE = os.path.join(BASE_DIR, 'results_large', 'model_comparison_metrics.csv')
IMPORTANCE_PATH = os.path.join(BASE_DIR, 'results_test', 'audio_feature_importance.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom Colors (Poster Theme)
pbStart = "#67129A" # deep purple
pbMid = "#9A2578"   # magenta-pink
pbEnd = "#D20D52"   # rich red

def plot_feature_importance(df):
    """Plot horizontal bar chart of feature importance."""
    plt.figure(figsize=(6, 5))
    
    # Sort for plot
    df = df.sort_values('Importance', ascending=True)
    
    plt.barh(df['Feature'], df['Importance'], color=pbMid, alpha=0.9, edgecolor='black')
    
    plt.title('Audio Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    save_path_png = os.path.join(OUTPUT_DIR, 'audio_feature_importance_rf.png')
    save_path_pdf = os.path.join(OUTPUT_DIR, 'audio_feature_importance_rf.pdf')
    
    plt.savefig(save_path_png, dpi=300)
    plt.savefig(save_path_pdf)
    print(f"Saved: {save_path_png}")

def plot_model_comparison(df):
    """Plot grouped bar charts for R² and MAE comparing splits (random vs forecast)."""
    sns.set_style("whitegrid")
    palette = {'Random': pbStart, 'Forecast': pbEnd}

    # R2 panel
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=df,
        x='Model',
        y='R2',
        hue='Split',
        palette=palette,
        edgecolor='black',
        alpha=0.9
    )
    plt.title('Model Performance (R²)', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=10)
    plt.xlabel('Algorithm', fontsize=10)
    plt.legend(title='Split', loc='upper right')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    plt.tight_layout()
    save_path_png = os.path.join(OUTPUT_DIR, 'model_performance_r2.png')
    save_path_pdf = os.path.join(OUTPUT_DIR, 'model_performance_r2.pdf')
    plt.savefig(save_path_png, dpi=300)
    plt.savefig(save_path_pdf)
    print(f"Saved: {save_path_png}")

    # MAE panel
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=df,
        x='Model',
        y='MAE',
        hue='Split',
        palette=palette,
        edgecolor='black',
        alpha=0.9
    )
    plt.title('Model Performance (MAE)', fontsize=12, fontweight='bold')
    plt.ylabel('MAE (popularity points)', fontsize=10)
    plt.xlabel('Algorithm', fontsize=10)
    plt.legend(title='Split', loc='upper right')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
    plt.tight_layout()
    save_path_png = os.path.join(OUTPUT_DIR, 'model_performance_mae.png')
    save_path_pdf = os.path.join(OUTPUT_DIR, 'model_performance_mae.pdf')
    plt.savefig(save_path_png, dpi=300)
    plt.savefig(save_path_pdf)
    print(f"Saved: {save_path_png}")

def main():
    print("="*60)
    print("  GENERATING POSTER FIGURES")
    print("="*60)
    
    # 1. Feature Importance
    if os.path.exists(IMPORTANCE_PATH):
        print("Plotting feature importance...")
        df_imp = pd.read_csv(IMPORTANCE_PATH)
        plot_feature_importance(df_imp)
    else:
        print(f"WARNING: Feature importance file not found at {IMPORTANCE_PATH}")
        
    # 2. Model Comparison
    # Prefer large metrics if available
    metrics_path = METRICS_PATH_LARGE if os.path.exists(METRICS_PATH_LARGE) else METRICS_PATH
    if os.path.exists(metrics_path):
        print(f"Plotting model comparison from {metrics_path} ...")
        df_met = pd.read_csv(metrics_path)
        if 'Split' not in df_met.columns:
            print("WARNING: Split column not found; plotting as-is.")
            df_met['Split'] = 'Random'
        plot_model_comparison(df_met)
    else:
        print(f"WARNING: Metrics file not found at {metrics_path}")

if __name__ == "__main__":
    main()

