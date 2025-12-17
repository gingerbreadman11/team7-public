# -*- coding: utf-8 -*-
"""
Optimal Distinctiveness using Cosine Typicality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "results_large", "sample_with_typicality_cosine.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pbStart = "#67129A"
pbMid = "#9A2578"
pbEnd = "#D20D52"

AUDIO_CONTROLS = [
    "danceability",
    "energy",
    "loudness",
    "valence",
    "instrumentalness",
    "acousticness",
    "speechiness",
    "tempo",
]


def plot_curve(df, n_bins=20):
    df = df.dropna(subset=["typicality_cosine_norm", "track.popularity"]).copy()
    df["bin"] = pd.qcut(df["typicality_cosine_norm"], q=n_bins, labels=False, duplicates="drop")
    binned = df.groupby("bin").agg(
        typicality_mean=("typicality_cosine_norm", "mean"),
        pop_mean=("track.popularity", "mean"),
        pop_std=("track.popularity", "std"),
        count=("track.popularity", "count"),
    ).reset_index()
    binned["se"] = binned["pop_std"] / np.sqrt(binned["count"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        binned["typicality_mean"],
        binned["pop_mean"] - 1.96 * binned["se"],
        binned["pop_mean"] + 1.96 * binned["se"],
        alpha=0.25,
        color=pbMid,
    )
    ax.plot(binned["typicality_mean"], binned["pop_mean"], "o-", color=pbMid, linewidth=3, markersize=6)
    ax.set_xlabel("Typicality (cosine, 0=distinctive, 1=typical)")
    ax.set_ylabel("Mean Popularity (0–100)")
    ax.set_title("Typicality vs Popularity (Cosine, k=100)")
    ax.grid(alpha=0.3, linestyle="--")
    out_png = os.path.join(OUTPUT_DIR, "typicality_vs_popularity_cosine.png")
    out_pdf = os.path.join(OUTPUT_DIR, "typicality_vs_popularity_cosine.pdf")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_png}")
    return binned


def run_regression(df):
    df = df.dropna(subset=["typicality_cosine_norm", "typicality_cosine_norm_sq", "track.popularity"] + AUDIO_CONTROLS).copy()
    X = df[["typicality_cosine_norm", "typicality_cosine_norm_sq"] + AUDIO_CONTROLS]
    X = sm.add_constant(X)
    y = df["track.popularity"]
    model = sm.OLS(y, X).fit()
    print(model.summary())
    print("\n--- KEY COEFFICIENTS ---")
    print(f"typicality_cosine_norm: {model.params['typicality_cosine_norm']:.4f}, p={model.pvalues['typicality_cosine_norm']:.4g}")
    print(f"typicality_cosine_norm_sq: {model.params['typicality_cosine_norm_sq']:.4f}, p={model.pvalues['typicality_cosine_norm_sq']:.4g}")
    summary_path = os.path.join(BASE_DIR, "results_large", "regression_summary_cosine.txt")
    with open(summary_path, "w") as f:
        f.write(model.summary().as_text())
    print(f"Saved regression summary to: {summary_path}")
    return model


def main():
    print("=" * 60)
    print(" COSINE TYPICALITY ANALYSIS ")
    print("=" * 60)
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run compute_typicality_cosine.py first.")
        return
    df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(df):,}")

    print("\n--- Curve ---")
    binned = plot_curve(df, n_bins=20)
    print(binned[["typicality_mean", "pop_mean"]].round(3))

    print("\n--- Regression ---")
    run_regression(df)
    print("\nDone.")


if __name__ == "__main__":
    main()

