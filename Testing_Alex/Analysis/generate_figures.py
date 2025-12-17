# -*- coding: utf-8 -*-
"""
Generate figures for poster:
1) Data pipeline schematic (PDF)
2) Popularity histogram ("pyramid") (PDF)
3) Popularity density (KDE) (PDF)

Color scheme matches poster gradient:
  pbStart: #67129A (deep purple)
  pbMid:   #9A2578 (magenta-pink)
  pbEnd:   #D20D52 (rich red)
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "results_test", "sample_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Colors
pbStart = "#67129A"
pbMid = "#9A2578"
pbEnd = "#D20D52"

# Matplotlib defaults
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def draw_pipeline():
    """Create data pipeline schematic."""
    plt.figure(figsize=(8.0, 4.5))
    ax = plt.gca()

    def box(x, y, w, h, text, color):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2",
            edgecolor="black", facecolor=color, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=11)

    # Boxes
    box(0.2, 0.55, 2.0, 0.9, "Daily Spotify\nplaylists (2018–2023)", pbStart)
    box(2.8, 0.55, 2.3, 0.9, "Unique tracks +\naudio features\n(danceability, energy, …)", pbMid)
    box(5.6, 1.0, 2.3, 0.45, "Spotify popularity\nscore (0–100)", pbEnd)
    box(5.6, 0.55, 2.3, 0.35, "Top-200 daily charts\n(by country)", pbEnd)

    # Arrows with slight curvature for clarity
    arrowprops = dict(arrowstyle="->", linewidth=1.6, color="black")
    ax.annotate("", xy=(2.8, 1.0), xytext=(2.2, 1.0), arrowprops=arrowprops)
    ax.annotate("", xy=(5.6, 1.2), xytext=(5.1, 1.05), arrowprops=arrowprops)
    ax.annotate("", xy=(5.6, 0.75), xytext=(5.1, 0.85), arrowprops=arrowprops)

    ax.axis("off")
    ax.set_xlim(-0.2, 8.3)
    ax.set_ylim(0.2, 2.1)
    plt.tight_layout(pad=0.8)
    out_pdf = os.path.join(OUT_DIR, "data_pipeline.pdf")
    out_png = os.path.join(OUT_DIR, "data_pipeline.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


def load_popularity():
    df = pd.read_csv(DATA_PATH)
    pop = pd.to_numeric(df["track.popularity"], errors="coerce").dropna()
    return pop


def plot_histogram(pop):
    plt.figure(figsize=(6, 4))
    plt.hist(pop, bins=30, color=pbMid, edgecolor=pbStart, alpha=0.9)
    plt.xlabel("Popularity (0–100)")
    plt.ylabel("Count of tracks")
    plt.title("Popularity Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_pdf = os.path.join(OUT_DIR, "popularity_pyramid.pdf")
    out_png = os.path.join(OUT_DIR, "popularity_pyramid.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


def plot_density(pop):
    plt.figure(figsize=(6, 4))
    pop = pop[(pop >= 0) & (pop <= 100)]
    sns = __import__("seaborn")
    sns.kdeplot(pop, fill=True, color=pbEnd, alpha=0.6, linewidth=1.5)
    plt.xlabel("Popularity (0–100)")
    plt.ylabel("Density")
    plt.title("Popularity Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_pdf = os.path.join(OUT_DIR, "popularity_density.pdf")
    out_png = os.path.join(OUT_DIR, "popularity_density.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


def main():
    pop = load_popularity()
    draw_pipeline()
    plot_histogram(pop)
    plot_density(pop)


if __name__ == "__main__":
    main()

