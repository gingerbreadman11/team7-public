# -*- coding: utf-8 -*-
"""
Typicality Explainer Diagram
============================
Visual explanation of how typicality is computed.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
pbStart = "#67129A"  # deep purple
pbMid = "#9A2578"    # magenta-pink
pbEnd = "#D20D52"    # rich red
GRAY = "#555555"
LIGHT_GRAY = "#CCCCCC"

def draw_typicality_explainer():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # === PANEL 1: Feature Space ===
    ax1 = axes[0]
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 10.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('1. Audio Feature Space', fontsize=13, fontweight='bold', pad=10)
    
    # Draw axes
    ax1.annotate('', xy=(10, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
    ax1.annotate('', xy=(0, 10), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
    ax1.text(10.3, -0.3, 'Energy', fontsize=10, ha='left')
    ax1.text(-0.3, 10.3, 'Danceability', fontsize=10, ha='right', rotation=90, va='bottom')
    
    # Scatter of tracks (cluster)
    np.random.seed(42)
    n_tracks = 60
    cluster_x = np.random.normal(5, 1.5, n_tracks)
    cluster_y = np.random.normal(5, 1.5, n_tracks)
    ax1.scatter(cluster_x, cluster_y, c=LIGHT_GRAY, s=40, alpha=0.6, edgecolors='white', linewidths=0.5)
    
    # Highlight focal track
    focal_x, focal_y = 5.2, 5.5
    ax1.scatter([focal_x], [focal_y], c=pbEnd, s=120, zorder=10, edgecolors='black', linewidths=1.5)
    ax1.text(focal_x + 0.4, focal_y + 0.4, 'Focal\nTrack', fontsize=9, color=pbEnd, fontweight='bold')
    
    # Caption
    ax1.text(5, -1.5, 'Each track is a point in\n8-dimensional audio space', 
             ha='center', fontsize=9, style='italic', color=GRAY)
    
    # === PANEL 2: k-Nearest Neighbors ===
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(-0.5, 10.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('2. Find k Nearest Neighbors', fontsize=13, fontweight='bold', pad=10)
    
    # Same scatter
    ax2.scatter(cluster_x, cluster_y, c=LIGHT_GRAY, s=40, alpha=0.4, edgecolors='white', linewidths=0.5)
    
    # Focal track
    ax2.scatter([focal_x], [focal_y], c=pbEnd, s=120, zorder=10, edgecolors='black', linewidths=1.5)
    
    # Draw circle for neighborhood
    circle = Circle((focal_x, focal_y), 2.0, fill=False, color=pbMid, linewidth=2, linestyle='--')
    ax2.add_patch(circle)
    
    # Highlight neighbors (k closest)
    distances = np.sqrt((cluster_x - focal_x)**2 + (cluster_y - focal_y)**2)
    k = 8
    neighbor_idx = np.argsort(distances)[:k]
    ax2.scatter(cluster_x[neighbor_idx], cluster_y[neighbor_idx], 
                c=pbMid, s=60, zorder=5, edgecolors='black', linewidths=1)
    
    # Draw lines to neighbors
    for idx in neighbor_idx:
        ax2.plot([focal_x, cluster_x[idx]], [focal_y, cluster_y[idx]], 
                 color=pbMid, alpha=0.5, linewidth=1, zorder=3)
    
    ax2.text(focal_x, focal_y - 2.8, f'k = {k} neighbors', ha='center', fontsize=10, color=pbMid, fontweight='bold')
    
    # Caption
    ax2.text(5, -1.5, 'Measure distance to\nk nearest neighbors', 
             ha='center', fontsize=9, style='italic', color=GRAY)
    
    # === PANEL 3: Typicality Score ===
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('3. Compute Typicality', fontsize=13, fontweight='bold', pad=10)
    
    # Formula box
    formula_box = FancyBboxPatch((0.5, 5.5), 9, 2.5, boxstyle="round,pad=0.3",
                                  facecolor='white', edgecolor=pbStart, linewidth=2)
    ax3.add_patch(formula_box)
    ax3.text(5, 7.2, 'Typicality = mean( similarity to neighbors )', 
             ha='center', va='center', fontsize=11, fontweight='bold', color=pbStart)
    ax3.text(5, 6.0, 'similarity = exp( −distance )', 
             ha='center', va='center', fontsize=10, color=GRAY, style='italic')
    
    # Two example tracks
    # High typicality (typical)
    ax3.add_patch(Circle((2.5, 3.5), 0.4, facecolor=pbMid, edgecolor='black'))
    ax3.text(2.5, 2.5, 'TYPICAL', ha='center', fontsize=10, fontweight='bold', color=pbMid)
    ax3.text(2.5, 1.8, 'Close to neighbors\n→ High similarity\n→ Typicality ≈ 0.9', 
             ha='center', fontsize=9, color=GRAY)
    
    # Low typicality (distinctive)
    ax3.add_patch(Circle((7.5, 3.5), 0.4, facecolor=pbEnd, edgecolor='black'))
    ax3.text(7.5, 2.5, 'DISTINCTIVE', ha='center', fontsize=10, fontweight='bold', color=pbEnd)
    ax3.text(7.5, 1.8, 'Far from neighbors\n→ Low similarity\n→ Typicality ≈ 0.3', 
             ha='center', fontsize=9, color=GRAY)
    
    # Arrow between
    ax3.annotate('', xy=(6.5, 3.5), xytext=(3.5, 3.5),
                arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.5))
    
    plt.tight_layout()
    
    # Save
    out_png = os.path.join(OUTPUT_DIR, 'typicality_explainer.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'typicality_explainer.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    draw_typicality_explainer()

