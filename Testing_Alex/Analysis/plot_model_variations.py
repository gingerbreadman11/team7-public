import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import seaborn as sns

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
metrics_path = os.path.join(BASE_DIR, 'results_feature_dedup', 'model_metrics_dedup.json')
out_dir = os.path.join(BASE_DIR, 'figures', 'model_variations')
os.makedirs(out_dir, exist_ok=True)

# Custom Colors from User
pbStart = '#67129A' # deep purple
pbMid = '#9A2578'   # magenta-pink
pbEnd = '#D20D52'   # rich red

c_r2 = pbStart   # Purple (for R2 - "Good")
c_mae = pbEnd    # Red (for MAE - "Error")
c_bg = '#F9F9F9' # Very light grey background for style

# Load Data
try:
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded metrics from {metrics_path}")
except FileNotFoundError:
    print("Warning: Metrics file not found, using fallback values.")
    data = {
        'Random Forest': {'R2': 0.0524, 'MAE': 12.35},
        'Gradient Boosting': {'R2': 0.0587, 'MAE': 12.30},
        'Bagging': {'R2': 0.0378, 'MAE': 12.46}
    }

df = pd.DataFrame(data).T.sort_values('R2', ascending=True) # Sort by performance
models = df.index
r2 = df['R2']
mae = df['MAE']

# Global Style Settings
plt.rcParams['font.family'] = 'sans-serif'
# lighter grid
plt.rcParams['grid.alpha'] = 0.3

# ==========================================
# PLOT 1: The "Butterfly" (Diverging Bars)
# ==========================================
# Best for comparing "Good" vs "Bad" metrics side-by-side
fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(models))

# Left bars (MAE, negative direction for visual)
# Using pbEnd for Error
ax.barh(y_pos, -mae, color=c_mae, alpha=0.9, height=0.6, label='MAE (Error)')

# Right bars (R2)
# Scaling R2 up visually so it's visible next to MAE (since 0.05 is tiny vs 12)
r2_scale = 200 
# Using pbStart for Score
ax.barh(y_pos, r2 * r2_scale, color=c_r2, alpha=0.9, height=0.6, label='R² (Score)')

# Annotations
for i, (m, r, err) in enumerate(zip(models, r2, mae)):
    # MAE Text
    ax.text(-err - 0.5, i, f'{err:.2f}', va='center', ha='right', color=c_mae, fontweight='bold')
    # R2 Text
    ax.text((r * r2_scale) + 0.5, i, f'{r:.3f}', va='center', ha='left', color=c_r2, fontweight='bold')
    # Model Name in center
    ax.text(0, i, f'  {m}  ', va='center', ha='center', fontweight='bold', 
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax.set_yticks([])
ax.set_xticks([]) # Hide numeric axis since we labelled bars
ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
ax.set_title('Model Performance: Accuracy (R²) vs Error (MAE)', fontweight='bold', color='#333333')
ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), frameon=False)

sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'style1_butterfly.png'), dpi=300)
plt.savefig(os.path.join(out_dir, 'style1_butterfly.pdf'))
plt.close()


# ==========================================
# PLOT 2: Dual-Axis Combo (Bar + Line)
# ==========================================
# Classic scientific look
fig, ax1 = plt.subplots(figsize=(8, 6))

x = np.arange(len(models))
width = 0.5

# Bar for R2
bars = ax1.bar(x, r2, width, color=c_r2, alpha=0.8, label='R² Score')
ax1.set_ylabel('R² Score (Higher is Better)', color=c_r2, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=c_r2)
ax1.set_ylim(0, max(r2)*1.4) # Give headroom

# Line/Point for MAE
ax2 = ax1.twinx()
# Use pbMid for the line to contrast slightly or pbEnd? pbEnd fits the schema better.
line = ax2.plot(x, mae, color=c_mae, marker='o', markersize=12, linewidth=3, linestyle='-', label='MAE')
ax2.set_ylabel('MAE (Lower is Better)', color=c_mae, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=c_mae)
# Zoom in on MAE differences
mae_min, mae_max = min(mae), max(mae)
margin = (mae_max - mae_min) * 0.5 if mae_max != mae_min else 1.0
ax2.set_ylim(mae_min - margin, mae_max + margin) 

# Labels
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight='bold', fontsize=10)
ax1.set_title('Model Comparison: Predictive Power', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'style2_dual_axis.png'), dpi=300)
plt.savefig(os.path.join(out_dir, 'style2_dual_axis.pdf'))
plt.close()


# ==========================================
# PLOT 3: The "Trade-off" Scatter
# ==========================================
# Best for showing the "efficient frontier"
fig, ax = plt.subplots(figsize=(8, 6))

# Plot points
# Use pbMid for the face color to blend
ax.scatter(mae, r2, s=600, c=pbMid, alpha=0.7, edgecolors=pbStart, linewidth=2) 

# Add labels
for i, txt in enumerate(models):
    ax.annotate(txt, (mae[i], r2[i]), xytext=(0, 12), textcoords='offset points', 
                ha='center', fontweight='bold', fontsize=11, color=pbStart)
    ax.annotate(f"R²: {r2[i]:.3f}\nMAE: {mae[i]:.2f}", (mae[i], r2[i]), 
                xytext=(0, -30), textcoords='offset points', ha='center', fontsize=9, color='#555')

ax.set_xlabel('Error (MAE) ← Lower is Better', fontsize=11, fontweight='bold', color=c_mae)
ax.set_ylabel('Accuracy (R²) → Higher is Better', fontsize=11, fontweight='bold', color=c_r2)
ax.set_title('Model Efficiency Frontier', fontsize=14, fontweight='bold')

# Invert X axis so "better" (lower error) is to the right
plt.gca().invert_xaxis()
ax.grid(True, linestyle='--', alpha=0.4)
sns.despine()

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'style3_scatter.png'), dpi=300)
plt.savefig(os.path.join(out_dir, 'style3_scatter.pdf'))
plt.close()


# ==========================================
# PLOT 4: Modern Lollipop / Dumbbell
# ==========================================
# Very clean, minimalistic
fig, ax = plt.subplots(figsize=(9, 5))

# Normalize metrics to 0-1 scale for direct visual comparison on same axis
# We want to show "Strength", so high R2 is good (keep as is), low MAE is good (invert)
r2_norm = (r2 - r2.min()) / (r2.max() - r2.min()) if r2.max() != r2.min() else r2/r2
# Invert MAE so 1 is best (lowest MAE)
mae_norm = 1 - (mae - mae.min()) / (mae.max() - mae.min()) if mae.max() != mae.min() else mae/mae

y = np.arange(len(models))

# Draw lines
ax.hlines(y=y, xmin=0, xmax=r2_norm, color=c_r2, alpha=0.4, linewidth=3)
ax.hlines(y=y, xmin=0, xmax=mae_norm, color=c_mae, alpha=0.4, linewidth=3)

# Draw dots
ax.plot(r2_norm, y, 'o', color=c_r2, markersize=15, label='R² Strength')
ax.plot(mae_norm, y, 'o', color=c_mae, markersize=15, label='MAE Strength')

# Annotate actual values
for i, (r_n, m_n) in enumerate(zip(r2_norm, mae_norm)):
    # Shift text slightly
    ax.text(r_n + 0.04, i, f"R² {r2[i]:.3f}", va='center', color=c_r2, fontweight='bold', fontsize=10)
    ax.text(m_n + 0.04, i, f"MAE {mae[i]:.2f}", va='center', color=c_mae, fontweight='bold', fontsize=10)

ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=12, fontweight='bold')
ax.set_xlim(-0.1, 1.4) # Extra room for text
ax.set_xticks([])
sns.despine(bottom=True, left=True)
ax.set_title('Relative Performance (Normalized Scores)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'style4_lollipop.png'), dpi=300)
plt.savefig(os.path.join(out_dir, 'style4_lollipop.pdf'))
plt.close()


# ==========================================
# PLOT 5: The "Scorecard" Tile
# ==========================================
# Info-graphic style
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Model Performance Scorecards', fontsize=16, fontweight='bold', y=1.05, color='#333')

for i, (model, ax) in enumerate(zip(models, axes)):
    # Background
    ax.set_facecolor(c_bg)
    
    # Title
    ax.text(0.5, 0.85, model, ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes, color='#444')
    
    # R2 Metric
    ax.text(0.5, 0.65, 'R² Score', ha='center', fontsize=9, color=c_r2, transform=ax.transAxes)
    ax.text(0.5, 0.50, f"{r2[i]:.4f}", ha='center', fontsize=22, fontweight='bold', color=c_r2, transform=ax.transAxes)
    
    # Divider - use pbMid for subtle divider
    ax.plot([0.3, 0.7], [0.45, 0.45], color='#DDD', transform=ax.transAxes, linewidth=1)
    
    # MAE Metric
    ax.text(0.5, 0.30, 'MAE Error', ha='center', fontsize=9, color=c_mae, transform=ax.transAxes)
    ax.text(0.5, 0.15, f"{mae[i]:.2f}", ha='center', fontsize=18, fontweight='bold', color=c_mae, transform=ax.transAxes)
    
    # Highlight the winner (GBM is usually best R2)
    # Since sorted by R2 ascending, last one is best
    is_best = (i == len(models) - 1)
    
    if is_best:
        for spine in ax.spines.values():
            spine.set_edgecolor(pbStart)
            spine.set_linewidth(3)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'style5_scorecard.png'), dpi=300)
plt.savefig(os.path.join(out_dir, 'style5_scorecard.pdf'))
plt.close()

print(f"Done! Check {out_dir}/ for 5 variations.")



