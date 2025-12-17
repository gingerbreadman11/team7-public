# -*- coding: utf-8 -*-
"""
Release Year Analysis Plots
===========================
Analyze typicality evolution by song release year (extracted from ISRC).

Multiple plot variations:
1. Typicality evolution by release year
2. Age vs popularity analysis
3. Typicality distribution shifts
4. Optimal distinctiveness over time
5. Feature evolution over time

No titles, PDF only, large axis labels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter1d
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

# Color palettes
YEAR_CMAP = plt.cm.viridis
BLUE_GREEN = ['#084081', '#0868ac', '#2b8cbe', '#4eb3d3', '#7bccc4', '#a8ddb5', '#ccebc5']


def extract_year(isrc):
    """Extract release year from ISRC code."""
    try:
        isrc_str = str(isrc)
        if len(isrc_str) >= 7 and isrc_str[5:7].isdigit():
            year_code = int(isrc_str[5:7])
            year = 2000 + year_code if year_code < 50 else 1900 + year_code
            # Filter out clearly wrong years
            if 1960 <= year <= 2023:
                return year
    except:
        pass
    return None


def load_all_data_with_year(results_path: str) -> pd.DataFrame:
    """Load all period data and add release year column."""
    dfs = []
    for period_name in PERIOD_NAMES:
        period_dir = os.path.join(results_path, f"period_{period_name}")
        typ_path = os.path.join(period_dir, 'sample_with_typicality.csv')
        if os.path.exists(typ_path):
            df = pd.read_csv(typ_path)
            df['period'] = period_name
            dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Extract release year
    combined['release_year'] = combined['isrc'].apply(extract_year)
    combined = combined.dropna(subset=['release_year', 'typicality_cosine_norm', 'track.popularity'])
    combined['release_year'] = combined['release_year'].astype(int)
    combined['track.popularity'] = pd.to_numeric(combined['track.popularity'], errors='coerce')
    
    print(f"Loaded {len(combined):,} songs with valid release year")
    print(f"Year range: {combined['release_year'].min()} to {combined['release_year'].max()}")
    
    return combined


# =============================================================================
# 1. TYPICALITY EVOLUTION BY RELEASE YEAR
# =============================================================================

def plot_typicality_by_year_line(df: pd.DataFrame, output_dir: str):
    """Line plot of mean typicality by release year."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter to years with enough data
    year_counts = df['release_year'].value_counts()
    valid_years = year_counts[year_counts >= 50].index
    df_valid = df[df['release_year'].isin(valid_years)]
    
    # Group by year
    yearly = df_valid.groupby('release_year')['typicality_cosine_norm'].agg(['mean', 'std', 'count'])
    yearly['sem'] = yearly['std'] / np.sqrt(yearly['count'])
    
    years = yearly.index.values
    means = yearly['mean'].values
    sems = yearly['sem'].values * 1.96  # 95% CI
    
    # Plot
    ax.plot(years, means, 'o-', color='#2874a6', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(years, means - sems, means + sems, color='#2874a6', alpha=0.2)
    
    ax.set_xlabel('Release Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Typicality (± 95% CI)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_by_year_line.pdf'))
    plt.close()
    print("Saved: typicality_by_year_line.pdf")


def plot_typicality_by_year_bar(df: pd.DataFrame, output_dir: str):
    """Bar plot of mean typicality by release year."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Filter years 2010-2020 for cleaner view
    df_recent = df[(df['release_year'] >= 2010) & (df['release_year'] <= 2020)]
    
    yearly = df_recent.groupby('release_year')['typicality_cosine_norm'].agg(['mean', 'std', 'count'])
    yearly['sem'] = yearly['std'] / np.sqrt(yearly['count'])
    
    years = yearly.index.values
    means = yearly['mean'].values
    sems = yearly['sem'].values * 1.96
    
    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(years)))
    
    bars = ax.bar(years, means, yerr=sems, color=colors, edgecolor='white', linewidth=1.5, capsize=4)
    
    # Add count labels
    for i, (year, count) in enumerate(zip(years, yearly['count'].values)):
        ax.text(year, means[i] + sems[i] + 0.01, f'n={count:,}', ha='center', fontsize=TICK_SIZE*0.7, rotation=45)
    
    ax.set_xlabel('Release Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_xticks(years)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_by_year_bar.pdf'))
    plt.close()
    print("Saved: typicality_by_year_bar.pdf")


def plot_typicality_by_year_smooth(df: pd.DataFrame, output_dir: str):
    """Smooth curve of typicality evolution."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter to years with enough data
    year_counts = df['release_year'].value_counts()
    valid_years = year_counts[year_counts >= 30].index
    df_valid = df[df['release_year'].isin(valid_years)]
    
    yearly = df_valid.groupby('release_year')['typicality_cosine_norm'].agg(['mean', 'std', 'count'])
    yearly['sem'] = yearly['std'] / np.sqrt(yearly['count'])
    yearly = yearly.sort_index()
    
    years = yearly.index.values
    means = yearly['mean'].values
    sems = yearly['sem'].values * 1.96
    
    # Smooth
    means_smooth = gaussian_filter1d(means, sigma=1.5)
    
    ax.plot(years, means_smooth, '-', color='#1a5276', linewidth=3)
    ax.fill_between(years, means_smooth - sems, means_smooth + sems, color='#3498db', alpha=0.25)
    ax.scatter(years, means, c='#1a5276', s=50, zorder=5, edgecolors='white', linewidth=1)
    
    ax.set_xlabel('Release Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_by_year_smooth.pdf'))
    plt.close()
    print("Saved: typicality_by_year_smooth.pdf")


# =============================================================================
# 2. TYPICALITY DISTRIBUTION SHIFTS
# =============================================================================

def plot_typicality_violin_by_decade(df: pd.DataFrame, output_dir: str):
    """Violin plot comparing typicality distributions by decade."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create decade bins
    df_copy = df.copy()
    df_copy['decade'] = (df_copy['release_year'] // 5) * 5  # 5-year bins
    df_copy = df_copy[df_copy['decade'] >= 2000]
    
    # Filter to bins with enough data
    decade_counts = df_copy['decade'].value_counts()
    valid_decades = decade_counts[decade_counts >= 100].index
    df_copy = df_copy[df_copy['decade'].isin(valid_decades)]
    
    decades = sorted(df_copy['decade'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(decades)))
    
    parts = ax.violinplot([df_copy[df_copy['decade'] == d]['typicality_cosine_norm'].values 
                           for d in decades], positions=range(len(decades)), showmeans=True, showmedians=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    
    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels([f"{d}-{d+4}" for d in decades], fontsize=TICK_SIZE)
    ax.set_xlabel('Release Period', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Typicality Distribution', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_violin_by_period.pdf'))
    plt.close()
    print("Saved: typicality_violin_by_period.pdf")


def plot_typicality_kde_by_year(df: pd.DataFrame, output_dir: str):
    """KDE plots showing distribution shift over years."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select representative years
    years_to_plot = [2010, 2013, 2016, 2018, 2019, 2020]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(years_to_plot)))
    
    for year, color in zip(years_to_plot, colors):
        data = df[df['release_year'] == year]['typicality_cosine_norm']
        if len(data) >= 50:
            data.plot.kde(ax=ax, color=color, linewidth=2.5, label=f'{year} (n={len(data):,})')
    
    ax.set_xlabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Density', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper left')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_kde_by_year.pdf'))
    plt.close()
    print("Saved: typicality_kde_by_year.pdf")


def plot_typicality_boxplot_by_year(df: pd.DataFrame, output_dir: str):
    """Boxplot of typicality by year (2010-2020)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    df_recent = df[(df['release_year'] >= 2010) & (df['release_year'] <= 2020)]
    years = sorted(df_recent['release_year'].unique())
    
    data_by_year = [df_recent[df_recent['release_year'] == y]['typicality_cosine_norm'].values for y in years]
    
    bp = ax.boxplot(data_by_year, positions=range(len(years)), widths=0.6, patch_artist=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(years)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=TICK_SIZE)
    ax.set_xlabel('Release Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_boxplot_by_year.pdf'))
    plt.close()
    print("Saved: typicality_boxplot_by_year.pdf")


# =============================================================================
# 3. AGE VS POPULARITY ANALYSIS
# =============================================================================

def plot_typicality_popularity_by_era(df: pd.DataFrame, output_dir: str):
    """Compare typicality-popularity relationship for different eras."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define eras
    eras = [
        (2010, 2015, 'Early 2010s', '#1a5276'),
        (2016, 2018, 'Late 2010s', '#27ae60'),
        (2019, 2020, '2019-2020', '#c0392b'),
    ]
    
    for start, end, label, color in eras:
        df_era = df[(df['release_year'] >= start) & (df['release_year'] <= end)]
        if len(df_era) < 100:
            continue
        
        x = df_era['typicality_cosine_norm'].values
        y = df_era['track.popularity'].values
        
        # Quadratic fit
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_line = np.linspace(0.1, 0.9, 100)
        
        ax.plot(x_line, p(x_line), color=color, linewidth=3, label=f'{label} (n={len(df_era):,})')
    
    ax.set_xlabel('Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Predicted Popularity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='best')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'typicality_popularity_by_era.pdf'))
    plt.close()
    print("Saved: typicality_popularity_by_era.pdf")


def plot_popularity_by_release_year(df: pd.DataFrame, output_dir: str):
    """Mean popularity by release year."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df_recent = df[(df['release_year'] >= 2000) & (df['release_year'] <= 2020)]
    
    yearly = df_recent.groupby('release_year')['track.popularity'].agg(['mean', 'std', 'count'])
    yearly['sem'] = yearly['std'] / np.sqrt(yearly['count'])
    yearly = yearly[yearly['count'] >= 30]
    
    years = yearly.index.values
    means = yearly['mean'].values
    sems = yearly['sem'].values * 1.96
    
    ax.plot(years, means, 'o-', color='#8e44ad', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(years, means - sems, means + sems, color='#8e44ad', alpha=0.2)
    
    ax.set_xlabel('Release Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Popularity (± 95% CI)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'popularity_by_release_year.pdf'))
    plt.close()
    print("Saved: popularity_by_release_year.pdf")


# =============================================================================
# 4. OPTIMAL DISTINCTIVENESS OVER TIME
# =============================================================================

def plot_optimal_typicality_by_year(df: pd.DataFrame, output_dir: str):
    """Track optimal typicality point (quadratic vertex) by year."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    years_to_analyze = range(2010, 2021)
    optimal_points = []
    
    for year in years_to_analyze:
        df_year = df[df['release_year'] == year]
        if len(df_year) < 100:
            continue
        
        x = df_year['typicality_cosine_norm'].values
        y = df_year['track.popularity'].values
        
        # Quadratic fit
        z = np.polyfit(x, y, 2)
        
        # Vertex: -b / 2a
        if z[0] != 0:
            optimal_x = -z[1] / (2 * z[0])
            if 0 < optimal_x < 1:  # Valid range
                optimal_points.append((year, optimal_x, len(df_year)))
    
    if optimal_points:
        years, optima, counts = zip(*optimal_points)
        
        ax.plot(years, optima, 'o-', color='#2c3e50', linewidth=2.5, markersize=10, 
                markerfacecolor='#3498db', markeredgewidth=2)
        
        # Add reference line at 0.5
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.text(max(years) + 0.3, 0.5, 'Midpoint', fontsize=TICK_SIZE, va='center', color='gray')
    
    ax.set_xlabel('Release Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Optimal Typicality', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_typicality_by_year.pdf'))
    plt.close()
    print("Saved: optimal_typicality_by_year.pdf")


def plot_optimal_typicality_with_ci(df: pd.DataFrame, output_dir: str):
    """Optimal typicality by year with bootstrap confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    years_to_analyze = range(2010, 2021)
    results = []
    
    for year in years_to_analyze:
        df_year = df[df['release_year'] == year]
        if len(df_year) < 100:
            continue
        
        x = df_year['typicality_cosine_norm'].values
        y = df_year['track.popularity'].values
        
        # Bootstrap for CI
        optima_boot = []
        for _ in range(100):
            idx = np.random.choice(len(x), size=len(x), replace=True)
            x_boot, y_boot = x[idx], y[idx]
            z = np.polyfit(x_boot, y_boot, 2)
            if z[0] != 0:
                opt = -z[1] / (2 * z[0])
                if 0 < opt < 1:
                    optima_boot.append(opt)
        
        if len(optima_boot) > 10:
            results.append({
                'year': year,
                'optimal': np.mean(optima_boot),
                'ci_low': np.percentile(optima_boot, 2.5),
                'ci_high': np.percentile(optima_boot, 97.5),
                'n': len(df_year)
            })
    
    if results:
        df_res = pd.DataFrame(results)
        
        ax.plot(df_res['year'], df_res['optimal'], 'o-', color='#16a085', linewidth=2.5, 
                markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(df_res['year'], df_res['ci_low'], df_res['ci_high'], color='#16a085', alpha=0.2)
        
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Release Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Optimal Typicality (± 95% CI)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_typicality_with_ci.pdf'))
    plt.close()
    print("Saved: optimal_typicality_with_ci.pdf")


# =============================================================================
# 5. FEATURE EVOLUTION OVER TIME
# =============================================================================

def plot_feature_evolution(df: pd.DataFrame, output_dir: str):
    """How individual audio features have changed over time."""
    features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    df_recent = df[(df['release_year'] >= 2010) & (df['release_year'] <= 2020)]
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        if feature not in df_recent.columns:
            continue
        
        yearly = df_recent.groupby('release_year')[feature].agg(['mean', 'std', 'count'])
        yearly['sem'] = yearly['std'] / np.sqrt(yearly['count'])
        
        years = yearly.index.values
        means = yearly['mean'].values
        sems = yearly['sem'].values * 1.96
        
        color = plt.cm.Set2(i / len(features))
        ax.plot(years, means, 'o-', color=color, linewidth=2, markersize=6)
        ax.fill_between(years, means - sems, means + sems, color=color, alpha=0.2)
        
        ax.set_xlabel('Release Year' if i >= 3 else '', fontsize=TICK_SIZE)
        ax.set_ylabel(feature.capitalize(), fontsize=TICK_SIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_SIZE * 0.8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_evolution_6panel.pdf'))
    plt.close()
    print("Saved: feature_evolution_6panel.pdf")


def plot_feature_evolution_normalized(df: pd.DataFrame, output_dir: str):
    """All features normalized and overlaid."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'tempo']
    colors = plt.cm.tab10(np.linspace(0, 1, len(features)))
    
    df_recent = df[(df['release_year'] >= 2010) & (df['release_year'] <= 2020)]
    
    for feature, color in zip(features, colors):
        if feature not in df_recent.columns:
            continue
        
        yearly = df_recent.groupby('release_year')[feature].mean()
        
        # Normalize to 0-1 for comparison
        yearly_norm = (yearly - yearly.min()) / (yearly.max() - yearly.min())
        
        ax.plot(yearly_norm.index, yearly_norm.values, 'o-', color=color, linewidth=2.5, 
                markersize=6, label=feature.capitalize())
    
    ax.set_xlabel('Release Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Normalized Feature Value', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_evolution_normalized.pdf'))
    plt.close()
    print("Saved: feature_evolution_normalized.pdf")


# =============================================================================
# 6. COMBINED HEATMAPS BY ERA
# =============================================================================

def plot_heatmap_by_era(df: pd.DataFrame, output_dir: str):
    """Side-by-side heatmaps for different eras."""
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.25)
    
    eras = [
        (2010, 2015, 'Early 2010s'),
        (2016, 2018, 'Late 2010s'),
        (2019, 2020, '2019-2020'),
    ]
    
    colors_gb = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#084081']
    cmap_gb = mcolors.LinearSegmentedColormap.from_list('green_blue', colors_gb)
    
    for i, (start, end, label) in enumerate(eras):
        ax = fig.add_subplot(gs[0, i])
        
        df_era = df[(df['release_year'] >= start) & (df['release_year'] <= end)]
        
        if len(df_era) < 50:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            continue
        
        x = df_era['typicality_cosine_norm'].values
        y = df_era['track.popularity'].values
        
        hb = ax.hexbin(x, y, gridsize=30, cmap=cmap_gb, mincnt=1)
        
        # Quadratic fit
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_line = np.linspace(0.1, 0.9, 100)
        ax.plot(x_line, p(x_line), color='#c0392b', linewidth=2.5)
        
        # Stats
        corr, _ = scipy_stats.pearsonr(x, y)
        ax.text(0.02, 0.98, f"n={len(df_era):,}\nr={corr:.3f}", transform=ax.transAxes,
                fontsize=TICK_SIZE * 0.9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.text(0.98, 0.02, label, transform=ax.transAxes, fontsize=TICK_SIZE,
                ha='right', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Typicality', fontsize=TICK_SIZE)
        ax.set_ylabel('Popularity' if i == 0 else '', fontsize=TICK_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE * 0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_by_era.pdf'))
    plt.close()
    print("Saved: heatmap_by_era.pdf")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def save_summary_statistics(df: pd.DataFrame, output_dir: str):
    """Save summary statistics to CSV."""
    
    # By year
    yearly_stats = df.groupby('release_year').agg({
        'typicality_cosine_norm': ['mean', 'std', 'median', 'count'],
        'track.popularity': ['mean', 'std', 'median']
    }).round(4)
    yearly_stats.columns = ['_'.join(col) for col in yearly_stats.columns]
    yearly_stats.to_csv(os.path.join(output_dir, 'stats_by_release_year.csv'))
    
    print("Saved: stats_by_release_year.csv")


def main():
    """Generate all release year analysis plots."""
    print("="*60)
    print("  RELEASE YEAR ANALYSIS PLOTS")
    print("="*60)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, 'results_temporal')
    output_dir = os.path.join(results_path, 'plots_release_year')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data with release years...")
    df = load_all_data_with_year(results_path)
    
    # Generate all plots
    print("\n" + "-"*40)
    print("1. Typicality Evolution by Year")
    plot_typicality_by_year_line(df, output_dir)
    plot_typicality_by_year_bar(df, output_dir)
    plot_typicality_by_year_smooth(df, output_dir)
    
    print("\n" + "-"*40)
    print("2. Typicality Distribution Shifts")
    plot_typicality_violin_by_decade(df, output_dir)
    plot_typicality_kde_by_year(df, output_dir)
    plot_typicality_boxplot_by_year(df, output_dir)
    
    print("\n" + "-"*40)
    print("3. Age vs Popularity")
    plot_typicality_popularity_by_era(df, output_dir)
    plot_popularity_by_release_year(df, output_dir)
    
    print("\n" + "-"*40)
    print("4. Optimal Distinctiveness Over Time")
    plot_optimal_typicality_by_year(df, output_dir)
    plot_optimal_typicality_with_ci(df, output_dir)
    
    print("\n" + "-"*40)
    print("5. Feature Evolution")
    plot_feature_evolution(df, output_dir)
    plot_feature_evolution_normalized(df, output_dir)
    
    print("\n" + "-"*40)
    print("6. Heatmaps by Era")
    plot_heatmap_by_era(df, output_dir)
    
    print("\n" + "-"*40)
    print("7. Summary Statistics")
    save_summary_statistics(df, output_dir)
    
    print("\n" + "="*60)
    print("  ALL PLOTS COMPLETE")
    print("="*60)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

