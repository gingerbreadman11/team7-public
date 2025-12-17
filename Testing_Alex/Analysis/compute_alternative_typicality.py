# -*- coding: utf-8 -*-
"""
Alternative Typicality Measures
===============================
Implements:
A) Popularity-weighted typicality: weight neighbors by their popularity
B) Temporal window kNN: typicality relative to songs from same time period

Uses cached data from temporal analysis.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Audio features
AUDIO_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
    'key', 'mode', 'time_signature'
]

PERIOD_NAMES = [
    "2020-03_2020-08", "2020-09_2021-02", "2021-03_2021-08",
    "2021-09_2022-02", "2022-03_2022-08", "2022-09_2023-01",
]


def extract_year(isrc):
    """Extract release year from ISRC code."""
    try:
        isrc_str = str(isrc)
        if len(isrc_str) >= 7 and isrc_str[5:7].isdigit():
            year_code = int(isrc_str[5:7])
            year = 2000 + year_code if year_code < 50 else 1900 + year_code
            if 1960 <= year <= 2023:
                return year
    except:
        pass
    return None


def load_all_data(results_path: str) -> pd.DataFrame:
    """Load all period data."""
    dfs = []
    for period_name in PERIOD_NAMES:
        period_dir = os.path.join(results_path, f"period_{period_name}")
        typ_path = os.path.join(period_dir, 'sample_with_typicality.csv')
        if os.path.exists(typ_path):
            df = pd.read_csv(typ_path)
            df['period'] = period_name
            dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['track.popularity'] = pd.to_numeric(combined['track.popularity'], errors='coerce')
    combined['release_year'] = combined['isrc'].apply(extract_year)
    
    # Clean
    combined = combined.dropna(subset=AUDIO_FEATURES + ['track.popularity', 'typicality_cosine_norm'])
    
    print(f"Loaded {len(combined):,} songs")
    return combined


# =============================================================================
# A) POPULARITY-WEIGHTED TYPICALITY
# =============================================================================

def compute_popularity_weighted_typicality(df: pd.DataFrame, k: int = 25) -> np.ndarray:
    """
    Compute typicality where each neighbor's contribution is weighted by its popularity.
    
    Instead of: typicality = mean(similarity_to_neighbors)
    We use:     typicality = sum(sim_i * pop_i) / sum(pop_i)
    
    This means: if your neighbors are popular songs, you're more "mainstream typical"
    """
    print(f"Computing popularity-weighted typicality (k={k})...")
    
    # Prepare features
    X = df[AUDIO_FEATURES].values.astype(np.float32)
    popularity = df['track.popularity'].values.astype(np.float32)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build kNN index
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df)), metric='cosine', algorithm='brute')
    nn.fit(X_scaled)
    
    # Query
    distances, indices = nn.kneighbors(X_scaled)
    
    # Exclude self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Convert distance to similarity
    similarities = 1 - distances  # cosine similarity
    
    # Compute popularity-weighted typicality
    n_samples = len(df)
    typicality_weighted = np.zeros(n_samples)
    
    for i in range(n_samples):
        neighbor_idx = indices[i]
        neighbor_pop = popularity[neighbor_idx]
        neighbor_sim = similarities[i]
        
        # Weighted average: sum(sim * pop) / sum(pop)
        # Add small epsilon to avoid division by zero
        total_pop = neighbor_pop.sum() + 1e-10
        typicality_weighted[i] = (neighbor_sim * neighbor_pop).sum() / total_pop
        
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1:,}/{n_samples:,}")
    
    # Normalize to 0-1
    typ_min, typ_max = typicality_weighted.min(), typicality_weighted.max()
    typicality_norm = (typicality_weighted - typ_min) / (typ_max - typ_min + 1e-10)
    
    print(f"  Range: [{typicality_norm.min():.4f}, {typicality_norm.max():.4f}]")
    print(f"  Mean: {typicality_norm.mean():.4f}")
    
    return typicality_norm


# =============================================================================
# B) TEMPORAL WINDOW KNN TYPICALITY
# =============================================================================

def compute_temporal_window_typicality(df: pd.DataFrame, window_years: int = 3, k: int = 25) -> np.ndarray:
    """
    Compute typicality relative to songs from a similar time period.
    
    For each song released in year Y:
    - Find neighbors only among songs from years [Y - window_years, Y]
    - This captures "how typical was this song for its era"
    
    Args:
        df: DataFrame with release_year column
        window_years: How many years back to look (e.g., 3 = same year + 2 prior years)
        k: Number of neighbors
    
    Returns:
        Array of temporal typicality scores
    """
    print(f"Computing temporal window typicality (window={window_years} years, k={k})...")
    
    # Filter to songs with valid release year
    df_valid = df.dropna(subset=['release_year']).copy()
    df_valid['release_year'] = df_valid['release_year'].astype(int)
    
    # Prepare features
    X = df_valid[AUDIO_FEATURES].values.astype(np.float32)
    years = df_valid['release_year'].values
    
    # Standardize globally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples = len(df_valid)
    typicality_temporal = np.full(n_samples, np.nan)
    
    # Get unique years
    unique_years = sorted(df_valid['release_year'].unique())
    print(f"  Processing {len(unique_years)} unique years...")
    
    for year in unique_years:
        # Define window: [year - window_years, year]
        year_mask = (years >= year - window_years) & (years <= year)
        window_indices = np.where(year_mask)[0]
        
        if len(window_indices) < k + 1:
            # Not enough songs in this window, skip
            continue
        
        # Songs from this specific year (what we're computing typicality for)
        target_mask = years == year
        target_indices = np.where(target_mask)[0]
        
        if len(target_indices) == 0:
            continue
        
        # Build kNN on window
        X_window = X_scaled[window_indices]
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(window_indices)), metric='cosine', algorithm='brute')
        nn.fit(X_window)
        
        # Query for songs from this year
        X_targets = X_scaled[target_indices]
        distances, _ = nn.kneighbors(X_targets)
        
        # Exclude self if present (first neighbor might be self)
        distances = distances[:, 1:] if distances.shape[1] > k else distances
        
        # Convert to similarity and average
        similarities = 1 - distances
        typ_scores = similarities.mean(axis=1)
        
        # Store
        typicality_temporal[target_indices] = typ_scores
        
        if year % 5 == 0:
            print(f"  Year {year}: {len(target_indices)} songs, window size: {len(window_indices)}")
    
    # Map back to original df indices
    # Create result array for full df
    result = np.full(len(df), np.nan)
    valid_mask = df['release_year'].notna()
    result[valid_mask] = typicality_temporal
    
    # Normalize non-NaN values
    valid = ~np.isnan(result)
    if valid.sum() > 0:
        result_valid = result[valid]
        result_min, result_max = result_valid.min(), result_valid.max()
        result[valid] = (result_valid - result_min) / (result_max - result_min + 1e-10)
    
    n_computed = valid.sum()
    print(f"  Computed for {n_computed:,}/{len(df):,} songs")
    if n_computed > 0:
        print(f"  Range: [{result[valid].min():.4f}, {result[valid].max():.4f}]")
        print(f"  Mean: {result[valid].mean():.4f}")
    
    return result


def compute_rolling_temporal_typicality(df: pd.DataFrame, window_years: int = 2, k: int = 25, 
                                        sample_size: int = 50000) -> np.ndarray:
    """
    Faster version: sample songs and compute temporal typicality.
    For each sampled song, compute typicality relative to songs from [year-window, year].
    """
    print(f"Computing rolling temporal typicality (sampled, window={window_years} years, k={k})...")
    
    # Filter and sample
    df_valid = df.dropna(subset=['release_year']).copy()
    df_valid['release_year'] = df_valid['release_year'].astype(int)
    
    # Focus on years with enough data
    year_counts = df_valid['release_year'].value_counts()
    good_years = year_counts[year_counts >= 100].index
    df_valid = df_valid[df_valid['release_year'].isin(good_years)]
    
    if len(df_valid) > sample_size:
        df_sample = df_valid.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_valid
    
    print(f"  Working with {len(df_sample):,} songs")
    
    # Prepare features
    X_all = df_valid[AUDIO_FEATURES].values.astype(np.float32)
    years_all = df_valid['release_year'].values
    idx_all = df_valid.index.values
    
    X_sample = df_sample[AUDIO_FEATURES].values.astype(np.float32)
    years_sample = df_sample['release_year'].values
    idx_sample = df_sample.index.values
    
    # Standardize
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    X_sample_scaled = scaler.transform(X_sample)
    
    # Create mapping from original index to position in X_all
    idx_to_pos = {idx: pos for pos, idx in enumerate(idx_all)}
    
    n_sample = len(df_sample)
    typicality_temporal = np.zeros(n_sample)
    
    for i in range(n_sample):
        year = years_sample[i]
        
        # Window mask on full data
        window_mask = (years_all >= year - window_years) & (years_all <= year)
        window_positions = np.where(window_mask)[0]
        
        if len(window_positions) < k + 1:
            typicality_temporal[i] = np.nan
            continue
        
        # Build kNN on window
        X_window = X_all_scaled[window_positions]
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(window_positions)), metric='cosine', algorithm='brute')
        nn.fit(X_window)
        
        # Query for this song
        distances, _ = nn.kneighbors(X_sample_scaled[i:i+1])
        distances = distances[0, 1:]  # Exclude self
        
        # Similarity
        similarities = 1 - distances
        typicality_temporal[i] = similarities.mean()
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{n_sample:,}")
    
    # Normalize
    valid = ~np.isnan(typicality_temporal)
    if valid.sum() > 0:
        typ_valid = typicality_temporal[valid]
        typ_min, typ_max = typ_valid.min(), typ_valid.max()
        typicality_temporal[valid] = (typ_valid - typ_min) / (typ_max - typ_min + 1e-10)
    
    # Create result for full df
    result = np.full(len(df), np.nan)
    result[idx_sample] = typicality_temporal
    
    n_computed = (~np.isnan(result)).sum()
    print(f"  Computed for {n_computed:,}/{len(df):,} songs")
    
    return result, df_sample.index


def main():
    """Compute alternative typicality measures and save."""
    print("="*60)
    print("  ALTERNATIVE TYPICALITY MEASURES")
    print("="*60)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, 'results_temporal')
    output_dir = os.path.join(results_path, 'alternative_typicality')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    df = load_all_data(results_path)
    
    # A) Popularity-weighted typicality
    print("\n" + "="*60)
    print("A) POPULARITY-WEIGHTED TYPICALITY")
    print("="*60)
    df['typicality_pop_weighted'] = compute_popularity_weighted_typicality(df, k=25)
    
    # B) Temporal window typicality (sampled for speed)
    print("\n" + "="*60)
    print("B) TEMPORAL WINDOW TYPICALITY")
    print("="*60)
    temporal_typ, sample_idx = compute_rolling_temporal_typicality(df, window_years=2, k=25, sample_size=50000)
    df['typicality_temporal'] = temporal_typ
    
    # Save full results
    output_path = os.path.join(output_dir, 'data_with_alt_typicality.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved full data to: {output_path}")
    
    # Save sampled subset (has all typicality measures)
    df_sample = df.loc[sample_idx].dropna(subset=['typicality_temporal'])
    sample_path = os.path.join(output_dir, 'sample_with_all_typicality.csv')
    df_sample.to_csv(sample_path, index=False)
    print(f"Saved sample ({len(df_sample):,} rows) to: {sample_path}")
    
    # Print correlations between typicality measures
    print("\n" + "="*60)
    print("CORRELATIONS BETWEEN TYPICALITY MEASURES")
    print("="*60)
    
    typ_cols = ['typicality_cosine_norm', 'typicality_pop_weighted', 'typicality_temporal']
    df_corr = df_sample[typ_cols + ['track.popularity']].corr()
    print(df_corr.round(4))
    
    # Save correlation matrix
    df_corr.to_csv(os.path.join(output_dir, 'typicality_correlations.csv'))
    
    print("\n" + "="*60)
    print("  COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

