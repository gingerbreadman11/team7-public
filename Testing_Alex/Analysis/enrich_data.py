# -*- coding: utf-8 -*-
"""
Enrich Data Module
==================
Adds release dates and calculates typicality scores for the sample dataset.

Process:
1. Load sample_data.csv (the 100k sample).
2. Load all track features and album features CSVs.
3. Merge Tracks -> Albums to get release_date.
4. Calculate 'typicality' as cosine similarity to the year's centroid.
5. Save enriched dataset.
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, 'results_test', 'sample_data.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'results_test', 'sample_data_enriched.csv')

TRACK_FEATURES_DIR = os.path.join(DATA_DIR, 'album and track files', 'tracks', 'csvs for paper sample')
ALBUM_FEATURES_DIR = os.path.join(DATA_DIR, 'album and track files', 'albums', 'album files')

# Audio features for typicality calculation
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

def load_all_data(directory, pattern):
    """Load and concatenate all CSVs matching pattern in directory."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        print(f"WARNING: No files found matching {pattern} in {directory}")
        return pd.DataFrame()
    
    print(f"Loading {len(files)} files from {os.path.basename(directory)}...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {os.path.basename(f)}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def extract_year(date_str):
    """Extract year from release_date string (YYYY-MM-DD or YYYY)."""
    if pd.isna(date_str):
        return None
    try:
        # Handle '2023-01-01' or '2023'
        return int(str(date_str)[:4])
    except:
        return None

def main():
    print("="*60)
    print("  ENRICHING DATA: RELEASE DATES & TYPICALITY")
    print("="*60)

    # 1. Load Sample Data
    if not os.path.exists(SAMPLE_DATA_PATH):
        print(f"ERROR: Sample data not found at {SAMPLE_DATA_PATH}")
        return

    print(f"Loading sample data: {SAMPLE_DATA_PATH}")
    df_sample = pd.read_csv(SAMPLE_DATA_PATH)
    print(f"  Rows: {len(df_sample):,}")

    # 2. Load Metadata (Tracks & Albums)
    # We need to map track.id -> album.id -> release_date
    print("\nLoading metadata for date matching...")
    
    # Load track features to get album.id
    df_tracks_meta = load_all_data(TRACK_FEATURES_DIR, 'track_features_*.csv')
    if 'track.id' in df_tracks_meta.columns and 'album.id' in df_tracks_meta.columns:
        track_album_map = df_tracks_meta[['track.id', 'album.id']].drop_duplicates('track.id')
    else:
        print("Error: track features missing required columns")
        return

    # Load album features to get release_date
    df_albums_meta = load_all_data(ALBUM_FEATURES_DIR, 'album_features_*.csv')
    if 'id' in df_albums_meta.columns and 'release_date' in df_albums_meta.columns:
        # Rename 'id' to 'album.id' for merging if needed, but usually it's 'id' in album file
        # Check standard column name in album files
        album_date_map = df_albums_meta[['id', 'release_date']].rename(columns={'id': 'album.id'}).drop_duplicates('album.id')
    else:
        print("Error: album features missing required columns")
        return

    # 3. Join Release Dates
    print("\nMapping release dates...")
    
    # Join Sample -> Track Meta (to get album.id)
    # Note: df_sample might already have album.id if it came from the features lookup? 
    # Let's check sample columns. If not, join.
    # The lookup script only kept specific columns.
    
    if 'album.id' not in df_sample.columns:
        df_sample = pd.merge(df_sample, track_album_map, on='track.id', how='left')
    
    # Join -> Album Meta (to get release_date)
    df_sample = pd.merge(df_sample, album_date_map, on='album.id', how='left')
    
    # Extract Year
    df_sample['release_year'] = df_sample['release_date'].apply(extract_year)
    
    # Fill missing years with median or drop? For this analysis, let's drop tracks without year
    # or forward fill / assume based on context? Better to drop for clean train/test split.
    missing_year = df_sample['release_year'].isna().sum()
    print(f"  Tracks with missing release year: {missing_year:,} (will be excluded from modeling)")
    
    # 4. Calculate Typicality
    print("\nCalculating typicality scores...")
    
    # We use the FULL track metadata to calculate centroids (more robust than just sample)
    # Ensure audio features are numeric
    for col in AUDIO_FEATURES:
        df_tracks_meta[col] = pd.to_numeric(df_tracks_meta[col], errors='coerce')
    
    # Filter valid tracks
    valid_tracks = df_tracks_meta.dropna(subset=AUDIO_FEATURES).copy()
    
    # Get years for all tracks (we need to join album info to the full track meta too)
    print("  Preparing full dataset for centroid calculation...")
    if 'album.id' not in valid_tracks.columns:
        valid_tracks = pd.merge(valid_tracks, track_album_map, on='track.id', how='left')
    
    valid_tracks = pd.merge(valid_tracks, album_date_map, on='album.id', how='left')
    valid_tracks['release_year'] = valid_tracks['release_date'].apply(extract_year)
    
    valid_tracks = valid_tracks.dropna(subset=['release_year'])
    valid_tracks['release_year'] = valid_tracks['release_year'].astype(int)
    
    # Calculate Centroids per Year
    centroids = valid_tracks.groupby('release_year')[AUDIO_FEATURES].mean()
    print(f"  Calculated centroids for {len(centroids)} years: {centroids.index.min()} - {centroids.index.max()}")
    
    # Function to calc similarity
    def calculate_similarity(row):
        year = row['release_year']
        if pd.isna(year) or year not in centroids.index:
            return np.nan
        
        track_vec = row[AUDIO_FEATURES].values.reshape(1, -1)
        centroid_vec = centroids.loc[year].values.reshape(1, -1)
        
        # Cosine similarity returns [[val]]
        return cosine_similarity(track_vec, centroid_vec)[0][0]

    # Apply to sample
    # Ensure sample features are numeric
    for col in AUDIO_FEATURES:
        df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce')
    
    print("  Computing cosine similarity for sample tracks...")
    df_sample['typicality'] = df_sample.apply(calculate_similarity, axis=1)
    df_sample['typicality_squared'] = df_sample['typicality'] ** 2
    
    # Final cleanup
    df_final = df_sample.dropna(subset=['release_year', 'typicality', 'track.popularity']).copy()
    df_final['release_year'] = df_final['release_year'].astype(int)
    
    print(f"\nFinal enriched dataset: {len(df_final):,} rows")
    print(f"Sample typicality: Mean={df_final['typicality'].mean():.4f}, Std={df_final['typicality'].std():.4f}")
    
    # 5. Save
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

