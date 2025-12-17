# -*- coding: utf-8 -*-
"""
Data Exploration & Sampling Script
===================================
Goal: Understand the structure and quality of our data before analysis.

This script will:
1. Sample track features data
2. Sample daily playlist data
3. Show basic statistics
4. Visualize distributions
5. Check for missing values
6. Preview a potential join between datasets
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')

TRACK_FEATURES_DIR = os.path.join(DATA_DIR, 'album and track files', 'tracks', 'csvs for paper sample')
ALBUM_FEATURES_DIR = os.path.join(DATA_DIR, 'album and track files', 'albums', 'album files')
DAILY_DATA_DIR = os.path.join(DATA_DIR, 'Data_Spotify')

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, 'exploration_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ============================================================
# 1. EXPLORE TRACK FEATURES
# ============================================================
separator("1. TRACK FEATURES DATA")

# Load just the first track features file as a sample
track_files = sorted(glob.glob(os.path.join(TRACK_FEATURES_DIR, 'track_features_*.csv')))
print(f"Found {len(track_files)} track feature files:")
for f in track_files:
    print(f"  - {os.path.basename(f)}")

if track_files:
    # Load first file
    print(f"\nLoading sample: {os.path.basename(track_files[0])}")
    df_tracks = pd.read_csv(track_files[0])
    
    print(f"\nShape: {df_tracks.shape} (rows, columns)")
    print(f"\nColumns ({len(df_tracks.columns)}):")
    print(df_tracks.columns.tolist())
    
    # Identify audio feature columns
    audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                      'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 
                      'key', 'mode', 'time_signature']
    
    available_audio = [col for col in audio_features if col in df_tracks.columns]
    print(f"\nAvailable audio features: {available_audio}")
    
    # Basic stats for audio features
    print("\n--- Audio Feature Statistics ---")
    print(df_tracks[available_audio].describe().round(3).T)
    
    # Check for popularity column
    pop_col = 'popularity' if 'popularity' in df_tracks.columns else None
    if pop_col:
        print(f"\n--- Popularity Distribution ---")
        print(df_tracks[pop_col].describe())
        print(f"Zero popularity tracks: {(df_tracks[pop_col] == 0).sum()} ({(df_tracks[pop_col] == 0).mean()*100:.1f}%)")
    
    # Missing values
    print("\n--- Missing Values ---")
    missing = df_tracks[available_audio + ([pop_col] if pop_col else [])].isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values in audio features!")
    
    # Save sample for inspection
    sample_file = os.path.join(OUTPUT_DIR, 'track_features_sample.csv')
    df_tracks.head(100).to_csv(sample_file, index=False)
    print(f"\nSaved 100-row sample to: {sample_file}")

# ============================================================
# 2. EXPLORE ALBUM FEATURES
# ============================================================
separator("2. ALBUM FEATURES DATA")

album_files = sorted(glob.glob(os.path.join(ALBUM_FEATURES_DIR, 'album_features_2022*.csv')))
print(f"Found {len(album_files)} album feature files")

if album_files:
    print(f"\nLoading sample: {os.path.basename(album_files[0])}")
    df_albums = pd.read_csv(album_files[0])
    
    print(f"\nShape: {df_albums.shape}")
    print(f"\nColumns: {df_albums.columns.tolist()}")
    
    # Check release_date
    if 'release_date' in df_albums.columns:
        print("\n--- Release Date Sample ---")
        print(df_albums['release_date'].head(10).tolist())
        
    # Check label distribution
    if 'label' in df_albums.columns:
        print("\n--- Top 10 Labels ---")
        print(df_albums['label'].value_counts().head(10))

# ============================================================
# 3. EXPLORE DAILY DATA (HDF)
# ============================================================
separator("3. DAILY PLAYLIST DATA (HDF)")

date_folders = sorted([f for f in os.listdir(DAILY_DATA_DIR) 
                       if os.path.isdir(os.path.join(DAILY_DATA_DIR, f))])
print(f"Found {len(date_folders)} date folders: {date_folders}")

if date_folders:
    # Pick first date folder
    sample_date = date_folders[0]
    sample_folder = os.path.join(DAILY_DATA_DIR, sample_date)
    
    hdf_files = glob.glob(os.path.join(sample_folder, '*.hdf'))
    print(f"\nFiles in {sample_date}:")
    for f in hdf_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load one playlist_track_info file
    track_info_files = [f for f in hdf_files if 'playlist_track_info' in f]
    if track_info_files:
        print(f"\nLoading sample: {os.path.basename(track_info_files[0])}")
        df_daily = pd.read_hdf(track_info_files[0], key='/playlist_track_info')
        
        print(f"\nShape: {df_daily.shape}")
        print(f"\nColumns: {df_daily.columns.tolist()}")
        
        # Check track.popularity
        if 'track.popularity' in df_daily.columns:
            df_daily['track.popularity'] = pd.to_numeric(df_daily['track.popularity'], errors='coerce')
            print("\n--- Daily Popularity Distribution ---")
            print(df_daily['track.popularity'].describe())
        
        # Check track.id format
        if 'track.id' in df_daily.columns:
            print("\n--- Track ID Sample ---")
            print(df_daily['track.id'].head(5).tolist())

# ============================================================
# 4. CHECK JOIN FEASIBILITY
# ============================================================
separator("4. JOIN FEASIBILITY CHECK")

if 'df_tracks' in dir() and 'df_daily' in dir():
    # Get track IDs from both sources
    # Track features might have 'track.id' or 'id_x' or similar
    track_id_col_features = None
    for col in ['track.id', 'id_x', 'id']:
        if col in df_tracks.columns:
            track_id_col_features = col
            break
    
    track_id_col_daily = 'track.id' if 'track.id' in df_daily.columns else None
    
    if track_id_col_features and track_id_col_daily:
        ids_features = set(df_tracks[track_id_col_features].dropna().unique())
        ids_daily = set(df_daily[track_id_col_daily].dropna().unique())
        
        common = ids_features.intersection(ids_daily)
        
        print(f"Track IDs in features file: {len(ids_features)}")
        print(f"Track IDs in daily file: {len(ids_daily)}")
        print(f"Common Track IDs: {len(common)} ({len(common)/len(ids_daily)*100:.1f}% of daily)")
        
        if len(common) > 0:
            print("\n✅ JOIN IS POSSIBLE! We can merge track features with daily data.")
        else:
            print("\n⚠️ NO OVERLAP - These datasets may be from different time periods.")
    else:
        print(f"Could not find track ID columns:")
        print(f"  Features: {track_id_col_features}")
        print(f"  Daily: {track_id_col_daily}")

# ============================================================
# 5. QUICK CORRELATION PREVIEW
# ============================================================
separator("5. CORRELATION PREVIEW (Track Features Only)")

if 'df_tracks' in dir() and pop_col and available_audio:
    # Compute correlations with popularity
    df_corr = df_tracks[available_audio + [pop_col]].copy()
    corr_matrix = df_corr.corr()
    
    print(f"Correlation with '{pop_col}':")
    pop_corr = corr_matrix[pop_col].drop(pop_col).sort_values(ascending=False)
    for feat, val in pop_corr.items():
        bar = '█' * int(abs(val) * 20)
        sign = '+' if val > 0 else '-'
        print(f"  {feat:<18} {sign}{abs(val):.3f} {bar}")
    
    # Save correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Matrix: Audio Features vs Popularity')
    plt.tight_layout()
    heatmap_file = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(heatmap_file, dpi=150)
    plt.close()
    print(f"\nSaved correlation heatmap to: {heatmap_file}")

separator("EXPLORATION COMPLETE")
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Review the sample CSVs and heatmap")
print("  2. If join is feasible, proceed to full correlation analysis")
print("  3. Build regression model to predict popularity")

