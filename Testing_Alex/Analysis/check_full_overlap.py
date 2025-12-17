# -*- coding: utf-8 -*-
"""
Check Total Overlap: All Track Features vs Daily Data
"""

import pandas as pd
import os
import glob

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')

TRACK_FEATURES_DIR = os.path.join(DATA_DIR, 'album and track files', 'tracks', 'csvs for paper sample')
DAILY_DATA_DIR = os.path.join(DATA_DIR, 'Data_Spotify')

print("="*60)
print("  LOADING ALL TRACK FEATURES FILES")
print("="*60)

# Load ALL track features CSVs
track_files = sorted(glob.glob(os.path.join(TRACK_FEATURES_DIR, 'track_features_*.csv')))

all_track_ids = set()
total_rows = 0

for f in track_files:
    print(f"Loading: {os.path.basename(f)}...", end=" ")
    df = pd.read_csv(f, usecols=['track.id'])  # Only load the ID column to save memory
    ids = set(df['track.id'].dropna().unique())
    all_track_ids.update(ids)
    total_rows += len(df)
    print(f"{len(df):,} rows, {len(ids):,} unique IDs")

print(f"\n📊 TRACK FEATURES SUMMARY:")
print(f"   Total rows across all files: {total_rows:,}")
print(f"   Total UNIQUE track IDs: {len(all_track_ids):,}")

# ============================================================
print("\n" + "="*60)
print("  LOADING ALL DAILY DATA FILES")
print("="*60)

# Load ALL daily HDF files
date_folders = sorted([f for f in os.listdir(DAILY_DATA_DIR) 
                       if os.path.isdir(os.path.join(DAILY_DATA_DIR, f))])

all_daily_ids = set()
total_daily_rows = 0

for date_folder in date_folders:
    folder_path = os.path.join(DAILY_DATA_DIR, date_folder)
    hdf_files = glob.glob(os.path.join(folder_path, '*_playlist_track_info_*.hdf'))
    
    print(f"\n📅 {date_folder}:")
    for f in hdf_files:
        print(f"   Loading: {os.path.basename(f)}...", end=" ")
        try:
            df = pd.read_hdf(f, key='/playlist_track_info')
            if 'track.id' in df.columns:
                ids = set(df['track.id'].dropna().unique())
                all_daily_ids.update(ids)
                total_daily_rows += len(df)
                print(f"{len(df):,} rows, {len(ids):,} unique IDs")
            else:
                print("No track.id column")
        except Exception as e:
            print(f"Error: {e}")

print(f"\n📊 DAILY DATA SUMMARY:")
print(f"   Total rows across all files: {total_daily_rows:,}")
print(f"   Total UNIQUE track IDs: {len(all_daily_ids):,}")

# ============================================================
print("\n" + "="*60)
print("  OVERLAP ANALYSIS")
print("="*60)

common_ids = all_track_ids.intersection(all_daily_ids)
only_in_features = all_track_ids - all_daily_ids
only_in_daily = all_daily_ids - all_track_ids

print(f"\n🔗 OVERLAP RESULTS:")
print(f"   Track IDs in features files:  {len(all_track_ids):,}")
print(f"   Track IDs in daily files:     {len(all_daily_ids):,}")
print(f"   ────────────────────────────────────")
pct = (len(common_ids)/len(all_daily_ids)*100) if len(all_daily_ids) > 0 else 0
print(f"   ✅ COMMON (joinable):          {len(common_ids):,} ({pct:.1f}% of daily)")
print(f"   ❌ Only in features:           {len(only_in_features):,}")
print(f"   ❌ Only in daily:              {len(only_in_daily):,}")

if len(common_ids) > 1000:
    print(f"\n✅ GOOD NEWS: {len(common_ids):,} tracks can be joined!")
    print("   This is enough for meaningful correlation/regression analysis.")
elif len(common_ids) > 100:
    print(f"\n⚠️ LIMITED: Only {len(common_ids):,} tracks overlap.")
    print("   We can still do analysis, but results may not generalize well.")
else:
    print(f"\n❌ PROBLEM: Only {len(common_ids):,} tracks overlap.")
    print("   Need to find additional track features data.")

# Save common IDs for later use
if common_ids:
    common_ids_file = os.path.join(BASE_DIR, 'exploration_results', 'common_track_ids.txt')
    os.makedirs(os.path.dirname(common_ids_file), exist_ok=True)
    with open(common_ids_file, 'w') as f:
        f.write('\n'.join(sorted(common_ids)))
    print(f"\n📁 Saved {len(common_ids):,} common IDs to: {common_ids_file}")

