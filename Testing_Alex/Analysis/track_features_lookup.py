# -*- coding: utf-8 -*-
"""
Track Features Lookup Module
============================
Loads all track feature CSVs, deduplicates using ISRC, and provides
O(1) lookup by track_id for streaming analysis.
"""

import pandas as pd
import numpy as np
import os
import glob
import ast
import pickle
from typing import Dict, Optional, List

# Audio features we care about for analysis
AUDIO_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
    'key', 'mode', 'time_signature'
]

# Columns to keep in lookup (minimize memory)
LOOKUP_COLUMNS = ['track.id', 'isrc'] + AUDIO_FEATURES + ['explicit', 'duration_ms_x']


def extract_isrc(external_ids_str: str) -> Optional[str]:
    """Extract ISRC from external_ids column (stored as string dict)."""
    if pd.isna(external_ids_str) or external_ids_str == '{}':
        return None
    try:
        d = ast.literal_eval(external_ids_str)
        return d.get('isrc', None)
    except:
        return None


def load_all_track_features(data_dir: str) -> pd.DataFrame:
    """
    Load all track feature CSVs into a single DataFrame.
    
    Args:
        data_dir: Path to 'album and track files/tracks/csvs for paper sample/'
    
    Returns:
        Combined DataFrame with all track features
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, 'track_features_*.csv')))
    
    if not csv_files:
        raise FileNotFoundError(f"No track_features_*.csv files found in {data_dir}")
    
    print(f"Loading {len(csv_files)} track feature files...")
    
    dfs = []
    for f in csv_files:
        print(f"  Loading: {os.path.basename(f)}...", end=" ")
        df = pd.read_csv(f)
        print(f"{len(df):,} rows")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(combined):,}")
    
    return combined


def deduplicate_by_isrc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate tracks using ISRC.
    For tracks with same ISRC, keeps MAX popularity and first occurrence of other columns.
    
    Args:
        df: DataFrame with track features
    
    Returns:
        Deduplicated DataFrame
    """
    print("\nExtracting ISRC...")
    df['isrc'] = df['external_ids'].apply(extract_isrc)
    
    isrc_available = df['isrc'].notna().sum()
    print(f"ISRC available for {isrc_available:,} / {len(df):,} tracks ({isrc_available/len(df)*100:.1f}%)")
    
    # Split into tracks with and without ISRC
    has_isrc = df[df['isrc'].notna()].copy()
    no_isrc = df[df['isrc'].isna()].copy()
    
    print(f"\nDeduplicating {len(has_isrc):,} tracks with ISRC...")
    
    # For tracks with ISRC: group and aggregate
    # Take max popularity, first value for other columns
    agg_dict = {}
    for col in df.columns:
        if col == 'isrc':
            continue
        elif col == 'popularity':
            agg_dict[col] = 'max'
        else:
            agg_dict[col] = 'first'
    
    deduped_isrc = has_isrc.groupby('isrc', as_index=False).agg(agg_dict)
    
    # Combine back
    result = pd.concat([deduped_isrc, no_isrc], ignore_index=True)
    
    removed = len(df) - len(result)
    print(f"Removed {removed:,} duplicates ({removed/len(df)*100:.1f}%)")
    print(f"Final unique tracks: {len(result):,}")
    
    return result


def create_lookup_dict(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Create a dictionary for O(1) lookup by track.id.
    
    Args:
        df: DataFrame with track features
    
    Returns:
        Dictionary: {track_id: {feature: value, ...}}
    """
    print("\nCreating lookup dictionary...")
    
    # Select only columns we need
    cols_to_use = [c for c in LOOKUP_COLUMNS if c in df.columns]
    df_slim = df[cols_to_use].copy()
    
    # Convert to dict
    lookup = {}
    for _, row in df_slim.iterrows():
        track_id = row['track.id']
        if pd.notna(track_id):
            features = {col: row[col] for col in cols_to_use if col != 'track.id'}
            lookup[track_id] = features
    
    print(f"Lookup dictionary created with {len(lookup):,} entries")
    
    return lookup


def save_lookup(lookup: Dict, cache_path: str):
    """Save lookup dictionary to pickle file."""
    print(f"\nSaving lookup to: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(lookup, f)
    
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Cache file size: {size_mb:.1f} MB")


def load_lookup(cache_path: str) -> Dict[str, Dict]:
    """Load lookup dictionary from pickle file."""
    print(f"Loading lookup from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        lookup = pickle.load(f)
    print(f"Loaded {len(lookup):,} entries")
    return lookup


def get_track_features_lookup(
    data_dir: str = None,
    cache_path: str = None,
    force_rebuild: bool = False
) -> Dict[str, Dict]:
    """
    Main function: Get track features lookup dictionary.
    Uses cache if available, otherwise builds from CSVs.
    
    Args:
        data_dir: Path to track features CSVs (required if no cache or force_rebuild)
        cache_path: Path to cache pickle file
        force_rebuild: If True, rebuild even if cache exists
    
    Returns:
        Lookup dictionary: {track_id: {features...}}
    """
    # Determine paths
    if cache_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(base_dir, 'track_features_lookup.pkl')
    
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(
            base_dir, '..', 'Data', 
            'album and track files', 'tracks', 'csvs for paper sample'
        )
    
    # Check cache
    if os.path.exists(cache_path) and not force_rebuild:
        return load_lookup(cache_path)
    
    # Build from scratch
    print("="*60)
    print("  BUILDING TRACK FEATURES LOOKUP")
    print("="*60)
    
    # Load all CSVs
    df = load_all_track_features(data_dir)
    
    # Deduplicate
    df_deduped = deduplicate_by_isrc(df)
    
    # Create lookup
    lookup = create_lookup_dict(df_deduped)
    
    # Save cache
    save_lookup(lookup, cache_path)
    
    print("\n" + "="*60)
    print("  LOOKUP READY")
    print("="*60)
    
    return lookup


# CLI for standalone testing
if __name__ == "__main__":
    import sys
    
    # Allow force rebuild via CLI
    force = '--force' in sys.argv
    
    lookup = get_track_features_lookup(force_rebuild=force)
    
    # Show sample
    print("\nSample entries:")
    for i, (track_id, features) in enumerate(lookup.items()):
        if i >= 3:
            break
        print(f"\n  Track: {track_id}")
        for k, v in list(features.items())[:5]:
            print(f"    {k}: {v}")

