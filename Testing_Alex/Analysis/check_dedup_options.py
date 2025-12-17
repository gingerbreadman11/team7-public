# -*- coding: utf-8 -*-
"""
Check Deduplication Options for Same Song / Multiple IDs Problem
"""

import pandas as pd
import os
import ast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
TRACK_FEATURES_FILE = os.path.join(DATA_DIR, 'album and track files', 'tracks', 'csvs for paper sample', 'track_features_1.csv')

print("="*60)
print("  CHECKING DEDUPLICATION OPTIONS")
print("="*60)

# Load sample of track features
df = pd.read_csv(TRACK_FEATURES_FILE, nrows=10000)

print(f"\nLoaded {len(df)} rows")
print(f"\nRelevant columns for deduplication:")

# Check what we have
dedup_cols = ['track.id', 'name', 'artistname', 'album.id', 'external_ids']
for col in dedup_cols:
    if col in df.columns:
        print(f"  ✅ {col}")
        print(f"     Sample: {df[col].iloc[0]}")
    else:
        print(f"  ❌ {col} - NOT FOUND")

# Check external_ids for ISRC
print("\n" + "="*60)
print("  ISRC ANALYSIS (International Standard Recording Code)")
print("="*60)

if 'external_ids' in df.columns:
    # external_ids is stored as string representation of dict
    print(f"\nSample external_ids values:")
    for i, val in enumerate(df['external_ids'].head(5)):
        print(f"  {i+1}. {val}")
    
    # Try to extract ISRC
    def extract_isrc(x):
        if pd.isna(x) or x == '{}':
            return None
        try:
            d = ast.literal_eval(x)
            return d.get('isrc', None)
        except:
            return None
    
    df['isrc'] = df['external_ids'].apply(extract_isrc)
    
    isrc_count = df['isrc'].notna().sum()
    print(f"\n✅ ISRC available for {isrc_count:,} / {len(df):,} tracks ({isrc_count/len(df)*100:.1f}%)")
    
    print(f"\nSample ISRCs:")
    for isrc in df['isrc'].dropna().head(5):
        print(f"  {isrc}")

# Check for duplicates by artist+title
print("\n" + "="*60)
print("  DUPLICATE DETECTION: Same Song, Different Track IDs")
print("="*60)

if 'name' in df.columns and 'artistname' in df.columns:
    # Create artist+title key
    df['artist_title'] = df['artistname'].str.lower().str.strip() + ' - ' + df['name'].str.lower().str.strip()
    
    # Find duplicates
    dup_counts = df.groupby('artist_title')['track.id'].nunique()
    multi_id_songs = dup_counts[dup_counts > 1]
    
    print(f"\nUnique artist+title combinations: {len(dup_counts):,}")
    print(f"Songs with MULTIPLE track IDs: {len(multi_id_songs):,}")
    
    if len(multi_id_songs) > 0:
        print(f"\n📋 Examples of same song with different track IDs:")
        for song, count in multi_id_songs.head(5).items():
            print(f"\n  '{song}' has {count} different track IDs:")
            song_rows = df[df['artist_title'] == song][['track.id', 'album.id', 'popularity']].head(3)
            for _, row in song_rows.iterrows():
                print(f"    - track.id: {row['track.id']}, album: {row['album.id'][:20]}..., popularity: {row['popularity']}")

# Check ISRC duplicates
print("\n" + "="*60)
print("  ISRC-BASED DEDUPLICATION CHECK")
print("="*60)

if 'isrc' in df.columns:
    isrc_counts = df[df['isrc'].notna()].groupby('isrc')['track.id'].nunique()
    multi_id_isrc = isrc_counts[isrc_counts > 1]
    
    print(f"\nUnique ISRCs: {len(isrc_counts):,}")
    print(f"ISRCs with multiple track IDs: {len(multi_id_isrc):,}")
    
    if len(multi_id_isrc) > 0:
        print(f"\n📋 Examples (same ISRC, different track IDs):")
        for isrc, count in multi_id_isrc.head(3).items():
            print(f"\n  ISRC '{isrc}' has {count} track IDs:")
            isrc_rows = df[df['isrc'] == isrc][['track.id', 'name', 'artistname', 'popularity']].head(3)
            for _, row in isrc_rows.iterrows():
                print(f"    - {row['name']} by {row['artistname']} (pop: {row['popularity']})")

print("\n" + "="*60)
print("  RECOMMENDATION")
print("="*60)
print("""
DEDUPLICATION STRATEGY OPTIONS:

1. ISRC-based (PREFERRED if available):
   - Group by ISRC
   - Take MAX popularity across duplicates
   - This is the most reliable unique identifier

2. Artist+Title based (FALLBACK):
   - Normalize: lowercase, strip whitespace
   - Group by (artist, title)
   - Take MAX popularity
   - Risk: Different songs with same name

3. Audio Feature Similarity (ADVANCED):
   - For tracks without ISRC
   - Compare danceability, energy, tempo, etc.
   - If features are very similar + same artist+title → same song
""")

