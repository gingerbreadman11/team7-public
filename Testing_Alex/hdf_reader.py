# -*- coding: utf-8 -*-
"""
Deep Comparison of HDF5 Content
"""

import pandas as pd
import os
import numpy as np

BASE_PATH = 'Testing_Alex/Data_Spotify'
DATE_1 = '2023-01-15'
DATE_2 = '2023-01-17'
FILE_TYPE = 'playlist_track_info_1.hdf'

def get_file_path(date_folder, file_type):
    # Construct filename: YYYY-MM-DD_filename.hdf
    filename = f"{date_folder}_{file_type}"
    return os.path.join(BASE_PATH, date_folder, filename)

path1 = get_file_path(DATE_1, FILE_TYPE)
path2 = get_file_path(DATE_2, FILE_TYPE)

print(f"Comparing:\n 1. {path1}\n 2. {path2}\n")

try:
    # Load DataFrames
    # key is usually /playlist_track_info based on previous scripts
    df1 = pd.read_hdf(path1, key='/playlist_track_info')
    df2 = pd.read_hdf(path2, key='/playlist_track_info')

    print("=== Basic Stats ===")
    print(f"{DATE_1} Shape: {df1.shape}")
    print(f"{DATE_2} Shape: {df2.shape}")
    
    print("\n=== Column Comparison ===")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if cols1 == cols2:
        print("Columns are identical.")
        print(f"Columns: {list(cols1)}")
    else:
        print(f"Columns differ!")
        print(f"In {DATE_1} only: {cols1 - cols2}")
        print(f"In {DATE_2} only: {cols2 - cols1}")

    # Data Sample
    print("\n=== Data Preview (First 3 rows of Date 1) ===")
    print(df1.head(3).T)

    # Check for overlap
    # Assuming 'track.id' or similar is a unique identifier. 
    # Let's check duplicates first
    id_col = 'track.id'
    if id_col in df1.columns:
        print(f"\n=== ID Analysis ({id_col}) ===")
        ids1 = set(df1[id_col].unique())
        ids2 = set(df2[id_col].unique())
        
        common = ids1.intersection(ids2)
        only1 = ids1 - ids2
        only2 = ids2 - ids1
        
        print(f"Unique IDs in {DATE_1}: {len(ids1)}")
        print(f"Unique IDs in {DATE_2}: {len(ids2)}")
        print(f"Common IDs: {len(common)}")
        print(f"Only in {DATE_1}: {len(only1)}")
        print(f"Only in {DATE_2}: {len(only2)}")
        
        if len(common) > 0:
            print("\n=== Value Changes in Common Rows ===")
            # Compare a specific column like 'track.popularity'
            comp_col = 'track.popularity'
            if comp_col in df1.columns:
                # Filter to common IDs and sort to align
                # Taking first occurrence if duplicates exist
                d1_common = df1[df1[id_col].isin(common)].drop_duplicates(subset=[id_col]).set_index(id_col)
                d2_common = df2[df2[id_col].isin(common)].drop_duplicates(subset=[id_col]).set_index(id_col)
                
                # Ensure numeric
                s1 = pd.to_numeric(d1_common[comp_col], errors='coerce')
                s2 = pd.to_numeric(d2_common[comp_col], errors='coerce')
                
                diff = s2 - s1
                changed = diff[diff != 0]
                
                print(f"Analyzing '{comp_col}' for {len(common)} common tracks:")
                print(f"  Exact matches: {len(diff) - len(changed)}")
                print(f"  Values changed: {len(changed)}")
                
                if len(changed) > 0:
                    print("\n  Sample Changes:")
                    sample_ids = changed.head(5).index
                    for tid in sample_ids:
                        print(f"    ID {tid}: {s1[tid]} -> {s2[tid]} (Diff: {diff[tid]})")

except Exception as e:
    print(f"An error occurred: {e}")
