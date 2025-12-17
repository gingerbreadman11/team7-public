# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:07:00 2025

File loader Spotify data

@author: zeijena
"""

import pandas as pd
import os
from datetime import datetime

main_folder_path = 'Z:/Public/25 music project data/downloaded files/Music_Streaming_Industry_Project/Data/Spotify'

all_folders = sorted([folder for folder in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, folder))])

all_folders = all_folders[:1]

for idx, folder_name in enumerate(all_folders):
    print(folder_name)
    folder_path = os.path.join(main_folder_path, folder_name)

    # Check if folder exists
    if not os.path.exists(folder_path):
        continue

    date_observed = datetime.strptime(folder_name, '%Y-%m-%d').date()
    
    dfs = []
    #for i in range(1, 11):
    for i in range(1, 3):
        print(i)
        file_name = f"{folder_name}_playlist_track_info_{i}.hdf"
        file_path = os.path.join(folder_path, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            continue

        # Read and process the file
        try:
            df = pd.read_hdf(file_path, key='/playlist_track_info')
            #df = df[['track.id', 'track.popularity']]
            df['track.popularity'] = pd.to_numeric(df['track.popularity'], errors='coerce')
            dfs.append(df)
            big_df_playlist_track_info = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            break  # Skip the entire day if there is an error with any file


    dfs = []
    #for i in range(1, 11):
    for i in range(1, 3):
        print(i)
        file_name = f"{folder_name}_playlist_ids_with_track_ids_{i}.hdf"
        file_path = os.path.join(folder_path, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            continue

        # Read and process the file
        try:
            df = pd.read_hdf(file_path, key='/playlist_ids_with_track_ids')
            dfs.append(df)
            big_df_playlist_ids_with_track_ids = dfs
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            break  # Skip the entire day if there is an error with any file

    dfs = []
    #for i in range(1, 11):
    for i in range(1, 3):
        print(i)
        file_name = f"{folder_name}_playlists_with_features_{i}.hdf"
        file_path = os.path.join(folder_path, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            continue

        # Read and process the file
        try:
            df = pd.read_hdf(file_path, key='/playlists')
            dfs.append(df)
            big_df_playlists = pd.concat(dfs, ignore_index=True)

        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            break  # Skip the entire day if there is an error with any file
            
            
    dfs = []
    #for i in range(1, 11):
    for i in range(1, 3):
        print(i)
        file_name = f"{folder_name}_track_ids_{i}.hdf"
        file_path = os.path.join(folder_path, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            continue

        # Read and process the file
        try:            
            df = pd.read_hdf(file_path, key='/gathered_track_ids')
            dfs.append(df)
            big_df_track_ids = pd.concat(dfs, ignore_index=True)

        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            break  # Skip the entire day if there is an error with any file

