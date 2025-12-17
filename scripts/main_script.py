v# ETH ZURICH
# MTEC Department
# TIM Group
# Music Streaming Industry Project
#
# Date: 5/02/17
# Author: Santiago Walliser (using code bits from Axel Zeijen's friend for Spotify Scraper)
#
# This script downloads data from Spotify, Apple Music, ...


# ------------------------------------------------------------------

###### timer
import time
time.sleep(7200)
######


### TEMPORARY - LOAD LIBRARIES AND FUNCTIONS ###

# Load libraries
import pandas as pd         # general libraries
import numpy as np
import collections          # for return argument from function: a tupel containing two dataframes
import time                 # get_url() function (is used in the called script_functions.py)
import oauthlib.oauth2
from requests_oauthlib import OAuth2Session # for get_url functions
from oauthlib.oauth2 import BackendApplicationClient # for get_url functions
import requests             # to handle ReadTimeOut exception in get_url_handling_timeout_error() function
from datetime \
    import datetime, \
    timedelta               # to create directory to save data in
import os.path              #t o check if a file exists
import errno                # to create directory - allows us to handle error in case directory already exists
import json                 # to work with / store data # JSON Stuff
from pandas.io.json import json_normalize               # JSON Stuff
import math                 # to use math.floor() function
import sys                  # sys.exit()
import warnings             # suppress warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------


### LOAD LIBRARIES AND FUNCTIONS ###

# Load libraries and read functions in from separate script (used to have a better overview)
import os                  # to evade / and \\ probelms on mac/linux vs. windows
dir_path = os.path.dirname(os.path.realpath(__file__))
exec(open(os.path.join(dir_path, "Spotify", "scripts", "spotify_script_functions.py")).read())
exec(open(os.path.join(dir_path, "Spotify", "scripts", "spotify_website_scraper.py")).read())
exec(open(os.path.join(dir_path, "Deezer", "scripts", "deezer_script_functions.py")).read())


# ------------------------------------------------------------------


### SETTINGS ###

# --- GENERAL --- #

# measure time
start = time.time()

# Get date to create folders
#date = datetime.datetime.now().strftime("%Y-%m-%d")  # date = '2017-12-22' # Layout
date = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
date_prev_day = datetime.datetime.strftime(datetime.datetime.now() - timedelta(1), '%Y-%m-%d')


# --- SPOTIFY --- #

# Spotify API credentials
spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID", "your_client_id_here")
spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "your_client_secret_here")

# Spotify API settings
spotify_token_url = 'https://accounts.spotify.com/api/token'

# Create OAuth Client
spotify_client = BackendApplicationClient(client_id=spotify_client_id)
spotify_oauth = OAuth2Session(client=spotify_client)
spotify_token = spotify_oauth.fetch_token(token_url=spotify_token_url, client_id=spotify_client_id, client_secret=spotify_client_secret)

# Create a folder to store data in - Using today's date
dir_path_spotify = os.path.join(dir_path, "Spotify", "")
dir_path_spotify_date = os.path.join(dir_path_spotify, date, '')  # '' in the end to add an ending slashs
dir_path_spotify_date_prev_day = os.path.join(dir_path_spotify, date_prev_day, '')  # '' in the end to add an ending slashs
dir_path_spotify_csv_output = os.path.join(dir_path_spotify, "csv_output", '')  # '' in the end to add an ending slashs

try:
    os.makedirs(dir_path_spotify_date)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Set playlists # TO CHECK STUFF: additional_playlists = ['axelzeyen'] #
spotify_additional_playlists = pd.read_csv(os.path.join(dir_path_spotify, 'usernames', 'spotify_usernames.csv'))
spotify_additional_playlists = spotify_additional_playlists['x'].tolist()
# spotify_additional_playlists = ['axelzeyen']


# --- DEEZER --- #

# Deezer API credentials
deezer_client_id = os.getenv("DEEZER_CLIENT_ID", "your_client_id_here")
deezer_client_secret = os.getenv("DEEZER_CLIENT_SECRET", "your_client_secret_here")

# Deezer API settings
deezer_token_url = 'https://connect.deezer.com/oauth/access_token.php'

# Create OAuth Client
deezer_client = BackendApplicationClient(client_id=deezer_client_id)
deezer_oauth = OAuth2Session(client=deezer_client)
deezer_token = deezer_oauth.fetch_token(token_url=deezer_token_url, client_id=deezer_client_id, client_secret=deezer_client_secret)

# Create a folder to store data in - Using today's date
dir_path_deezer = os.path.join(dir_path, "Deezer", "")
dir_path_deezer_date = os.path.join(dir_path_deezer, date, '')  # '' in the end to add an ending slashs
dir_path_deezer_date_prev_day = os.path.join(dir_path_deezer, date_prev_day, '')  # '' in the end to add an ending slashs
dir_path_deezer_csv_output = os.path.join(dir_path_deezer, "csv_output", '')  # '' in the end to add an ending slashs

try:
    os.makedirs(dir_path_deezer_date)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Set playlists
deezer_additional_playlists = pd.read_csv(os.path.join(dir_path_deezer, 'usernames', 'deezer_usernames.csv'))
deezer_additional_playlists = deezer_additional_playlists['x'].tolist()
deezer_additional_playlists = list(map(str, deezer_additional_playlists)) # convert to strings


# --- APPLE MUSIC --- #


# ------------------------------------------------------------------

### DOWNLOAD DATA ###


# --- SPOTIFY --- #

# *** ~~ SPOTIFY - PLAYLISTS ~~ ***
# * Files (that are created in the finalized downloading process):
# DATE_playlists.pkl
# * Download
spotify_playlists = get_playlist_df_spotify(dir_path_spotify_date, date, spotify_additional_playlists, spotify_oauth)

# *** ~~ SPOTIFY - PLAYLIST WITH FEATURES, PLAYLIST TRACK INFO AND TRACK IDS ~~ ***
# * Tuple elements:
# pl_track_tuple.playlist_with_features
# pl_track_tuple.gathered_track_ids
# pl_track_tuple.playlist_ids_with_track_ids
# pl_track_tuple.playlist_track_info
# * Files:
# DATE_playlists_with_features.pkl
# DATE_track_ids.pkl
# DATE_playlist_ids_with_track_ids.pkl
# DATE_playlist_track_info.pkl
# * Download
# Split it into ten fragments, such that the quantities to save are smaller - further working memory is then less under pressure
fragment = math.floor(spotify_playlists.shape[0] / 10)
for fragment_i in range(1,11):
    spotify_playlists_fraq_i = spotify_playlists[(fragment_i - 1) * fragment: fragment_i * fragment]
    if fragment_i == 10: spotify_playlists_fraq_i = spotify_playlists[(fragment_i - 1) * fragment: spotify_playlists.shape[0]]
    get_playlist_with_features_and_gathered_track_ids_df(dir_path_spotify_date, spotify_playlists_fraq_i, spotify_oauth, str(fragment_i))

# *** ~~ SPOTIFY - TRACK INFO ~~ ***
# * Tuple elements:
# track_info_tuple.new_tracks_info
# track_info_tuple.overall_tracks_info
# * Files:
# DATE_tracks_info.pkl
# overall_tracks_info.pkl
# * Download
# track_info_tuple = get_track_info(pl_track_tuple) # inactive

# *** ~~ SPOTIFY - ALBUM ~~ ***
# * Tuple elements:
# album_tuple.newly_added_albumfeatures
# album_tuple.overall_albumfeatures
# * Files:
# DATE_albumfeatures.pkl
# overall_albumfeatures.pkl
# * Download
# album_tuple = get_album_features_df(dir_path_spotify, dir_path_spotify_date,
#                                     date, track_info_tuple.new_tracks_info) # inactive

# *** ~~ SPOTIFY - TOP 200 / VIRAL 50 ~~ ***
# Top 200
download_spotify_type_region_date_csv("regional", "global", date_prev_day, dir_path_spotify_date_prev_day)
download_spotify_type_region_date_csv("regional", "us", date_prev_day, dir_path_spotify_date_prev_day)
download_spotify_type_region_date_csv("regional", "fr", date_prev_day, dir_path_spotify_date_prev_day)
download_spotify_type_region_date_csv("regional", "se", date_prev_day, dir_path_spotify_date_prev_day)
# Viral 50
download_spotify_type_region_date_csv("viral", "global", date_prev_day, dir_path_spotify_date_prev_day)
download_spotify_type_region_date_csv("viral", "us", date_prev_day, dir_path_spotify_date_prev_day)
download_spotify_type_region_date_csv("viral", "fr", date_prev_day, dir_path_spotify_date_prev_day)
download_spotify_type_region_date_csv("viral", "se", date_prev_day, dir_path_spotify_date_prev_day)


# --- DEEZER --- #

# *** ~~ DEEZER - PLAYLISTS ~~ ***
deezer_playlists = get_playlist_df_deezer(dir_path_deezer_date, date, deezer_additional_playlists, deezer_oauth)

# *** ~~ DEEZER - TRACKS IN PLAYLISTS ~~ ***
tracks_in_playlists = get_tracks_in_playlists(dir_path_deezer_date, date, deezer_playlists, deezer_oauth)


# --- APPLE MUSIC --- #




# ------------------------------------------------------------------

### TIME MEASUREMENT ###
end = time.time()
print("Needed time: " + str(round((end - start) / 60, 2)) + " min")


# ------------------------------------------------------------------

### STOP PROCESS HERE ###

sys.exit()















# ------------------------------------------------------------------
# ------------------------------------------------------------------

### TO CSV ###

# --- SPOTIFY --- #

# Choose a exisiting data date:
date_to_csv = '2018-01-27'
dir_path_spotify_date_to_csv = os.path.join(dir_path_spotify, date_to_csv, '')  # '' in the end to add an ending slashs

# Playlists
pd.read_pickle(dir_path_spotify_date_to_csv + date + "_playlists.pkl").to_csv(dir_path_spotify_csv_output + date + "_playlists.csv")

# Playlist with features, playlist track info and track ids
for n in range(1, 11):
    n = str(n)
    pd.read_hdf(dir_path_spotify_date_to_csv + date + "_playlists_with_features_" + n + ".hdf", 'playlists').to_csv(dir_path_spotify_csv_output + date + "_playlists_with_features_" + n + ".csv")
    pd.read_hdf(dir_path_spotify_date_to_csv + date + "_track_ids_" + n + ".hdf", 'gathered_track_ids').to_csv(dir_path_spotify_csv_output + date + "_track_ids_" + n + ".csv")
    pd.read_hdf(dir_path_spotify_date_to_csv + date + "_playlist_ids_with_track_ids_" + n + ".hdf", 'playlist_ids_with_track_ids').to_csv(dir_path_spotify_csv_output + date + "_playlist_ids_with_track_ids_" + n + ".csv")
    pd.read_hdf(dir_path_spotify_date_to_csv + date + "_playlist_track_info_" + n + ".hdf", 'playlist_track_info').to_csv(dir_path_spotify_csv_output + date + "_playlist_track_info_" + n + ".csv")

# Track info
# pd.read_pickle(dir_path_spotify_date_to_csv + date + "_tracks_info.pkl").to_csv(dir_path_spotify_csv_output + date + "_tracks_info.csv")
# pd.read_pickle(os.path.join(dir_path_spotify_date_to_csv, "overall_data", "overall_tracks_info.pkl")).to_csv(dir_path_spotify_csv_output + date + "overall_tracks_info.csv")


# Album
# pd.read_pickle(dir_path_spotify_date_to_csv + date + "_albumfeatures.pkl").to_csv(dir_path_spotify_csv_output + date + "overall_albumfeatures.csv")
# pd.read_pickle(os.path.join(dir_path_spotify_date_to_csv, "overall_data", "overall_albumfeatures.pkl")).to_csv(dir_path_spotify_csv_output + date + "overall_albumfeatures.csv")
