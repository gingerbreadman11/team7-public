# -*- coding: utf-8 -*-
"""
Train models on the 500K sample (results_large) with:
- Random split (80/20)
- Forecast split (train <= 2021, test >= 2022)

Models: RandomForest, GradientBoosting, Bagging
Outputs: results_large/model_comparison_metrics.csv
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_large")
INPUT_SAMPLE = os.path.join(RESULTS_DIR, "sample_data.csv")
METRICS_OUT = os.path.join(RESULTS_DIR, "model_comparison_metrics.csv")

TRACK_FEATURES_DIR = os.path.join(BASE_DIR, "..", "Data", "album and track files", "tracks", "csvs for paper sample")
ALBUM_FEATURES_DIR = os.path.join(BASE_DIR, "..", "Data", "album and track files", "albums", "album files")

# Features
AUDIO_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
    "key", "mode", "time_signature"
]

TARGET = "track.popularity"


def load_all_data(directory, pattern):
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Error loading {os.path.basename(f)}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def extract_year(date_str):
    if pd.isna(date_str):
        return None
    try:
        return int(str(date_str)[:4])
    except Exception:
        return None


def train_and_eval(model, X_train, y_train, X_test, y_test, name, split_type):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Split": split_type,
        "Model": name,
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def main():
    print("=" * 60)
    print("  MODELING ON 500K SAMPLE (results_large)")
    print("=" * 60)

    if not os.path.exists(INPUT_SAMPLE):
        print(f"ERROR: sample not found at {INPUT_SAMPLE}")
        return

    df = pd.read_csv(INPUT_SAMPLE)
    print(f"Loaded sample: {len(df):,} rows")

    # Load metadata to enrich with release_year
    print("Loading track/album metadata for release_year...")
    tracks_meta = load_all_data(TRACK_FEATURES_DIR, "track_features_*.csv")
    albums_meta = load_all_data(ALBUM_FEATURES_DIR, "album_features_*.csv")

    if tracks_meta.empty or albums_meta.empty:
        print("ERROR: missing metadata; cannot time-split.")
        return

    track_album = tracks_meta[["track.id", "album.id"]].drop_duplicates("track.id")
    album_dates = albums_meta[["id", "release_date"]].rename(columns={"id": "album.id"}).drop_duplicates("album.id")

    df = pd.merge(df, track_album, on="track.id", how="left")
    df = pd.merge(df, album_dates, on="album.id", how="left")
    df["release_year"] = df["release_date"].apply(extract_year)

    before = len(df)
    df = df.dropna(subset=AUDIO_FEATURES + [TARGET, "release_year"])
    df["release_year"] = df["release_year"].astype(int)
    print(f"Kept {len(df):,} rows after requiring release_year and features (dropped {before - len(df):,}).")

    # Ensure numeric
    for col in AUDIO_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=AUDIO_FEATURES + [TARGET])

    # Prepare splits
    results = []

    # Random split
    print("\n--- Random split (80/20) ---")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df[AUDIO_FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[AUDIO_FEATURES]
    y_test = test_df[TARGET]

    models = [
        ("RandomForest", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)),
        ("Bagging", BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=100, random_state=42, n_jobs=-1)),
    ]
    for name, model in models:
        m = train_and_eval(model, X_train, y_train, X_test, y_test, name, "Random")
        results.append(m)
        print(f"  {name} Random: R2={m['R2']:.4f}, MAE={m['MAE']:.2f}")

    # Forecast split: train <= 2021, test >= 2022
    print("\n--- Forecast split (train <=2021, test >=2022) ---")
    train_mask = df["release_year"] <= 2021
    test_mask = df["release_year"] >= 2022
    train_df = df[train_mask]
    test_df = df[test_mask]

    if len(test_df) == 0 or len(train_df) == 0:
        print("WARNING: insufficient data for forecast split; writing placeholder rows.")
        for name, _ in models:
            results.append({
                "Split": "Forecast",
                "Model": name,
                "R2": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan
            })
    else:
        X_train = train_df[AUDIO_FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[AUDIO_FEATURES]
        y_test = test_df[TARGET]
        for name, model in models:
            m = train_and_eval(model, X_train, y_train, X_test, y_test, name, "Forecast")
            results.append(m)
            print(f"  {name} Forecast: R2={m['R2']:.4f}, MAE={m['MAE']:.2f}")

    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_OUT, index=False)
    print(f"\nSaved metrics to: {METRICS_OUT}")
    print(metrics_df[["Split", "Model", "R2", "MAE"]])


if __name__ == "__main__":
    main()

