# -*- coding: utf-8 -*-
"""
Comparison Models Module
========================
Trains Random Forest, Gradient Boosting, and Bagging regressors
to compare performance of "Audio Only" vs "Audio + Typicality".

Experiment:
1. Split data by year (Train <= 2021, Test > 2021)
2. Train 6 models (3 algorithms x 2 feature sets)
3. Evaluate R2 and MAE
4. Extract feature importance
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'results_test', 'sample_data_enriched.csv')
OUTPUT_METRICS_PATH = os.path.join(BASE_DIR, 'results_test', 'model_comparison_metrics.csv')
OUTPUT_IMPORTANCE_PATH = os.path.join(BASE_DIR, 'results_test', 'audio_feature_importance.csv')

# Feature Sets
FEATURES_AUDIO = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

FEATURES_FULL = FEATURES_AUDIO + ['typicality', 'typicality_squared']

TARGET = 'track.popularity'

def train_and_evaluate(model, X_train, y_train, X_test, y_test, name):
    """Train model and return metrics."""
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'Model': name,
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    print(f"    R2: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.4f}")
    return metrics, model

def main():
    print("="*60)
    print("  MODEL COMPARISON: AUDIO vs. TYPICALITY")
    print("="*60)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Enriched data not found at {DATA_PATH}")
        return

    # 1. Load Data
    print(f"Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Random Train/Test Split (80/20)
    print(f"  Year range: {df['release_year'].min()} - {df['release_year'].max()}")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    all_metrics = []

    # 3. Define Models
    models_config = [
        ('RandomForest', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)),
        ('Bagging', BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=100, random_state=42, n_jobs=-1))
    ]
    
    # 4. Train & Evaluate
    
    # --- Feature Set A: Audio Only ---
    print("\n--- Feature Set A: Audio Only ---")
    X_train_A = train_df[FEATURES_AUDIO]
    X_test_A = test_df[FEATURES_AUDIO]
    
    rf_audio_model = None # To save for feature importance
    
    for name, model in models_config:
        metrics, trained_model = train_and_evaluate(model, X_train_A, y_train, X_test_A, y_test, f"{name} (Audio)")
        metrics['Features'] = 'Audio Only'
        metrics['Algorithm'] = name
        all_metrics.append(metrics)
        
        if name == 'RandomForest':
            rf_audio_model = trained_model

    # --- Feature Set B: Audio + Typicality ---
    print("\n--- Feature Set B: Audio + Typicality ---")
    X_train_B = train_df[FEATURES_FULL]
    X_test_B = test_df[FEATURES_FULL]
    
    # Re-instantiate models to start fresh
    models_config_B = [
        ('RandomForest', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)),
        ('Bagging', BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=100, random_state=42, n_jobs=-1))
    ]
    
    for name, model in models_config_B:
        metrics, _ = train_and_evaluate(model, X_train_B, y_train, X_test_B, y_test, f"{name} (Audio+Typ)")
        metrics['Features'] = 'Audio + Typicality'
        metrics['Algorithm'] = name
        all_metrics.append(metrics)

    # 5. Save Metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"\nSaved metrics to: {OUTPUT_METRICS_PATH}")
    print(metrics_df[['Model', 'R2', 'MAE']])

    # 6. Extract Feature Importance (RF Audio)
    print("\nExtracting feature importance...")
    if rf_audio_model:
        importances = rf_audio_model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': FEATURES_AUDIO,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        fi_df.to_csv(OUTPUT_IMPORTANCE_PATH, index=False)
        print(f"Saved feature importance to: {OUTPUT_IMPORTANCE_PATH}")
        print(fi_df)

if __name__ == "__main__":
    main()

