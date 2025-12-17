# -*- coding: utf-8 -*-
"""
Popularity Prediction Model
===========================
Train regression models to predict track popularity from audio features.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from typing import Dict, Tuple, Optional, List

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from track_features_lookup import AUDIO_FEATURES


class PopularityModel:
    """
    Train and evaluate models to predict track popularity.
    """
    
    def __init__(
        self,
        features: List[str] = None,
        target: str = 'track.popularity',
        model_type: str = 'random_forest'
    ):
        """
        Initialize model.
        
        Args:
            features: List of feature column names
            target: Target column name
            model_type: One of 'random_forest', 'gradient_boosting', 'ridge', 'linear'
        """
        self.features = features or AUDIO_FEATURES
        self.target = target
        self.model_type = model_type
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.metrics = {}
    
    def _create_model(self):
        """Create the specified model type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from DataFrame.
        
        Args:
            df: DataFrame with features and target
        
        Returns:
            Tuple of (X, y) arrays
        """
        # Select available features
        available_features = [f for f in self.features if f in df.columns]
        
        if not available_features:
            raise ValueError("No feature columns found in DataFrame")
        
        self.features = available_features
        
        # Get features
        X = df[self.features].copy()
        
        # Convert to numeric and handle NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Get target
        y = pd.to_numeric(df[self.target], errors='coerce')
        
        # Drop rows with NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask].values
        y = y[mask].values
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        cross_validate: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            df: DataFrame with features and target
            test_size: Fraction of data for test set
            cross_validate: Whether to run cross-validation
        
        Returns:
            Dict with training metrics
        """
        print("="*60)
        print(f"  TRAINING {self.model_type.upper()} MODEL")
        print("="*60)
        
        # Prepare data
        X, y = self.prepare_data(df)
        print(f"Data shape: {X.shape[0]:,} samples, {X.shape[1]} features")
        print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        print(f"\nTraining {self.model_type}...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Metrics
        self.metrics = {
            'train': {
                'r2': r2_score(y_train, y_pred_train),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
            },
            'test': {
                'r2': r2_score(y_test, y_pred_test),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
            }
        }
        
        print("\n--- Training Metrics ---")
        print(f"Train R²: {self.metrics['train']['r2']:.4f}")
        print(f"Train MAE: {self.metrics['train']['mae']:.2f}")
        print(f"Train RMSE: {self.metrics['train']['rmse']:.2f}")
        
        print("\n--- Test Metrics ---")
        print(f"Test R²: {self.metrics['test']['r2']:.4f}")
        print(f"Test MAE: {self.metrics['test']['mae']:.2f}")
        print(f"Test RMSE: {self.metrics['test']['rmse']:.2f}")
        
        # Cross-validation
        if cross_validate:
            print("\n--- Cross-Validation (5-fold) ---")
            cv_scores = cross_val_score(
                self._create_model(),
                self.scaler.fit_transform(X),
                y,
                cv=5,
                scoring='r2'
            )
            self.metrics['cv'] = {
                'r2_mean': cv_scores.mean(),
                'r2_std': cv_scores.std(),
                'r2_scores': cv_scores.tolist()
            }
            print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Feature importance
        self._compute_feature_importance()
        
        return self.metrics
    
    def _compute_feature_importance(self):
        """Compute and store feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models - use absolute coefficient values
            importance = np.abs(self.model.coef_)
        else:
            importance = np.zeros(len(self.features))
        
        # Create DataFrame sorted by importance
        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n--- Feature Importance ---")
        for _, row in self.feature_importance.head(10).iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"  {row['feature']:<18} {row['importance']:.4f} {bar}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with feature columns
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Path to save model (without extension)
        """
        # Save model
        model_path = f"{path}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'target': self.target,
                'model_type': self.model_type
            }, f)
        print(f"Model saved to: {model_path}")
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = f"{path}_feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            print(f"Feature importance saved to: {importance_path}")
        
        # Save metrics
        metrics_path = f"{path}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
    
    @classmethod
    def load(cls, path: str) -> 'PopularityModel':
        """
        Load model from file.
        
        Args:
            path: Path to model file (without extension)
        
        Returns:
            Loaded PopularityModel instance
        """
        model_path = f"{path}_model.pkl"
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            features=data['features'],
            target=data['target'],
            model_type=data['model_type']
        )
        instance.model = data['model']
        instance.scaler = data['scaler']
        
        return instance


def train_and_compare_models(
    df: pd.DataFrame,
    output_path: str,
    features: List[str] = None
) -> Dict:
    """
    Train multiple models and compare their performance.
    
    Args:
        df: Training data
        output_path: Path to save results
        features: Feature columns to use
    
    Returns:
        Dict with comparison results
    """
    print("="*60)
    print("  COMPARING MULTIPLE MODELS")
    print("="*60)
    
    model_types = ['random_forest', 'gradient_boosting', 'ridge']
    results = {}
    best_model = None
    best_r2 = -float('inf')
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        model = PopularityModel(features=features, model_type=model_type)
        metrics = model.train(df, cross_validate=True)
        
        results[model_type] = {
            'test_r2': metrics['test']['r2'],
            'test_mae': metrics['test']['mae'],
            'test_rmse': metrics['test']['rmse'],
            'cv_r2': metrics.get('cv', {}).get('r2_mean', 0)
        }
        
        # Track best model
        if metrics['test']['r2'] > best_r2:
            best_r2 = metrics['test']['r2']
            best_model = model
    
    # Print comparison
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Test R²':<12} {'Test MAE':<12} {'CV R²':<12}")
    print("-"*60)
    for model_type, metrics in results.items():
        print(f"{model_type:<20} {metrics['test_r2']:.4f}       "
              f"{metrics['test_mae']:.2f}         {metrics['cv_r2']:.4f}")
    
    # Save best model
    if best_model:
        print(f"\nBest model: {best_model.model_type} (R² = {best_r2:.4f})")
        best_model.save(os.path.join(output_path, 'best_model'))
    
    # Save comparison
    comparison_path = os.path.join(output_path, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Comparison saved to: {comparison_path}")
    
    return results, best_model


# CLI for standalone testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Popularity Model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to sample data CSV')
    parser.add_argument('--output-path', type=str, default='results',
                       help='Path to save model')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'ridge', 'linear'],
                       help='Model type to train')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple model types')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df):,} rows")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.compare:
        results, best_model = train_and_compare_models(df, args.output_path)
    else:
        model = PopularityModel(model_type=args.model_type)
        model.train(df)
        model.save(os.path.join(args.output_path, f'{args.model_type}_model'))

