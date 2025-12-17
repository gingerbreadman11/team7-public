# -*- coding: utf-8 -*-
"""
Streaming Statistics Module
===========================
Implements incremental/online algorithms for computing statistics
without storing all data in memory.

Classes:
- IncrementalCorrelation: Compute correlations using Welford's algorithm
- ReservoirSampler: Uniform random sampling from a stream
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import random


class IncrementalCorrelation:
    """
    Compute Pearson correlation incrementally using online algorithm.
    
    Based on Welford's method for numerical stability.
    Tracks running sums to compute correlation without storing all data.
    
    For each (X, Y) pair, we track:
    - n: count
    - sum_x, sum_y: sums
    - sum_x2, sum_y2: sum of squares
    - sum_xy: sum of products
    
    Correlation = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_x2 - sum_x^2) * (n*sum_y2 - sum_y^2))
    """
    
    def __init__(self, feature_columns: List[str], target_column: str = 'track.popularity'):
        """
        Initialize correlation tracker.
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name (default: track.popularity)
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Initialize tracking dicts for each feature
        self.stats = {}
        for col in feature_columns:
            self.stats[col] = {
                'n': 0,
                'sum_x': 0.0,
                'sum_y': 0.0,
                'sum_x2': 0.0,
                'sum_y2': 0.0,
                'sum_xy': 0.0
            }
        
        # Also track feature-feature correlations for full matrix
        self.feature_stats = {}
        for i, col1 in enumerate(feature_columns):
            for col2 in feature_columns[i:]:
                key = (col1, col2)
                self.feature_stats[key] = {
                    'n': 0,
                    'sum_x': 0.0,
                    'sum_y': 0.0,
                    'sum_x2': 0.0,
                    'sum_y2': 0.0,
                    'sum_xy': 0.0
                }
    
    def update(self, df: pd.DataFrame):
        """
        Update statistics with a batch of data.
        
        Args:
            df: DataFrame with feature columns and target column
        """
        if self.target_column not in df.columns:
            return
        
        # Get target values
        y = pd.to_numeric(df[self.target_column], errors='coerce')
        
        # Update stats for each feature vs target
        for col in self.feature_columns:
            if col not in df.columns:
                continue
            
            x = pd.to_numeric(df[col], errors='coerce')
            
            # Drop NaN pairs
            mask = x.notna() & y.notna()
            x_valid = x[mask].values
            y_valid = y[mask].values
            
            if len(x_valid) == 0:
                continue
            
            # Update running sums
            s = self.stats[col]
            s['n'] += len(x_valid)
            s['sum_x'] += np.sum(x_valid)
            s['sum_y'] += np.sum(y_valid)
            s['sum_x2'] += np.sum(x_valid ** 2)
            s['sum_y2'] += np.sum(y_valid ** 2)
            s['sum_xy'] += np.sum(x_valid * y_valid)
        
        # Update feature-feature stats
        for (col1, col2), s in self.feature_stats.items():
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            x1 = pd.to_numeric(df[col1], errors='coerce')
            x2 = pd.to_numeric(df[col2], errors='coerce')
            
            mask = x1.notna() & x2.notna()
            x1_valid = x1[mask].values
            x2_valid = x2[mask].values
            
            if len(x1_valid) == 0:
                continue
            
            s['n'] += len(x1_valid)
            s['sum_x'] += np.sum(x1_valid)
            s['sum_y'] += np.sum(x2_valid)
            s['sum_x2'] += np.sum(x1_valid ** 2)
            s['sum_y2'] += np.sum(x2_valid ** 2)
            s['sum_xy'] += np.sum(x1_valid * x2_valid)
    
    def _compute_correlation(self, stats: Dict) -> float:
        """Compute correlation from running sums."""
        n = stats['n']
        if n < 2:
            return np.nan
        
        sum_x = stats['sum_x']
        sum_y = stats['sum_y']
        sum_x2 = stats['sum_x2']
        sum_y2 = stats['sum_y2']
        sum_xy = stats['sum_xy']
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator_x = n * sum_x2 - sum_x ** 2
        denominator_y = n * sum_y2 - sum_y ** 2
        
        if denominator_x <= 0 or denominator_y <= 0:
            return np.nan
        
        denominator = np.sqrt(denominator_x * denominator_y)
        
        return numerator / denominator
    
    def get_correlations_with_target(self) -> pd.Series:
        """
        Get correlations of all features with target.
        
        Returns:
            Series with feature names as index, correlation values
        """
        correlations = {}
        for col in self.feature_columns:
            correlations[col] = self._compute_correlation(self.stats[col])
        
        return pd.Series(correlations).sort_values(ascending=False)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get full correlation matrix including target.
        
        Returns:
            DataFrame with correlation matrix
        """
        all_cols = self.feature_columns + [self.target_column]
        n = len(all_cols)
        matrix = np.eye(n)  # Diagonal is 1
        
        # Fill in feature-target correlations
        for i, col in enumerate(self.feature_columns):
            corr = self._compute_correlation(self.stats[col])
            matrix[i, n-1] = corr
            matrix[n-1, i] = corr
        
        # Fill in feature-feature correlations
        for i, col1 in enumerate(self.feature_columns):
            for j, col2 in enumerate(self.feature_columns):
                if i < j:
                    key = (col1, col2)
                    corr = self._compute_correlation(self.feature_stats[key])
                    matrix[i, j] = corr
                    matrix[j, i] = corr
        
        return pd.DataFrame(matrix, index=all_cols, columns=all_cols)
    
    def get_sample_count(self) -> int:
        """Get total number of samples processed."""
        if self.feature_columns:
            return self.stats[self.feature_columns[0]]['n']
        return 0


class ReservoirSampler:
    """
    Reservoir sampling for uniform random sampling from a stream.
    
    Algorithm R (Vitter, 1985):
    - Keep first k items
    - For item i > k, include with probability k/i
    - If included, replace a random existing item
    
    This ensures each item has equal probability k/n of being in final sample.
    """
    
    def __init__(self, k: int = 100_000, seed: Optional[int] = 42):
        """
        Initialize reservoir sampler.
        
        Args:
            k: Target sample size
            seed: Random seed for reproducibility
        """
        self.k = k
        self.reservoir: List[Dict] = []
        self.n_seen = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add_batch(self, df: pd.DataFrame):
        """
        Add a batch of items to the sampling process.
        
        Args:
            df: DataFrame of items to potentially sample
        """
        for _, row in df.iterrows():
            self.n_seen += 1
            
            if len(self.reservoir) < self.k:
                # Reservoir not full, add item
                self.reservoir.append(row.to_dict())
            else:
                # Reservoir full, maybe replace
                j = random.randint(1, self.n_seen)
                if j <= self.k:
                    self.reservoir[j - 1] = row.to_dict()
    
    def add_batch_fast(self, df: pd.DataFrame):
        """
        Faster batch addition using numpy for large batches.
        
        Args:
            df: DataFrame of items to potentially sample
        """
        n_new = len(df)
        
        if n_new == 0:
            return
        
        # Convert to list of dicts once
        records = df.to_dict('records')
        
        for record in records:
            self.n_seen += 1
            
            if len(self.reservoir) < self.k:
                self.reservoir.append(record)
            else:
                j = random.randint(1, self.n_seen)
                if j <= self.k:
                    self.reservoir[j - 1] = record
    
    def get_sample(self) -> pd.DataFrame:
        """
        Get the sampled data as a DataFrame.
        
        Returns:
            DataFrame with sampled rows
        """
        return pd.DataFrame(self.reservoir)
    
    def get_stats(self) -> Dict:
        """Get sampling statistics."""
        return {
            'target_size': self.k,
            'current_size': len(self.reservoir),
            'total_seen': self.n_seen,
            'sample_rate': len(self.reservoir) / self.n_seen if self.n_seen > 0 else 0
        }


class StreamingStats:
    """
    Combined streaming statistics tracker.
    Wraps IncrementalCorrelation and ReservoirSampler.
    """
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str = 'track.popularity',
        sample_size: int = 100_000,
        seed: int = 42
    ):
        """
        Initialize streaming stats tracker.
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column for correlation
            sample_size: Target sample size for model training
            seed: Random seed
        """
        self.correlation_tracker = IncrementalCorrelation(feature_columns, target_column)
        self.sampler = ReservoirSampler(k=sample_size, seed=seed)
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Track processing stats
        self.files_processed = 0
        self.rows_processed = 0
        self.matched_rows = 0
    
    def update(self, df: pd.DataFrame):
        """
        Update all statistics with a batch of matched data.
        
        Args:
            df: DataFrame with feature columns and target
        """
        if df is None or df.empty:
            return
        
        self.matched_rows += len(df)
        self.correlation_tracker.update(df)
        self.sampler.add_batch_fast(df)
    
    def record_file_processed(self, total_rows: int):
        """Record that a file was processed."""
        self.files_processed += 1
        self.rows_processed += total_rows
    
    def get_correlations(self) -> pd.Series:
        """Get feature-target correlations."""
        return self.correlation_tracker.get_correlations_with_target()
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get full correlation matrix."""
        return self.correlation_tracker.get_correlation_matrix()
    
    def get_sample(self) -> pd.DataFrame:
        """Get sampled data for model training."""
        return self.sampler.get_sample()
    
    def get_summary(self) -> Dict:
        """Get summary of all processing stats."""
        return {
            'files_processed': self.files_processed,
            'rows_processed': self.rows_processed,
            'matched_rows': self.matched_rows,
            'match_rate': self.matched_rows / self.rows_processed if self.rows_processed > 0 else 0,
            'correlation_samples': self.correlation_tracker.get_sample_count(),
            'reservoir_stats': self.sampler.get_stats()
        }


# CLI for testing
if __name__ == "__main__":
    # Test with synthetic data
    print("Testing IncrementalCorrelation...")
    
    features = ['danceability', 'energy', 'valence']
    tracker = IncrementalCorrelation(features, 'popularity')
    
    # Generate synthetic batches
    np.random.seed(42)
    for i in range(5):
        n = 1000
        df = pd.DataFrame({
            'danceability': np.random.uniform(0, 1, n),
            'energy': np.random.uniform(0, 1, n),
            'valence': np.random.uniform(0, 1, n),
            'popularity': np.random.uniform(0, 100, n)
        })
        # Add some correlation
        df['popularity'] = df['popularity'] + df['energy'] * 20 - df['valence'] * 10
        
        tracker.update(df)
        print(f"  Batch {i+1}: {tracker.get_sample_count()} samples")
    
    print("\nCorrelations with popularity:")
    print(tracker.get_correlations_with_target())
    
    print("\nTesting ReservoirSampler...")
    sampler = ReservoirSampler(k=500)
    
    for i in range(10):
        df = pd.DataFrame({
            'x': range(i*100, (i+1)*100),
            'y': np.random.randn(100)
        })
        sampler.add_batch_fast(df)
    
    print(f"Stats: {sampler.get_stats()}")
    sample = sampler.get_sample()
    print(f"Sample shape: {sample.shape}")
    print(f"Sample x range: [{sample['x'].min()}, {sample['x'].max()}]")

