# -*- coding: utf-8 -*-
"""
Temporal Streaming Statistics with Inline Deduplication
========================================================
Implements a deduplicating reservoir sampler that ensures no duplicate songs
(based on audio features) are stored, keeping the one with max popularity.

For temporal analysis where each 6-month period needs a fresh, deduplicated
sample of up to 500K unique songs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Audio features used for deduplication and typicality
AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "key",
    "mode",
    "time_signature",
]


def create_feature_key(record: Dict, features: List[str] = AUDIO_FEATURES) -> Optional[Tuple]:
    """
    Create a hashable key from audio features for deduplication.
    
    Args:
        record: Dictionary containing audio feature values
        features: List of feature column names
    
    Returns:
        Tuple of feature values (hashable) or None if any feature is missing/NaN
    """
    values = []
    for f in features:
        val = record.get(f)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        # Round floats to handle floating point precision issues
        if isinstance(val, float):
            val = round(val, 6)
        values.append(val)
    return tuple(values)


class DeduplicatingReservoirSampler:
    """
    Deduplicating sampler that collects unique songs by audio features.
    
    This maintains:
    1. A reservoir list of records (up to max_size unique songs)
    2. A hash map from feature tuple -> index in reservoir
    
    When adding a record:
    - If features already seen: keep the one with HIGHER popularity (update in place)
    - If new features and reservoir not full: add directly
    - If new features and reservoir full: SKIP (we already have max unique songs)
    
    This ensures:
    - ALL records in the reservoir have UNIQUE audio feature combinations
    - When duplicates are seen, we always keep the max popularity version
    - Once we reach max_size unique songs, no new songs are added
    """
    
    def __init__(
        self,
        max_size: int = 500_000,
        popularity_column: str = 'track.popularity',
        features: List[str] = None,
        seed: Optional[int] = 42
    ):
        """
        Initialize deduplicating reservoir sampler.
        
        Args:
            max_size: Maximum number of unique songs to keep
            popularity_column: Column name for popularity (to keep max)
            features: Audio feature columns for deduplication
            seed: Random seed for reproducibility
        """
        self.max_size = max_size
        self.popularity_column = popularity_column
        self.features = features or AUDIO_FEATURES
        
        # Core data structures
        self.reservoir: List[Dict] = []
        self.feature_to_idx: Dict[Tuple, int] = {}  # feature_tuple -> reservoir index
        
        # Stats tracking
        self.n_seen = 0
        self.n_duplicates_replaced = 0
        self.n_duplicates_skipped = 0
        self.n_skipped_full = 0  # New unique songs skipped because reservoir is full
        self.n_skipped_missing = 0  # Records skipped due to missing features/popularity
    
    def add_record(self, record: Dict) -> bool:
        """
        Add a single record to the sampler with deduplication.
        
        Args:
            record: Dictionary with audio features and popularity
        
        Returns:
            True if record was added/updated, False if skipped
        """
        self.n_seen += 1
        
        # Create feature key for deduplication
        feature_key = create_feature_key(record, self.features)
        if feature_key is None:
            # Missing features, skip this record
            self.n_skipped_missing += 1
            return False
        
        # Get popularity
        popularity = record.get(self.popularity_column)
        if popularity is None or (isinstance(popularity, float) and np.isnan(popularity)):
            self.n_skipped_missing += 1
            return False
        popularity = float(popularity)
        
        # Check if we've seen this feature combination before
        if feature_key in self.feature_to_idx:
            idx = self.feature_to_idx[feature_key]
            existing_popularity = float(self.reservoir[idx].get(self.popularity_column, 0))
            
            if popularity > existing_popularity:
                # Replace with higher popularity version
                self.reservoir[idx] = record.copy()
                self.n_duplicates_replaced += 1
                return True
            else:
                # Keep existing (has higher or equal popularity)
                self.n_duplicates_skipped += 1
                return False
        
        # New unique feature combination
        if len(self.reservoir) < self.max_size:
            # Reservoir not full, add directly
            idx = len(self.reservoir)
            self.reservoir.append(record.copy())
            self.feature_to_idx[feature_key] = idx
            return True
        else:
            # Reservoir full - we already have max_size unique songs
            # Skip this new unique song (don't replace existing ones)
            self.n_skipped_full += 1
            return False
    
    def add_batch(self, df: pd.DataFrame):
        """
        Add a batch of records from a DataFrame.
        
        Args:
            df: DataFrame with audio features and popularity
        """
        records = df.to_dict('records')
        for record in records:
            self.add_record(record)
    
    def get_sample(self) -> pd.DataFrame:
        """
        Get the current reservoir as a DataFrame.
        
        Returns:
            DataFrame with all unique sampled records
        """
        if not self.reservoir:
            return pd.DataFrame()
        return pd.DataFrame(self.reservoir)
    
    def get_stats(self) -> Dict:
        """Get sampling and deduplication statistics."""
        return {
            'max_size': self.max_size,
            'current_size': len(self.reservoir),
            'total_seen': self.n_seen,
            'duplicates_replaced': self.n_duplicates_replaced,
            'duplicates_skipped': self.n_duplicates_skipped,
            'skipped_reservoir_full': self.n_skipped_full,
            'skipped_missing_data': self.n_skipped_missing,
            'unique_feature_combos': len(self.feature_to_idx),
            'fill_rate': len(self.reservoir) / self.max_size if self.max_size > 0 else 0,
            'is_full': len(self.reservoir) >= self.max_size
        }
    
    def get_state(self) -> Dict:
        """Get full state for checkpointing."""
        return {
            'reservoir': self.reservoir.copy(),
            'feature_to_idx': self.feature_to_idx.copy(),
            'n_seen': self.n_seen,
            'n_duplicates_replaced': self.n_duplicates_replaced,
            'n_duplicates_skipped': self.n_duplicates_skipped,
            'n_skipped_full': self.n_skipped_full,
            'n_skipped_missing': self.n_skipped_missing,
            'max_size': self.max_size,
            'popularity_column': self.popularity_column,
            'features': self.features
        }
    
    def load_state(self, state: Dict):
        """Restore state from checkpoint."""
        self.reservoir = state['reservoir']
        self.feature_to_idx = state['feature_to_idx']
        self.n_seen = state['n_seen']
        self.n_duplicates_replaced = state['n_duplicates_replaced']
        self.n_duplicates_skipped = state['n_duplicates_skipped']
        self.n_skipped_full = state.get('n_skipped_full', 0)
        self.n_skipped_missing = state.get('n_skipped_missing', 0)
        self.max_size = state.get('max_size', self.max_size)
        self.popularity_column = state.get('popularity_column', self.popularity_column)
        self.features = state.get('features', self.features)


def compute_typicality_cosine(
    df: pd.DataFrame,
    features: List[str] = AUDIO_FEATURES,
    k: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute typicality using cosine similarity to k nearest neighbors.
    
    Args:
        df: DataFrame with audio features
        features: List of feature columns
        k: Number of neighbors for typicality
    
    Returns:
        Tuple of (raw_typicality, normalized_typicality) arrays
    """
    # Drop rows with missing features
    df_clean = df.dropna(subset=features)
    
    if len(df_clean) < k + 1:
        # Not enough data for kNN
        return np.full(len(df), np.nan), np.full(len(df), np.nan)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[features].values).astype(np.float32)
    
    # Build kNN index with cosine distance
    nn = NearestNeighbors(
        n_neighbors=min(k + 1, len(df_clean)),
        metric='cosine',
        algorithm='brute'
    )
    nn.fit(X)
    
    # Query neighbors
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # Exclude self (first neighbor)
    
    # Convert cosine distance to similarity
    cos_sim = 1 - distances
    typicality = cos_sim.mean(axis=1)
    
    # Normalize to 0-1
    typ_min, typ_max = typicality.min(), typicality.max()
    if typ_max > typ_min:
        typicality_norm = (typicality - typ_min) / (typ_max - typ_min)
    else:
        typicality_norm = np.ones_like(typicality) * 0.5
    
    return typicality, typicality_norm


class TemporalStreamingStats:
    """
    Combined stats tracker for temporal analysis with deduplication.
    
    Wraps DeduplicatingReservoirSampler and tracks processing stats.
    """
    
    def __init__(
        self,
        max_size: int = 500_000,
        features: List[str] = None,
        target_column: str = 'track.popularity',
        seed: int = 42
    ):
        """
        Initialize temporal streaming stats.
        
        Args:
            max_size: Maximum unique songs per period
            features: Audio feature columns
            target_column: Popularity column name
            seed: Random seed
        """
        self.features = features or AUDIO_FEATURES
        self.target_column = target_column
        
        self.sampler = DeduplicatingReservoirSampler(
            max_size=max_size,
            popularity_column=target_column,
            features=self.features,
            seed=seed
        )
        
        # Processing stats
        self.files_processed = 0
        self.rows_processed = 0
        self.matched_rows = 0
    
    def update(self, df: pd.DataFrame):
        """
        Update stats with a batch of matched data.
        
        Args:
            df: DataFrame with features and popularity
        """
        if df is None or df.empty:
            return
        
        self.matched_rows += len(df)
        self.sampler.add_batch(df)
    
    def record_file_processed(self, total_rows: int):
        """Record that a file was processed."""
        self.files_processed += 1
        self.rows_processed += total_rows
    
    def get_sample(self) -> pd.DataFrame:
        """Get deduplicated sample."""
        return self.sampler.get_sample()
    
    def get_sample_with_typicality(self, k: int = 25) -> pd.DataFrame:
        """
        Get sample with typicality computed.
        
        Args:
            k: Number of neighbors for typicality
        
        Returns:
            DataFrame with typicality columns added
        """
        df = self.get_sample()
        if df.empty:
            return df
        
        # Compute typicality
        typicality, typicality_norm = compute_typicality_cosine(df, self.features, k)
        
        df = df.copy()
        df['typicality_cosine'] = typicality
        df['typicality_cosine_norm'] = typicality_norm
        df['typicality_cosine_norm_sq'] = typicality_norm ** 2
        
        return df
    
    def get_summary(self) -> Dict:
        """Get summary of processing and sampling stats."""
        sampler_stats = self.sampler.get_stats()
        return {
            'files_processed': self.files_processed,
            'rows_processed': self.rows_processed,
            'matched_rows': self.matched_rows,
            'match_rate': self.matched_rows / self.rows_processed if self.rows_processed > 0 else 0,
            'sampler_stats': sampler_stats
        }
    
    def get_state(self) -> Dict:
        """Get full state for checkpointing."""
        return {
            'sampler_state': self.sampler.get_state(),
            'files_processed': self.files_processed,
            'rows_processed': self.rows_processed,
            'matched_rows': self.matched_rows,
            'features': self.features,
            'target_column': self.target_column
        }
    
    def load_state(self, state: Dict):
        """Restore state from checkpoint."""
        self.sampler.load_state(state['sampler_state'])
        self.files_processed = state['files_processed']
        self.rows_processed = state['rows_processed']
        self.matched_rows = state['matched_rows']
        self.features = state.get('features', self.features)
        self.target_column = state.get('target_column', self.target_column)


# CLI for testing
if __name__ == "__main__":
    print("Testing DeduplicatingReservoirSampler...")
    
    # Create synthetic data with duplicates
    np.random.seed(42)
    n_records = 1000
    
    # Create base features (some will be duplicates)
    base_features = np.random.rand(500, len(AUDIO_FEATURES))
    
    records = []
    for i in range(n_records):
        # 50% chance of being a duplicate of an earlier record
        if i > 100 and np.random.rand() > 0.5:
            base_idx = np.random.randint(0, min(i, 500))
            features = base_features[base_idx % 500]
        else:
            features = base_features[i % 500]
        
        record = {f: features[j] for j, f in enumerate(AUDIO_FEATURES)}
        record['track.popularity'] = np.random.randint(0, 100)
        record['track.id'] = f'track_{i}'
        records.append(record)
    
    df = pd.DataFrame(records)
    print(f"Created {len(df)} records")
    
    # Test sampler
    sampler = DeduplicatingReservoirSampler(max_size=300, seed=42)
    sampler.add_batch(df)
    
    stats = sampler.get_stats()
    print(f"\nSampler stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    sample = sampler.get_sample()
    print(f"\nSample shape: {sample.shape}")
    
    # Test typicality
    print("\nComputing typicality...")
    typ, typ_norm = compute_typicality_cosine(sample, AUDIO_FEATURES, k=10)
    print(f"Typicality range: [{typ_norm.min():.4f}, {typ_norm.max():.4f}]")
    print(f"Typicality mean: {typ_norm.mean():.4f}")

