# -*- coding: utf-8 -*-
"""
Streaming Popularity Analysis Pipeline
======================================
Main pipeline for processing 1TB+ of daily Spotify data.
Streams through HDF files, joins with track features, and computes
correlations + samples for regression training.

Memory efficient: Only loads one HDF file at a time.
"""

import pandas as pd
import numpy as np
import os
import glob
import gc
import json
import pickle
from datetime import datetime
from typing import Dict, Optional, List

from track_features_lookup import get_track_features_lookup, AUDIO_FEATURES
from streaming_stats import StreamingStats


def join_with_features(
    df_daily: pd.DataFrame,
    lookup: Dict[str, Dict],
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Join daily data with track features using lookup dictionary.
    
    Args:
        df_daily: DataFrame from HDF file with track.id and track.popularity
        lookup: Dictionary {track_id: {features...}}
        feature_columns: List of feature columns to extract
    
    Returns:
        DataFrame with matched rows containing features and popularity
    """
    if 'track.id' not in df_daily.columns:
        return pd.DataFrame()
    
    # Find matching track IDs
    matched_rows = []
    
    for _, row in df_daily.iterrows():
        track_id = row.get('track.id')
        if track_id and track_id in lookup:
            # Combine daily data with features
            combined = {
                'track.id': track_id,
                'track.popularity': row.get('track.popularity'),
                'playlist_followers': row.get('followers.total'),
            }
            # Add audio features from lookup
            features = lookup[track_id]
            for col in feature_columns:
                combined[col] = features.get(col)
            
            matched_rows.append(combined)
    
    if not matched_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(matched_rows)


def join_with_features_fast(
    df_daily: pd.DataFrame,
    lookup: Dict[str, Dict],
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Fast vectorized join using pandas merge.
    
    Args:
        df_daily: DataFrame from HDF file
        lookup: Dictionary {track_id: {features...}}
        feature_columns: List of feature columns
    
    Returns:
        DataFrame with matched rows
    """
    if 'track.id' not in df_daily.columns:
        return pd.DataFrame()
    
    # Convert lookup to DataFrame for fast merge
    lookup_records = [
        {'track.id': tid, **features}
        for tid, features in lookup.items()
    ]
    df_features = pd.DataFrame(lookup_records)
    
    # Select columns from daily data
    daily_cols = ['track.id', 'track.popularity']
    if 'followers.total' in df_daily.columns:
        daily_cols.append('followers.total')
    
    df_daily_slim = df_daily[daily_cols].copy()
    df_daily_slim['track.popularity'] = pd.to_numeric(
        df_daily_slim['track.popularity'], errors='coerce'
    )
    
    # Merge
    merged = pd.merge(
        df_daily_slim,
        df_features,
        on='track.id',
        how='inner'
    )
    
    return merged


class PopularityAnalysisPipeline:
    """
    Main pipeline for streaming popularity analysis.
    """
    
    def __init__(
        self,
        data_path: str,
        output_path: str,
        sample_size: int = 100_000,
        features: List[str] = None,
        checkpoint_interval: int = 10,
        folder_stride: int = 1,
        start_date: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            data_path: Path to Data_Spotify folder (can be external drive)
            output_path: Path to save results
            sample_size: Target sample size for model training
            features: List of audio features to analyze (default: all)
            checkpoint_interval: Save checkpoint every N folders
            start_date: Only process folders with date >= start_date (YYYY-MM-DD)
        """
        self.data_path = data_path
        self.output_path = output_path
        self.sample_size = sample_size
        self.features = features or AUDIO_FEATURES
        self.checkpoint_interval = checkpoint_interval
        self.folder_stride = max(1, int(folder_stride))
        self.start_date = start_date
        self.start_date_dt = None
        if start_date:
            try:
                self.start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            except Exception:
                self.start_date_dt = None
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # State
        self.lookup = None
        self.stats = None
        self.processed_folders = set()
        
        # Paths
        self.checkpoint_path = os.path.join(output_path, 'checkpoint.pkl')
        self.log_path = os.path.join(output_path, 'processing_log.txt')
    
    def log(self, message: str):
        """Log message to console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        
        with open(self.log_path, 'a') as f:
            f.write(log_line + '\n')
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        if os.path.exists(self.checkpoint_path):
            self.log("Loading checkpoint...")
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.stats = checkpoint['stats']
            self.processed_folders = checkpoint['processed_folders']
            self.log(f"Resumed from checkpoint: {len(self.processed_folders)} folders already processed")
            return True
        return False
    
    def save_checkpoint(self):
        """Save current state to checkpoint file."""
        checkpoint = {
            'stats': self.stats,
            'processed_folders': self.processed_folders,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        self.log(f"Checkpoint saved: {len(self.processed_folders)} folders processed")
    
    def initialize(self, force_rebuild_lookup: bool = False):
        """Initialize lookup and stats."""
        self.log("="*60)
        self.log("INITIALIZING PIPELINE")
        self.log("="*60)
        
        # Load track features lookup
        self.log("Loading track features lookup...")
        self.lookup = get_track_features_lookup(force_rebuild=force_rebuild_lookup)
        self.log(f"Lookup loaded: {len(self.lookup):,} tracks")
        
        # Try to load checkpoint
        if not self.load_checkpoint():
            # Fresh start
            self.stats = StreamingStats(
                feature_columns=self.features,
                target_column='track.popularity',
                sample_size=self.sample_size
            )
            self.processed_folders = set()
    
    def process_hdf_file(self, file_path: str) -> int:
        """
        Process a single HDF file.
        
        Args:
            file_path: Path to HDF file
        
        Returns:
            Number of matched rows
        """
        try:
            # Load HDF
            df = pd.read_hdf(file_path, key='/playlist_track_info')
            total_rows = len(df)
            
            # Join with features
            df_matched = join_with_features_fast(df, self.lookup, self.features)
            matched_rows = len(df_matched)
            
            # Update stats
            if not df_matched.empty:
                self.stats.update(df_matched)
            
            self.stats.record_file_processed(total_rows)
            
            # Free memory
            del df, df_matched
            gc.collect()
            
            return matched_rows
            
        except Exception as e:
            self.log(f"  ERROR processing {os.path.basename(file_path)}: {e}")
            return 0
    
    def process_folder(self, folder_path: str) -> Dict:
        """
        Process all HDF files in a date folder.
        
        Args:
            folder_path: Path to date folder (e.g., Data_Spotify/2023-01-15)
        
        Returns:
            Dict with processing stats for this folder
        """
        folder_name = os.path.basename(folder_path)
        
        # Find playlist_track_info files
        hdf_files = sorted(glob.glob(
            os.path.join(folder_path, '*_playlist_track_info_*.hdf')
        ))
        
        if not hdf_files:
            return {'folder': folder_name, 'files': 0, 'matched': 0}
        
        self.log(f"Processing {folder_name}: {len(hdf_files)} files")
        
        total_matched = 0
        for hdf_file in hdf_files:
            matched = self.process_hdf_file(hdf_file)
            total_matched += matched
        
        return {
            'folder': folder_name,
            'files': len(hdf_files),
            'matched': total_matched
        }
    
    def run(self):
        """
        Run the full analysis pipeline.
        """
        self.log("="*60)
        self.log("STARTING STREAMING ANALYSIS")
        self.log("="*60)
        self.log(f"Data path: {self.data_path}")
        self.log(f"Output path: {self.output_path}")
        
        # Get all date folders (YYYY-MM-DD) and skip known non-data folders
        import re
        # Match standard date folder names like 2018-04-12
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        skip_folders = {
            'csv_output', 'overall_data', 'scripts', 'spotify',
            'usernames', 'z_2019', 'z_corrupted', 'z_possible_duplicate'
        }

        raw_folders = [
            f for f in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, f))
        ]

        # Case: data_path itself is a date folder (no subfolders)
        if not raw_folders and date_pattern.match(os.path.basename(self.data_path).strip()):
            all_folders = ['.']
        else:
            # Normalize names (strip whitespace)
            cleaned_folders = [f.strip() for f in raw_folders]

            all_folders = sorted([
                f for f in cleaned_folders
                if date_pattern.match(f) and f not in skip_folders
            ])

            # Safety fallback: if nothing matched, warn and process all subfolders
            if not all_folders and raw_folders:
                self.log("WARNING: No date folders matched YYYY-MM-DD; falling back to all subfolders.")
                sample = cleaned_folders[:10]
                self.log(f"  Sample subfolders: {sample}")
                all_folders = sorted(cleaned_folders)
        
        self.log(f"Found {len(all_folders)} date folders")
        
        # Filter by start_date if provided
        if self.start_date_dt:
            filtered_folders = []
            for f in all_folders:
                try:
                    d = datetime.strptime(f, "%Y-%m-%d").date()
                    if d >= self.start_date_dt:
                        filtered_folders.append(f)
                except Exception:
                    continue
        else:
            filtered_folders = all_folders

        # Filter out already processed, then apply stride (e.g., process every Nth folder)
        folders_to_process = [
            f for f in filtered_folders if f not in self.processed_folders
        ]
        if self.folder_stride > 1:
            folders_to_process = folders_to_process[::self.folder_stride]
        self.log(f"Folders to process: {len(folders_to_process)} (stride={self.folder_stride})")
        
        # Process each folder
        for i, folder in enumerate(folders_to_process):
            folder_path = os.path.join(self.data_path, folder)
            
            result = self.process_folder(folder_path)
            self.processed_folders.add(folder)
            
            self.log(f"  [{i+1}/{len(folders_to_process)}] {folder}: "
                    f"{result['files']} files, {result['matched']:,} matched rows")
            
            # Checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint()
        
        self.log("="*60)
        self.log("PROCESSING COMPLETE")
        self.log("="*60)
        
        # Print summary
        summary = self.stats.get_summary()
        self.log(f"Total files processed: {summary['files_processed']}")
        self.log(f"Total rows processed: {summary['rows_processed']:,}")
        self.log(f"Matched rows: {summary['matched_rows']:,}")
        self.log(f"Match rate: {summary['match_rate']*100:.2f}%")
        self.log(f"Samples collected: {summary['reservoir_stats']['current_size']:,}")
    
    def save_results(self):
        """Save all results to output directory."""
        self.log("="*60)
        self.log("SAVING RESULTS")
        self.log("="*60)
        
        # 1. Save correlations
        correlations = self.stats.get_correlations()
        corr_path = os.path.join(self.output_path, 'correlations.csv')
        correlations.to_csv(corr_path)
        self.log(f"Saved correlations to: {corr_path}")
        
        # 2. Save correlation matrix
        corr_matrix = self.stats.get_correlation_matrix()
        matrix_path = os.path.join(self.output_path, 'correlation_matrix.csv')
        corr_matrix.to_csv(matrix_path)
        self.log(f"Saved correlation matrix to: {matrix_path}")
        
        # 3. Save sample data for model training
        sample_df = self.stats.get_sample()
        sample_path = os.path.join(self.output_path, 'sample_data.csv')
        sample_df.to_csv(sample_path, index=False)
        self.log(f"Saved sample data ({len(sample_df):,} rows) to: {sample_path}")
        
        # 4. Save summary stats
        summary = self.stats.get_summary()
        summary_path = os.path.join(self.output_path, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.log(f"Saved summary to: {summary_path}")
        
        # 5. Print correlations
        self.log("\n" + "="*60)
        self.log("CORRELATION RESULTS")
        self.log("="*60)
        self.log("Correlation with track.popularity:")
        if correlations.dropna().empty:
            self.log("  (no data)")
        else:
            for feature, corr in correlations.items():
                if pd.isna(corr):
                    continue
                bar = '█' * int(abs(corr) * 30)
                sign = '+' if corr > 0 else '-'
                self.log(f"  {feature:<18} {sign}{abs(corr):.4f} {bar}")


def run_analysis(
    data_path: str,
    output_path: str,
    sample_size: int = 100_000,
    force_rebuild: bool = False
):
    """
    Convenience function to run the full analysis.
    
    Args:
        data_path: Path to Data_Spotify folder
        output_path: Path to save results
        sample_size: Target sample size
        force_rebuild: Force rebuild of lookup cache
    """
    pipeline = PopularityAnalysisPipeline(
        data_path=data_path,
        output_path=output_path,
        sample_size=sample_size
    )
    
    pipeline.initialize(force_rebuild_lookup=force_rebuild)
    pipeline.run()
    pipeline.save_results()
    
    return pipeline


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Streaming Popularity Analysis')
    parser.add_argument('--data-path', type=str, 
                       default='../Data/Data_Spotify',
                       help='Path to Data_Spotify folder')
    parser.add_argument('--output-path', type=str,
                       default='results',
                       help='Path to save results')
    parser.add_argument('--sample-size', type=int,
                       default=100_000,
                       help='Target sample size for model training')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of lookup cache')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(base_dir, args.output_path)
    
    run_analysis(
        data_path=args.data_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        force_rebuild=args.force_rebuild
    )

