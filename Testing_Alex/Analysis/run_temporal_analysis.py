#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Analysis Pipeline
==========================
Main entry point for running temporal analysis with inline deduplication.

Processes data in 6-month windows, with fresh deduplication cache per period,
collecting up to 500K unique songs per period (by audio features),
computing per-period typicality, and generating temporal comparison plots.

Usage:
    python run_temporal_analysis.py \
        --data-path '/Volumes/Seagate Expansion Drive/.../Data/Spotify' \
        --output-path ./results_temporal \
        --sample-size 500000 \
        --checkpoint-interval 10 \
        --folder-stride 3
"""

import argparse
import os
import sys
import json
import pickle
import gc
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from track_features_lookup import get_track_features_lookup, AUDIO_FEATURES
from temporal_streaming_stats import TemporalStreamingStats, AUDIO_FEATURES as DEDUP_FEATURES


# Define 6-month periods from 2020-03-01 to 2023-02-01
TIME_PERIODS = [
    ("2020-03-01", "2020-08-31", "2020-03_2020-08"),
    ("2020-09-01", "2021-02-28", "2020-09_2021-02"),
    ("2021-03-01", "2021-08-31", "2021-03_2021-08"),
    ("2021-09-01", "2022-02-28", "2021-09_2022-02"),
    ("2022-03-01", "2022-08-31", "2022-03_2022-08"),
    ("2022-09-01", "2023-01-31", "2022-09_2023-01"),
]


def parse_date(date_str: str) -> date:
    """Parse YYYY-MM-DD string to date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def is_folder_in_period(folder_name: str, start_date: date, end_date: date) -> bool:
    """Check if a folder (YYYY-MM-DD) falls within the period."""
    try:
        folder_date = datetime.strptime(folder_name.strip(), "%Y-%m-%d").date()
        return start_date <= folder_date <= end_date
    except ValueError:
        return False


def join_with_features_fast(
    df_daily: pd.DataFrame,
    lookup: Dict[str, Dict],
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Fast vectorized join of daily data with track features.
    
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


class TemporalAnalysisPipeline:
    """
    Pipeline for processing one 6-month period with deduplication.
    """
    
    def __init__(
        self,
        data_path: str,
        output_path: str,
        period_name: str,
        start_date: str,
        end_date: str,
        sample_size: int = 500_000,
        checkpoint_interval: int = 10,
        folder_stride: int = 3,
        k_neighbors: int = 25
    ):
        """
        Initialize pipeline for a single period.
        
        Args:
            data_path: Path to Data_Spotify folder
            output_path: Base output path (period subfolder will be created)
            period_name: Name for this period (e.g., "2020-03_2020-08")
            start_date: Start date string YYYY-MM-DD
            end_date: End date string YYYY-MM-DD
            sample_size: Max unique songs per period
            checkpoint_interval: Save checkpoint every N folders
            folder_stride: Process every Nth folder (e.g., 3)
            k_neighbors: k for typicality calculation
        """
        self.data_path = data_path
        self.period_name = period_name
        self.start_date = parse_date(start_date)
        self.end_date = parse_date(end_date)
        self.sample_size = sample_size
        self.checkpoint_interval = checkpoint_interval
        self.folder_stride = max(1, int(folder_stride))
        self.k_neighbors = k_neighbors
        
        # Period-specific output directory
        self.output_path = os.path.join(output_path, f"period_{period_name}")
        os.makedirs(self.output_path, exist_ok=True)
        
        # State
        self.lookup = None
        self.stats = None
        self.processed_folders = set()
        
        # Paths
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint.pkl')
        self.log_path = os.path.join(self.output_path, 'processing_log.txt')
    
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
            
            self.stats = TemporalStreamingStats(
                max_size=self.sample_size,
                features=DEDUP_FEATURES
            )
            self.stats.load_state(checkpoint['stats_state'])
            self.processed_folders = checkpoint['processed_folders']
            
            self.log(f"Resumed from checkpoint: {len(self.processed_folders)} folders already processed")
            return True
        return False
    
    def save_checkpoint(self):
        """Save current state to checkpoint file."""
        checkpoint = {
            'stats_state': self.stats.get_state(),
            'processed_folders': self.processed_folders,
            'timestamp': datetime.now().isoformat(),
            'period_name': self.period_name
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        self.log(f"Checkpoint saved: {len(self.processed_folders)} folders processed")
    
    def initialize(self, lookup: Dict[str, Dict]):
        """
        Initialize pipeline with lookup (shared across periods).
        
        Args:
            lookup: Track features lookup dictionary
        """
        self.lookup = lookup
        
        self.log("="*60)
        self.log(f"INITIALIZING PERIOD: {self.period_name}")
        self.log("="*60)
        self.log(f"Date range: {self.start_date} to {self.end_date}")
        
        # Try to load checkpoint
        if not self.load_checkpoint():
            # Fresh start
            self.stats = TemporalStreamingStats(
                max_size=self.sample_size,
                features=DEDUP_FEATURES
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
            df_matched = join_with_features_fast(df, self.lookup, DEDUP_FEATURES)
            matched_rows = len(df_matched)
            
            # Update stats (with deduplication)
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
            folder_path: Path to date folder
        
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
    
    def get_folders_in_period(self) -> List[str]:
        """Get all date folders within this period."""
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        skip_folders = {
            'csv_output', 'overall_data', 'scripts', 'spotify',
            'usernames', 'z_2019', 'z_corrupted', 'z_possible_duplicate'
        }
        
        all_folders = []
        try:
            raw_folders = os.listdir(self.data_path)
            for f in raw_folders:
                f_clean = f.strip()
                if (date_pattern.match(f_clean) 
                    and f_clean not in skip_folders
                    and os.path.isdir(os.path.join(self.data_path, f))):
                    if is_folder_in_period(f_clean, self.start_date, self.end_date):
                        all_folders.append(f_clean)
        except Exception as e:
            self.log(f"ERROR listing folders: {e}")
        
        return sorted(all_folders)
    
    def run(self) -> bool:
        """
        Run the analysis for this period.
        
        Returns:
            True if completed successfully
        """
        self.log("="*60)
        self.log(f"STARTING PERIOD: {self.period_name}")
        self.log("="*60)
        
        # Get folders in this period
        all_folders = self.get_folders_in_period()
        self.log(f"Found {len(all_folders)} date folders in period")
        
        # Filter out already processed, then apply stride
        folders_to_process = [
            f for f in all_folders if f not in self.processed_folders
        ]
        
        # Apply folder stride (every Nth folder)
        if self.folder_stride > 1:
            folders_to_process = folders_to_process[::self.folder_stride]
        
        self.log(f"Folders to process: {len(folders_to_process)} (stride={self.folder_stride})")
        
        if not folders_to_process:
            self.log("No new folders to process")
            return True
        
        # Process each folder
        for i, folder in enumerate(folders_to_process):
            folder_path = os.path.join(self.data_path, folder)
            
            result = self.process_folder(folder_path)
            self.processed_folders.add(folder)
            
            # Get current sample stats
            sample_stats = self.stats.sampler.get_stats()
            current_unique = sample_stats['current_size']
            
            self.log(f"  [{i+1}/{len(folders_to_process)}] {folder}: "
                    f"{result['files']} files, {result['matched']:,} matched, "
                    f"unique songs: {current_unique:,}")
            
            # Checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint()
        
        self.log("="*60)
        self.log(f"PERIOD {self.period_name} COMPLETE")
        self.log("="*60)
        
        return True
    
    def save_results(self):
        """Save all results for this period."""
        self.log("="*60)
        self.log("SAVING PERIOD RESULTS")
        self.log("="*60)
        
        # 1. Save deduplicated sample
        sample_df = self.stats.get_sample()
        dedup_path = os.path.join(self.output_path, 'sample_dedup.csv')
        sample_df.to_csv(dedup_path, index=False)
        self.log(f"Saved deduplicated sample ({len(sample_df):,} rows) to: {dedup_path}")
        
        # 2. Compute typicality and save
        self.log(f"Computing typicality (k={self.k_neighbors})...")
        sample_with_typ = self.stats.get_sample_with_typicality(k=self.k_neighbors)
        typ_path = os.path.join(self.output_path, 'sample_with_typicality.csv')
        sample_with_typ.to_csv(typ_path, index=False)
        self.log(f"Saved sample with typicality to: {typ_path}")
        
        # 3. Save summary stats
        summary = self.stats.get_summary()
        summary['period_name'] = self.period_name
        summary['start_date'] = str(self.start_date)
        summary['end_date'] = str(self.end_date)
        summary['k_neighbors'] = self.k_neighbors
        
        # Add typicality stats
        if 'typicality_cosine_norm' in sample_with_typ.columns:
            typ_col = sample_with_typ['typicality_cosine_norm'].dropna()
            summary['typicality_mean'] = float(typ_col.mean())
            summary['typicality_std'] = float(typ_col.std())
            summary['typicality_median'] = float(typ_col.median())
        
        # Add popularity stats
        if 'track.popularity' in sample_with_typ.columns:
            pop_col = pd.to_numeric(sample_with_typ['track.popularity'], errors='coerce').dropna()
            summary['popularity_mean'] = float(pop_col.mean())
            summary['popularity_std'] = float(pop_col.std())
            summary['popularity_median'] = float(pop_col.median())
        
        stats_path = os.path.join(self.output_path, 'period_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.log(f"Saved stats to: {stats_path}")
        
        return sample_with_typ


def run_full_temporal_analysis(
    data_path: str,
    output_path: str,
    sample_size: int = 500_000,
    checkpoint_interval: int = 10,
    folder_stride: int = 3,
    k_neighbors: int = 25,
    force_rebuild_lookup: bool = False
):
    """
    Run full temporal analysis across all periods.
    
    Args:
        data_path: Path to Data_Spotify folder
        output_path: Base output directory
        sample_size: Max unique songs per period
        checkpoint_interval: Save checkpoint every N folders
        folder_stride: Process every Nth folder
        k_neighbors: k for typicality calculation
        force_rebuild_lookup: Force rebuild of track features lookup
    """
    print("="*60)
    print("  TEMPORAL ANALYSIS PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"Sample size per period: {sample_size:,}")
    print(f"Folder stride: {folder_stride}")
    print(f"Periods: {len(TIME_PERIODS)}")
    print()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load progress tracker
    progress_path = os.path.join(output_path, 'progress.json')
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
    else:
        progress = {'completed_periods': [], 'started': datetime.now().isoformat()}
    
    # Load track features lookup (shared across all periods)
    print("="*60)
    print("  LOADING TRACK FEATURES LOOKUP")
    print("="*60)
    lookup = get_track_features_lookup(force_rebuild=force_rebuild_lookup)
    print(f"Lookup loaded: {len(lookup):,} tracks")
    print()
    
    # Process each period
    all_results = {}
    
    for start_str, end_str, period_name in TIME_PERIODS:
        print("\n" + "="*60)
        print(f"  PERIOD: {period_name}")
        print("="*60)
        
        # Check if already completed
        if period_name in progress['completed_periods']:
            print(f"Period {period_name} already completed, loading results...")
            period_output = os.path.join(output_path, f"period_{period_name}")
            typ_path = os.path.join(period_output, 'sample_with_typicality.csv')
            if os.path.exists(typ_path):
                all_results[period_name] = pd.read_csv(typ_path)
            continue
        
        # Create and run pipeline for this period
        pipeline = TemporalAnalysisPipeline(
            data_path=data_path,
            output_path=output_path,
            period_name=period_name,
            start_date=start_str,
            end_date=end_str,
            sample_size=sample_size,
            checkpoint_interval=checkpoint_interval,
            folder_stride=folder_stride,
            k_neighbors=k_neighbors
        )
        
        pipeline.initialize(lookup)
        pipeline.run()
        sample_with_typ = pipeline.save_results()
        
        all_results[period_name] = sample_with_typ
        
        # Mark period as complete
        progress['completed_periods'].append(period_name)
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Force garbage collection between periods
        del pipeline
        gc.collect()
    
    # Generate temporal comparison plots
    print("\n" + "="*60)
    print("  GENERATING TEMPORAL COMPARISON PLOTS")
    print("="*60)
    
    try:
        from plot_temporal_typicality import generate_temporal_plots
        generate_temporal_plots(output_path, all_results)
    except ImportError:
        print("Warning: plot_temporal_typicality.py not found, skipping plots")
    except Exception as e:
        print(f"Warning: Error generating plots: {e}")
    
    # Final summary
    progress['completed'] = datetime.now().isoformat()
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print("\n" + "="*60)
    print("  TEMPORAL ANALYSIS COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_path}")
    print(f"Periods completed: {len(progress['completed_periods'])}")


def main():
    parser = argparse.ArgumentParser(
        description='Temporal Analysis Pipeline with Inline Deduplication',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full temporal analysis
  python run_temporal_analysis.py \\
      --data-path '/Volumes/Seagate Expansion Drive/Music_Project/.../Data/Spotify' \\
      --output-path ./results_temporal \\
      --sample-size 500000 \\
      --folder-stride 3

  # Resume from checkpoint (same command)
  python run_temporal_analysis.py \\
      --data-path '/Volumes/Seagate Expansion Drive/Music_Project/.../Data/Spotify' \\
      --output-path ./results_temporal
        """
    )
    
    # Data paths
    parser.add_argument(
        '--data-path', type=str, required=True,
        help='Path to Data_Spotify folder (e.g., external drive)'
    )
    parser.add_argument(
        '--output-path', type=str, default='results_temporal',
        help='Path to save all results (default: ./results_temporal)'
    )
    
    # Processing options
    parser.add_argument(
        '--sample-size', type=int, default=500_000,
        help='Max unique songs per period (default: 500000)'
    )
    parser.add_argument(
        '--checkpoint-interval', type=int, default=10,
        help='Save checkpoint every N folders (default: 10)'
    )
    parser.add_argument(
        '--folder-stride', type=int, default=3,
        help='Process every Nth folder within each period (default: 3)'
    )
    parser.add_argument(
        '--k-neighbors', type=int, default=25,
        help='Number of neighbors for typicality calculation (default: 25)'
    )
    
    # Other options
    parser.add_argument(
        '--force-rebuild-lookup', action='store_true',
        help='Force rebuild of track features lookup cache'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(base_dir, args.output_path)
    
    # Check data path exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path not found: {args.data_path}")
        sys.exit(1)
    
    # Run analysis
    run_full_temporal_analysis(
        data_path=args.data_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        checkpoint_interval=args.checkpoint_interval,
        folder_stride=args.folder_stride,
        k_neighbors=args.k_neighbors,
        force_rebuild_lookup=args.force_rebuild_lookup
    )


if __name__ == "__main__":
    main()

