#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Analysis - Main Entry Point
================================
Complete pipeline for streaming popularity analysis.

Usage:
    python run_analysis.py --data-path /path/to/Data_Spotify --output-path ./results
    
For external hard drive:
    python run_analysis.py --data-path /Volumes/MyHDD/Data_Spotify --output-path ./results
"""

import argparse
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='Streaming Popularity Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on local data (current test folders)
  python run_analysis.py
  
  # Run on external hard drive
  python run_analysis.py --data-path /Volumes/ExternalHDD/Data_Spotify
  
  # Full analysis with custom sample size
  python run_analysis.py --data-path /path/to/data --sample-size 200000
  
  # Only generate visualizations from existing results
  python run_analysis.py --visualize-only --results-path ./results
        """
    )
    
    # Data paths
    parser.add_argument(
        '--data-path', type=str,
        default=None,
        help='Path to Data_Spotify folder (can be external drive)'
    )
    parser.add_argument(
        '--output-path', type=str,
        default='results',
        help='Path to save all results (default: ./results)'
    )
    
    # Processing options
    parser.add_argument(
        '--sample-size', type=int,
        default=100_000,
        help='Target sample size for model training (default: 100000)'
    )
    parser.add_argument(
        '--force-rebuild', action='store_true',
        help='Force rebuild of track features lookup cache'
    )
    parser.add_argument(
        '--checkpoint-interval', type=int,
        default=10,
        help='Save checkpoint every N folders (default: 10)'
    )
    parser.add_argument(
        '--folder-stride', type=int,
        default=1,
        help='Process every Nth folder (e.g., 30 to sample monthly)'
    )
    
    # Model options
    parser.add_argument(
        '--skip-model', action='store_true',
        help='Skip model training (only compute correlations)'
    )
    parser.add_argument(
        '--compare-models', action='store_true',
        help='Train and compare multiple model types'
    )
    
    # Visualization
    parser.add_argument(
        '--visualize-only', action='store_true',
        help='Only generate visualizations from existing results'
    )
    parser.add_argument(
        '--results-path', type=str,
        default=None,
        help='Path to existing results (for --visualize-only)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default data path
    if args.data_path is None:
        args.data_path = os.path.join(base_dir, '..', 'Data', 'Data_Spotify')
    elif not os.path.isabs(args.data_path):
        args.data_path = os.path.abspath(args.data_path)
    
    # Output path
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(base_dir, args.output_path)
    
    # Results path for visualize-only
    if args.results_path is None:
        args.results_path = args.output_path
    elif not os.path.isabs(args.results_path):
        args.results_path = os.path.join(base_dir, args.results_path)
    
    # Print banner
    print("="*60)
    print("  SPOTIFY POPULARITY ANALYSIS PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print()
    
    # Visualize only mode
    if args.visualize_only:
        print("Mode: Visualization only")
        from visualize_results import generate_all_plots
        generate_all_plots(args.results_path, args.output_path)
        return
    
    # Check data path exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path not found: {args.data_path}")
        print("Please specify the correct path with --data-path")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # ========================================
    # STEP 1: Run streaming analysis
    # ========================================
    print("\n" + "="*60)
    print("  STEP 1: STREAMING DATA PROCESSING")
    print("="*60)
    
    from popularity_analysis_streaming import PopularityAnalysisPipeline
    
    pipeline = PopularityAnalysisPipeline(
        data_path=args.data_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        checkpoint_interval=args.checkpoint_interval,
        folder_stride=args.folder_stride
    )
    
    pipeline.initialize(force_rebuild_lookup=args.force_rebuild)
    pipeline.run()
    pipeline.save_results()
    
    # ========================================
    # STEP 2: Train model
    # ========================================
    if not args.skip_model:
        print("\n" + "="*60)
        print("  STEP 2: MODEL TRAINING")
        print("="*60)
        
        import pandas as pd
        from popularity_model import PopularityModel, train_and_compare_models
        
        # Load sampled data
        sample_path = os.path.join(args.output_path, 'sample_data.csv')
        
        if os.path.exists(sample_path):
            sample_df = pd.read_csv(sample_path)
            print(f"Loaded sample data: {len(sample_df):,} rows")
            
            if len(sample_df) >= 100:  # Minimum for meaningful training
                if args.compare_models:
                    train_and_compare_models(sample_df, args.output_path)
                else:
                    model = PopularityModel(model_type='random_forest')
                    model.train(sample_df)
                    model.save(os.path.join(args.output_path, 'popularity_model'))
            else:
                print("WARNING: Not enough samples for model training")
        else:
            print(f"WARNING: Sample data not found at {sample_path}")
    
    # ========================================
    # STEP 3: Generate visualizations
    # ========================================
    print("\n" + "="*60)
    print("  STEP 3: GENERATING VISUALIZATIONS")
    print("="*60)
    
    from visualize_results import generate_all_plots
    generate_all_plots(args.output_path)
    
    # ========================================
    # COMPLETE
    # ========================================
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {args.output_path}")
    print(f"\nFiles generated:")
    for f in sorted(os.listdir(args.output_path)):
        fpath = os.path.join(args.output_path, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  - {f} ({size_str})")


if __name__ == "__main__":
    main()

