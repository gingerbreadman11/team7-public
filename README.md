# Standing Out in Platforms: Spotify Typicality Analysis

## Overview
Analysis of the relationship between musical typicality and track popularity on Spotify. Implements a streaming pipeline that processes daily playlist data in 6-month temporal windows, computes cosine-based typicality, and examines optimal distinctiveness.

## Quick Start

```bash
cd Testing_Alex/Analysis
pip install -r requirements.txt

python run_temporal_analysis.py \
    --data-path /path/to/Data_Spotify \
    --output-path ./results_temporal \
    --sample-size 500000 \
    --folder-stride 3
```

## Data Access & Structure

**Data Access:** The dataset used in this analysis is not included in this repository due to restrictions on publishing data online. Please contact me directly if you would like to request access for research purposes.

**Required Structure:**
- **Spotify Data:** Daily HDF files organized in folders by date (e.g., `Data_Spotify/YYYY-MM-DD/`).
- **Audio Features:** CSV files containing track metadata and audio features (located in `album and track files/`).

**Configuring Paths:** You can point the pipeline to your data by using the `--data-path` flag when running the script (as shown in the Quick Start).

## Documentation

See [`Testing_Alex/Analysis/README.md`](Testing_Alex/Analysis/README.md) for:
- Complete pipeline documentation
- Required folder structure
- File descriptions
- CLI options
- How to generate plots

## Key Features
- **Temporal analysis**: 6-month windows from Mar 2020 to Jan 2023
- **Inline deduplication**: Unique songs by audio features, keep max popularity
- **Cosine typicality**: kNN-based similarity measure (k=25)
- **Checkpointing**: Resume interrupted runs
- **Multiple plot styles**: Heatmaps, trends, distributions

## Repository Structure
```
team7/
├── Testing_Alex/Analysis/    # Main analysis pipeline ← START HERE
│   ├── README.md             # Full documentation
│   ├── requirements.txt      # Python dependencies
│   ├── run_temporal_analysis.py
│   └── ...
├── scripts/                  # Legacy scripts
└── notebooks/                # Exploration notebooks
```
