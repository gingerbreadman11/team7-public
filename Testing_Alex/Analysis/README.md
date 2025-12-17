# Spotify Typicality Analysis Pipeline

Analyze the relationship between musical typicality and track popularity using Spotify data across temporal windows.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main temporal analysis (6-month windows)
python run_temporal_analysis.py \
    --data-path /path/to/Data_Spotify \
    --output-path ./results_temporal \
    --sample-size 500000 \
    --folder-stride 3 \
    --checkpoint-interval 10
```

---

## Required Folder Structure

```
Data_Spotify/                          # Daily HDF data (input)
├── 2020-03-01/
│   └── 2020-03-01_playlist_track_info_1.hdf
├── 2020-03-02/
│   └── ...
└── ...

album and track files/                 # Audio features (input)
└── tracks/
    └── csvs for paper sample/
        ├── track_features_1.csv
        ├── track_features_2.csv
        └── ...

results_temporal/                      # Output (auto-created)
├── progress.json
├── period_2020-03_2020-08/
│   ├── checkpoint.pkl
│   ├── sample_dedup.csv
│   ├── sample_with_typicality.csv
│   └── period_stats.json
├── period_2020-09_2021-02/
│   └── ...
└── plots_*/
```

---

## Core Pipeline Files

| File | Purpose |
|------|---------|
| `run_temporal_analysis.py` | **Main entry point.** Runs 6-month window analysis with deduplication. |
| `temporal_streaming_stats.py` | Deduplicating sampler + cosine kNN typicality computation. |
| `track_features_lookup.py` | Builds/loads audio feature lookup dictionary from CSVs. |
| `streaming_stats.py` | Incremental correlation + reservoir sampling utilities. |

---

## Plotting Scripts

| File | What it generates |
|------|-------------------|
| `plot_temporal_beautiful.py` | Heatmaps (green-blue), trend overlays, 6-panel comparisons. |
| `plot_temporal_typicality.py` | Per-period hexbin plots, summary metrics. |
| `plot_release_year_analysis.py` | Typicality by release year (from ISRC), feature evolution. |
| `plot_additional_figures.py` | Distributions, feature importance with typicality. |
| `plot_alternative_typicality.py` | Compare standard vs weighted vs temporal typicality. |
| `visualize_results.py` | Correlation heatmaps, bar charts, distributions. |

---

## Alternative Analyses

| File | Purpose |
|------|---------|
| `compute_alternative_typicality.py` | Popularity-weighted & temporal-window typicality. |
| `run_analysis.py` | Older single-run pipeline (non-temporal). |
| `popularity_model.py` | Train RF/GB models predicting popularity. |

---

## How the Pipeline Works

1. **Load audio features** → `track_features_lookup.pkl` (O(1) lookup by track.id)
2. **Stream HDF files** → daily playlist data with track.id + popularity
3. **Join** → match track.id to audio features
4. **Deduplicate** → by audio feature tuple, keep max popularity
5. **Compute typicality** → cosine kNN (k=25) on standardized features
6. **Save per-window** → CSV with typicality, checkpoints, stats

### Deduplication Logic
- Songs are keyed by their **12 audio feature values** (rounded tuple)
- If the same song appears multiple times (different IDs), we keep **max popularity**
- Cache fills to `sample_size` unique songs, then stops adding new songs

### Typicality Calculation
```
For each song:
  1. Standardize 12 audio features
  2. Find k=25 nearest neighbors (cosine distance)
  3. typicality = mean(cosine_similarity to neighbors)
  4. Normalize to [0, 1]
```

---

## CLI Options

```
--data-path          Path to Data_Spotify folder (daily HDFs)
--output-path        Where to save results (default: ./results_temporal)
--sample-size        Max unique songs per period (default: 500000)
--folder-stride      Process every Nth folder (default: 3)
--checkpoint-interval Save checkpoint every N folders (default: 10)
--k-neighbors        k for typicality kNN (default: 25)
--force-rebuild-lookup  Rebuild track_features_lookup.pkl
```

---

## Generate Plots (after pipeline completes)

```bash
# Beautiful heatmaps + trends
python plot_temporal_beautiful.py

# Release year analysis (uses ISRC to infer year)
python plot_release_year_analysis.py

# Feature importance including typicality
python plot_additional_figures.py

# Alternative typicality comparisons
python compute_alternative_typicality.py
python plot_alternative_typicality.py
```

---

## File Descriptions

### Core
- **`run_temporal_analysis.py`** — Orchestrates 6 time windows, calls pipeline per window
- **`temporal_streaming_stats.py`** — `DeduplicatingReservoirSampler` class, `compute_typicality_cosine()`
- **`track_features_lookup.py`** — Load CSVs → deduplicate by ISRC → build lookup dict
- **`streaming_stats.py`** — `IncrementalCorrelation`, `ReservoirSampler`, `StreamingStats`
- **`popularity_analysis_streaming.py`** — `PopularityAnalysisPipeline` (older non-temporal)

### Plotting
- **`plot_temporal_beautiful.py`** — Green-blue heatmaps, GP-style/polynomial/zigzag trends
- **`plot_temporal_typicality.py`** — 6-panel evolution, overlaid trends, summary bars
- **`plot_release_year_analysis.py`** — Typicality by release year, feature evolution, era comparisons
- **`plot_additional_figures.py`** — Typicality/popularity distributions, feature importance with typicality
- **`plot_alternative_typicality.py`** — Compare 3 typicality measures
- **`visualize_results.py`** — General correlation/distribution plots

### Alternative Analysis
- **`compute_alternative_typicality.py`** — Popularity-weighted typicality, temporal-window typicality
- **`popularity_model.py`** — Train Random Forest / Gradient Boosting on features + typicality
- **`compute_typicality.py`** — Euclidean-distance typicality (older)
- **`compute_typicality_cosine.py`** — Standalone cosine typicality script
- **`dedup_and_typicality.py`** — Post-hoc deduplication + typicality (older approach)

### Utility / Legacy
- **`run_analysis.py`** — Single-run entry point (non-temporal)
- **`enrich_data.py`**, **`data_exploration.py`** — Data exploration utilities
- **`check_*.py`**, **`comparison_*.py`** — Exploratory / debugging scripts

---

## Output Files

Each period folder contains:
- `sample_dedup.csv` — Deduplicated songs (no typicality yet)
- `sample_with_typicality.csv` — Final output with `typicality_cosine_norm`
- `period_stats.json` — Summary statistics
- `checkpoint.pkl` — Resume state
- `processing_log.txt` — Timestamped logs

