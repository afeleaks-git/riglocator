# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rig locator pipeline for Reeves County, Texas (Permian Basin). Combines Texas Railroad Commission (RRC) permit data, Baker Hughes weekly rig counts, and Sentinel-2 satellite imagery to identify and verify active drilling rig locations. Well status classification relies entirely on RRC-reported spud/completion dates; satellite imagery provides independent verification.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (February 2026, default)
python src/pipeline.py

# Run December 2025 test case (recommended first - all data available)
python src/pipeline.py --test

# Run for a specific month
python src/pipeline.py --year 2025 --month 12

# Run individual modules standalone (each has __main__ blocks)
python src/rrc_permits.py
python src/baker_hughes.py
python src/satellite.py
python src/change_detection.py
python src/sentinel_hub.py
```

No test framework is configured. Validation is done by running the pipeline and comparing detected drilling rigs against Baker Hughes rig counts (Step 7 output).

## Architecture

### Pipeline Flow (src/pipeline.py)

The 8-step pipeline orchestrates all modules sequentially:

1. **rrc_permits.py** - Load/parse RRC fixed-width permit files (da800a master + da800b status/trailer). Merges spud dates from status file into permit records. Filters to Reeves County (code 389), District 08, horizontal wells. Has `create_sample_permit_data()` that generates ~96 realistic permits when real data isn't available.

2. **well_status.py** - Classifies each permit into: `SPUD_IN_PERIOD`, `DRILLING_THROUGH`, `COMPLETING`, `COMPLETED_BEFORE`, `NO_SPUD`, `SPUD_AND_COMPLETED`, `SPUD_AFTER`. Classification is purely date-based (no external data).

3. **baker_hughes.py** - Parses Baker Hughes Excel pivot tables or generates sample weekly rig counts (~18-20 rigs/week for Reeves). Used only for validation, not primary classification.

4. **satellite.py** - Searches Copernicus CDSE catalog for Sentinel-2 L2A scenes. Falls back to simulated scene catalog if API is unreachable. `create_synthetic_scene()` generates fake raster data with realistic spectral values for methodology demos.

5. **sentinel_hub.py** - Downloads real Sentinel-2 imagery via Sentinel Hub Process API (OAuth2 client credentials). Returns 4-band (R/G/B/NIR) + SCL cloud mask as numpy arrays. Pipeline falls back to synthetic scenes if this fails.

6. **change_detection.py** - Core analysis: NDVI differencing + brightness change between before/during periods. Extracts per-well signatures within buffer zones. Classifies activity type (`DRILLING_RIG`, `COMPLETION_RIG`, `PAD_CONSTRUCTION`, `PROBABLE_DRILLING`, `NO_ACTIVITY`) using spud date timing (<=35 days = drilling, >35 days = completion). Includes `deduplicate_by_pad()` to group wells within 250m into single pad detections.

### Data Flow

All modules use `use_sample=True` by default, generating synthetic data. Set `use_sample=False` and provide real files in `data/rrc/` and `data/baker_hughes/` for production runs.

Scene data flows as dicts with keys: `red`, `green`, `blue`, `nir` (numpy float32 arrays), `transform` (rasterio Affine), `crs`, `width`, `height`.

### Configuration (config/settings.py)

All thresholds, URLs, geographic bounds, and API credentials live here. Key values:
- `NDVI_CHANGE_THRESHOLD = -0.15` (vegetation loss detection)
- `BRIGHTNESS_CHANGE_THRESHOLD = 0.10` (bare earth detection)
- `BUFFER_RADIUS_M = 30` (well buffer, but `create_well_buffers` uses min 200m)
- Sentinel Hub credentials read from `SH_CLIENT_ID` / `SH_CLIENT_SECRET` env vars with hardcoded fallbacks

### Import Pattern

Modules use `sys.path.insert` to add parent directories rather than package-relative imports. When running any module standalone or from pipeline, the working directory should be the repo root or `src/`.

### Output

Results go to `data/output/`:
- `reeves_county_rig_analysis_YYYY_MM.csv` - Full results table
- `reeves_county_wells_YYYY_MM.geojson` - GeoJSON for mapping (QGIS, kepler.gl)

### Key Domain Concepts

- **Spud date**: When a well actually starts drilling (reported to RRC by operator)
- **Drilling vs completion**: Drilling rigs (single mast, first 35 days) vs frac fleets (wider footprint, after 35 days). The satellite can't distinguish them visually at 10m - spud date timing is essential.
- **Multi-well pads**: Multiple horizontal wells drilled from one surface pad. One rig services them sequentially. Deduplication prevents overcounting.
- Reeves County typically has 15-25 active rigs and 80-120+ relevant permits at any time.
