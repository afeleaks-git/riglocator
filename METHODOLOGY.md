# Reeves County Permian Basin Rig Locator

## Overview

Pipeline that uses Texas Railroad Commission (RRC) reported dates, Baker Hughes rig counts, and Sentinel-2 satellite imagery to identify and verify active drilling rig locations in Reeves County, Texas.

All well status classification comes from RRC-reported spud and completion dates. The satellite imagery provides independent verification, especially for permits with no spud date reported.

## Data Sources

### 1. Texas Railroad Commission (RRC) Permit Data
- **What**: Drilling permit master file + status/trailer file
- **Key fields**: Permit number, operator, well location (lat/lon), well direction, **spud date**, completion date
- **The spud date is in the permit data itself** - operators report spud dates to the RRC
- **Download**: https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/
- **Format**: Fixed-width text files in ZIP archives (record layout PDFs provided)
- **Filter**: District 08, County Code 389 (Reeves), Well Direction = H (Horizontal)
- **Important**: Pull ALL permits for the county where spud >= period start, OR no spud date, OR spud before period with no completion yet. Not just permits filed in the analysis month.

### 2. Baker Hughes Rig Count
- **What**: Weekly rig count by county/basin/operator
- **Download**: Excel pivot table from https://rigcount.bakerhughes.com/na-rig-count
- **Current**: Reeves County typically runs 15-25 rigs (~44% of Permian Basin activity)
- **No API** - must download the Excel file manually
- Used as a validation check, not a primary data source

### 3. Sentinel-2 Satellite Imagery
- **Why Sentinel-2 over Landsat**: 10m resolution vs 30m. A 150m drill pad = 15x15 pixels (Sentinel-2) vs 5x5 pixels (Landsat). 5-day revisit vs 16-day.
- **Source**: Copernicus Data Space Ecosystem (free, requires registration)
- **API**: OData catalog at https://catalogue.dataspace.copernicus.eu/odata/v1
- **Bands used**: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR) - all at 10m
- **Tile**: T13SDA covers most of Reeves County

## Methodology

### Step 1: Load All Relevant Permits
1. Download RRC permit master + status files for District 08
2. Filter to Reeves County horizontal wells
3. Spud dates come from the RRC permit status/trailer records
4. Keep all permits where: spud >= period start OR no spud date OR still drilling (no completion)
5. Parse lat/lon for surface hole locations
6. Reeves County will have ~80-120+ relevant horizontal permits at any time

### Step 2: Classify Well Status from Reported Dates
Using only RRC-reported dates, classify each well:
- `SPUD_IN_PERIOD`: Started drilling during the analysis month
- `DRILLING_THROUGH`: Spud before the period, no completion yet (still drilling)
- `COMPLETING`: Drilling done, completion date falls in the period
- `NO_SPUD`: Permitted but no spud date reported
- `COMPLETED_BEFORE`: Already done before our window

### Step 3: Baker Hughes Validation Check
Load the weekly rig count for Reeves County. Compare the number of wells classified as actively drilling (SPUD_IN_PERIOD + DRILLING_THROUGH) against Baker Hughes. If RRC says 32 wells are drilling and Baker Hughes says 18 rigs, the gap tells you something - some wells may have already released their rig, or the timing window is fuzzy.

### Step 4: Satellite Imagery Acquisition
1. Define before period (prior month) and during period (analysis month)
2. Query CDSE for Sentinel-2 L2A scenes with <20% cloud cover
3. West Texas is favorable - most scenes are usable
4. Download B04, B03, B02, B08 bands for the area of interest

### Step 5: Change Detection
For each well location with a 200m buffer:

1. **NDVI Differencing**: Compute NDVI = (NIR - Red) / (NIR + Red) for before and during periods. NDVI drop > 0.15 indicates vegetation clearance (pad construction).

2. **Brightness Change**: Average visible brightness increase > 0.10 indicates newly exposed bare earth or equipment.

3. **Combined Score**: 60% NDVI change weight + 40% brightness change weight.

### Step 6: Classification (Drilling vs Completion)
This is the critical distinction:

| Signal | Drilling Rig | Completion Rig |
|--------|-------------|----------------|
| Timeline | 0-35 days after spud | >35 days after spud |
| Footprint | ~150m pad, single mast | Wider spread (frac fleet) |
| Duration | 15-30 days typical | 15-25 days typical |

- **HIGH confidence DRILLING_RIG**: Change detected + spud date within last 35 days
- **HIGH confidence COMPLETION_RIG**: Change detected + spud date >35 days ago
- **MEDIUM confidence**: Change detected, no spud date to confirm phase
- **PAD_CONSTRUCTION**: Change detected at unspud location (pad being built)

### Step 7: Validation
Compare detected drilling rigs against Baker Hughes weekly count. Large gaps indicate:
- Some wells classified as "drilling" may have already released their rig
- Rigs on wells permitted in adjacent counties (not in our dataset)
- Threshold calibration needed

## Key Limitations

1. **Cannot see the rig itself at 10m resolution** - we detect the pad and activity pattern, not the physical rig structure
2. **Completion activity looks like drilling activity** from space - the spud date timing is essential to distinguish them
3. **Cloud cover** can block views, though West Texas in winter is generally clear
4. **New pad construction** without drilling yet produces the same NDVI/brightness change as active drilling
5. **Multi-well pads** with simultaneous drilling complicate per-well attribution
6. **RRC reporting lag** - spud dates may not be reported immediately, so some "NO_SPUD" wells may actually be drilling

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run February 2026 analysis
python src/pipeline.py

# Run December 2025 test case (recommended first - data is available)
python src/pipeline.py --test

# Run for any month
python src/pipeline.py --year 2025 --month 12
```

## December 2025 Test Case

Running December 2025 first is recommended because:
- RRC permit data for Dec 2025 is already filed and in the system
- Sentinel-2 imagery from Nov-Dec 2025 is already available
- Baker Hughes rig counts for Dec 2025 are published
- You can validate against known outcomes before trusting Feb 2026 results

## Production Use

To use with real data instead of samples:

1. **RRC data**: Download permit files from RRC, place in `data/rrc/`
2. **Baker Hughes**: Download pivot table Excel, place in `data/baker_hughes/`
3. **Sentinel-2**: Register at https://dataspace.copernicus.eu/, the pipeline will query the catalog API

Then set `use_sample=False` in the pipeline call.

## Output Files

- `data/output/reeves_county_rig_analysis_YYYY_MM.csv` - Full results table
- `data/output/reeves_county_wells_YYYY_MM.geojson` - GeoJSON for mapping (load in QGIS, kepler.gl, etc.)
