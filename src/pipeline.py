"""
Main orchestration pipeline for the Reeves County rig locator.

Runs the full methodology:
1. Load RRC permit data (with spud dates from the permit file itself)
2. Load Baker Hughes rig count for the analysis period
3. Cross-reference with AFE Leaks data
4. Classify well status (drilling, completing, no spud, etc.)
5. Search for Sentinel-2 imagery
6. Run change detection between before/during periods
7. Classify detected activity (drilling rig vs completion rig)
8. Validate against Baker Hughes rig count
9. Output results

Supports both February 2026 (target) and December 2025 (test case with
data actually available in the RRC system).
"""

import os
import sys
import logging
from datetime import date, timedelta

import pandas as pd
import geopandas as gpd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import ANALYSIS_YEAR, ANALYSIS_MONTH
from rrc_permits import load_permits
from baker_hughes import load_rig_counts
from afe_crossref import (
    load_afe_data, crossref_permits_with_afe, classify_well_status,
    identify_active_drilling_feb, identify_completion_activity_feb,
    identify_satellite_verification_candidates,
)
from satellite import (
    create_well_buffers, get_search_bbox, search_sentinel2_scenes,
    get_analysis_periods, create_synthetic_scene, save_scene_geotiff,
)
from change_detection import (
    detect_pad_changes, extract_well_signatures, classify_detections,
    estimate_rig_vs_completion,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    year: int = ANALYSIS_YEAR,
    month: int = ANALYSIS_MONTH,
    output_dir: str = None,
    use_sample: bool = True,
) -> dict:
    """
    Run the complete rig locator pipeline.

    Args:
        year: Analysis year
        month: Analysis month
        output_dir: Where to save outputs
        use_sample: Use sample data for methodology development
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    results = {"year": year, "month": month}

    # =====================================================================
    # STEP 1: Load RRC Permit Data
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 1: Loading RRC Permit Data - Reeves County Horizontal Wells")
    print(f"{'='*80}")

    permits = load_permits(use_sample=use_sample)
    results["total_permits"] = len(permits)

    spud_count = permits["SPUD_DATE_parsed"].notna().sum()
    no_spud = permits["SPUD_DATE_parsed"].isna().sum()
    print(f"  Total permits: {len(permits)}")
    print(f"  With spud date (from RRC): {spud_count}")
    print(f"  Without spud date: {no_spud}")

    # =====================================================================
    # STEP 2: Load Baker Hughes Rig Count
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 2: Baker Hughes Rig Count - Reeves County")
    print(f"{'='*80}")

    rig_counts = load_rig_counts(use_sample=use_sample)
    weekly_totals = rig_counts.groupby("Week")["Rig_Count"].sum()
    avg_rigs = weekly_totals.mean()
    results["avg_weekly_rigs"] = float(avg_rigs)

    print(f"  Weekly rig counts:")
    for week, count in weekly_totals.items():
        print(f"    {week}: {count} rigs")
    print(f"  Average: {avg_rigs:.0f} rigs/week")

    # =====================================================================
    # STEP 3: Cross-reference with AFE Leaks
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 3: AFE Leaks Cross-Reference")
    print(f"{'='*80}")

    afe_data = load_afe_data(use_sample=use_sample)
    merged = crossref_permits_with_afe(permits, afe_data)
    merged = classify_well_status(merged)

    results["afe_matched"] = int(merged["in_afe_leaks"].sum())
    results["afe_gaps"] = int((~merged["in_afe_leaks"]).sum())

    active_drilling = identify_active_drilling_feb(merged)
    completing = identify_completion_activity_feb(merged)
    candidates = identify_satellite_verification_candidates(merged)

    results["active_drilling_wells"] = len(active_drilling)
    results["completion_wells"] = len(completing)
    results["verification_candidates"] = len(candidates)

    print(f"  Wells with active drilling rigs: {len(active_drilling)}")
    print(f"  Wells in completion phase: {len(completing)}")
    print(f"  Satellite verification candidates: {len(candidates)}")

    # =====================================================================
    # STEP 4: Sentinel-2 Scene Search
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 4: Sentinel-2 Scene Search")
    print(f"{'='*80}")

    (before_start, before_end), (during_start, during_end) = get_analysis_periods()
    bbox = get_search_bbox(permits)

    print(f"  Before period: {before_start} to {before_end}")
    print(f"  During period: {during_start} to {during_end}")
    print(f"  Search area: {bbox}")

    before_scenes = search_sentinel2_scenes(bbox, before_start, before_end)
    during_scenes = search_sentinel2_scenes(bbox, during_start, during_end)

    results["before_scenes"] = len(before_scenes)
    results["during_scenes"] = len(during_scenes)

    print(f"  Before period: {len(before_scenes)} usable scenes")
    print(f"  During period: {len(during_scenes)} usable scenes")

    # =====================================================================
    # STEP 5: Change Detection
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 5: Change Detection Analysis")
    print(f"{'='*80}")

    # Create synthetic scenes for methodology demo
    before_scene = create_synthetic_scene(permits, before_start)
    during_scene = create_synthetic_scene(
        permits, during_start, active_wells=active_drilling
    )

    # Run change detection
    changes = detect_pad_changes(before_scene, during_scene)
    detections = extract_well_signatures(changes, permits)

    print(f"  Analyzed {len(detections)} well locations")
    print(f"  Mean NDVI change range: {detections['ndvi_change_mean'].min():.3f} to {detections['ndvi_change_mean'].max():.3f}")
    print(f"  Mean brightness change range: {detections['brightness_change_mean'].min():.3f} to {detections['brightness_change_mean'].max():.3f}")

    # =====================================================================
    # STEP 6: Classification
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 6: Activity Classification")
    print(f"{'='*80}")

    classified = classify_detections(detections, analysis_date=during_end)
    results["classifications"] = classified["activity_class"].value_counts().to_dict()

    print(f"\n  Classification summary:")
    for cls, count in classified["activity_class"].value_counts().items():
        print(f"    {cls}: {count}")

    print(f"\n  Detailed results:")
    display_cols = ["PERMIT_NO", "LEASE_NAME", "OPERATOR_NAME", "SPUD_DATE",
                    "activity_class", "confidence"]
    print(classified[display_cols].to_string(index=False))

    # =====================================================================
    # STEP 7: Validation Against Rig Count
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 7: Validation Against Baker Hughes Rig Count")
    print(f"{'='*80}")

    validation = estimate_rig_vs_completion(classified, rig_count_target=int(avg_rigs))
    results["validation"] = validation

    print(f"\n  {validation['interpretation']}")
    print(f"  Detection rate: {validation['detection_rate']:.0%}")

    # =====================================================================
    # STEP 8: Save Results
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 8: Saving Results")
    print(f"{'='*80}")

    # Save classified results
    output_csv = os.path.join(output_dir, f"reeves_county_rig_analysis_{year}_{month:02d}.csv")
    classified.drop(columns=["geometry"], errors="ignore").to_csv(output_csv, index=False)
    print(f"  Results CSV: {output_csv}")

    # Save GeoJSON for mapping
    output_geojson = os.path.join(output_dir, f"reeves_county_wells_{year}_{month:02d}.geojson")
    classified.to_file(output_geojson, driver="GeoJSON")
    print(f"  GeoJSON: {output_geojson}")

    results["output_csv"] = output_csv
    results["output_geojson"] = output_geojson

    return results


def run_december_2025_test():
    """
    Run the pipeline for December 2025 as a test case.

    December 2025 is a better test case than February 2026 because:
    - The RRC permit data is already in the system (permits filed, spud dates reported)
    - Sentinel-2 imagery from Nov/Dec 2025 is already available for download
    - Baker Hughes rig count data for Dec 2025 is published
    - We can validate results against known outcomes

    This lets us calibrate the methodology before applying it to Feb 2026.
    """
    print(f"\n{'#'*80}")
    print(f"# DECEMBER 2025 TEST CASE")
    print(f"# Using historical data to validate methodology before Feb 2026 analysis")
    print(f"{'#'*80}")

    # For the test case, we use December 2025 settings
    # In production, you'd download the actual RRC data for this period
    results = run_pipeline(year=2025, month=12)

    print(f"\n{'#'*80}")
    print(f"# TEST CASE SUMMARY")
    print(f"{'#'*80}")
    print(f"  Total permits analyzed: {results['total_permits']}")
    print(f"  Active drilling wells detected: {results['active_drilling_wells']}")
    print(f"  Baker Hughes avg rigs: {results['avg_weekly_rigs']:.0f}")
    print(f"  Detection rate: {results['validation']['detection_rate']:.0%}")
    print(f"\n  This detection rate should be validated against known outcomes")
    print(f"  for December 2025 before trusting February 2026 results.")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Reeves County Rig Locator Pipeline")
    parser.add_argument("--test", action="store_true", help="Run December 2025 test case")
    parser.add_argument("--year", type=int, default=ANALYSIS_YEAR)
    parser.add_argument("--month", type=int, default=ANALYSIS_MONTH)
    args = parser.parse_args()

    if args.test:
        run_december_2025_test()
    else:
        run_pipeline(year=args.year, month=args.month)
