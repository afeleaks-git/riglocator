"""
AFE Leaks cross-reference module.

Cross-references RRC permit data (which already contains spud dates from
the permit status/trailer records) with AFE Leaks datasets to:
1. Enrich permits with AFE cost data and any additional status info
2. Identify wells AFE Leaks is tracking vs gaps in coverage
3. Flag wells where the RRC has a spud date but AFE Leaks doesn't (or vice versa)

The RRC permit file itself is the primary source for spud dates - operators
report spud dates to the Railroad Commission. AFE Leaks provides the
financial/cost side (AFE amounts, cost breakdowns) that the RRC doesn't have.
"""

import os
import logging
from datetime import datetime

import pandas as pd
import geopandas as gpd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ANALYSIS_YEAR, ANALYSIS_MONTH

logger = logging.getLogger(__name__)


def create_sample_afe_data() -> pd.DataFrame:
    """
    Create representative AFE Leaks data for cross-referencing.

    AFE Leaks tracks the financial side - AFE filings, cost breakdowns,
    operator spending patterns. The RRC permit data has the spud dates.
    The value of cross-referencing is:
    - Matching AFE cost data to known drilling activity
    - Finding wells AFE Leaks doesn't have coverage for yet
    - Identifying cost patterns for wells at different drilling stages
    """
    records = [
        # Wells AFE Leaks has cost data for
        {"api_no": "38938901", "operator": "DIAMONDBACK ENERGY",
         "lease_name": "UNIVERSITY LANDS 45-08", "well_no": "1H",
         "afe_amount": 8500000, "afe_status": "CLOSED",
         "cost_per_foot": 425, "target_formation": "WOLFCAMP A"},

        {"api_no": "38938902", "operator": "APACHE CORP",
         "lease_name": "STATE ANTELOPE 22-15", "well_no": "2H",
         "afe_amount": 9300000, "afe_status": "ACTIVE",
         "cost_per_foot": 465, "target_formation": "BONE SPRING"},

        {"api_no": "38938904", "operator": "OXY USA",
         "lease_name": "PECOS VALLEY 31-42", "well_no": "1AH",
         "afe_amount": 9200000, "afe_status": "ACTIVE",
         "cost_per_foot": 460, "target_formation": "WOLFCAMP A"},

        {"api_no": "38938907", "operator": "DIAMONDBACK ENERGY",
         "lease_name": "DELAWARE MOUNTAIN A", "well_no": "5H",
         "afe_amount": 8800000, "afe_status": "ACTIVE",
         "cost_per_foot": 440, "target_formation": "WOLFCAMP B"},

        {"api_no": "38938908", "operator": "DEVON ENERGY",
         "lease_name": "WOLFCAMP STATE 44-05", "well_no": "1H",
         "afe_amount": 9500000, "afe_status": "ACTIVE",
         "cost_per_foot": 475, "target_formation": "WOLFCAMP A"},

        {"api_no": "38938911", "operator": "PIONEER NATURAL RES",
         "lease_name": "RATTLESNAKE UNIT B", "well_no": "6H",
         "afe_amount": 8900000, "afe_status": "ACTIVE",
         "cost_per_foot": 445, "target_formation": "BONE SPRING"},

        {"api_no": "38938913", "operator": "CHEVRON USA",
         "lease_name": "SOUTH PECOS UNIT", "well_no": "4AH",
         "afe_amount": 10100000, "afe_status": "ACTIVE",
         "cost_per_foot": 505, "target_formation": "WOLFCAMP A"},

        {"api_no": "38938905", "operator": "CENTENNIAL RESOURCE",
         "lease_name": "RED HILLS STATE", "well_no": "4H",
         "afe_amount": 8700000, "afe_status": "PENDING",
         "cost_per_foot": 435, "target_formation": "WOLFCAMP B"},

        {"api_no": "38938914", "operator": "OXY USA",
         "lease_name": "BALMORHEA STATE 12", "well_no": "2H",
         "afe_amount": 8600000, "afe_status": "PENDING",
         "cost_per_foot": 430, "target_formation": "BONE SPRING"},
    ]

    return pd.DataFrame(records)


def crossref_permits_with_afe(
    permits: gpd.GeoDataFrame,
    afe_data: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Cross-reference RRC permits with AFE Leaks data.

    The permits GeoDataFrame already has spud dates from the RRC data.
    We join on API number to bring in AFE cost data.
    """
    merged = permits.merge(
        afe_data[["api_no", "afe_amount", "afe_status", "cost_per_foot", "target_formation"]],
        left_on="API_NO",
        right_on="api_no",
        how="left",
    )

    # Wells not in AFE data
    merged["in_afe_leaks"] = merged["api_no"].notna()

    logger.info(
        f"Cross-reference results:\n"
        f"  Total permits: {len(merged)}\n"
        f"  In AFE Leaks: {merged['in_afe_leaks'].sum()}\n"
        f"  NOT in AFE Leaks: {(~merged['in_afe_leaks']).sum()} (coverage gap)\n"
        f"  With spud date (from RRC): {merged['SPUD_DATE_parsed'].notna().sum()}\n"
        f"  Without spud date: {merged['SPUD_DATE_parsed'].isna().sum()}"
    )

    return merged


def classify_well_status(merged: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Classify each well's drilling status based on RRC spud/completion dates.

    Uses the RRC permit data (the authoritative source for spud dates) to
    determine what was happening at each well during February 2026.
    """
    feb_start = pd.Timestamp(ANALYSIS_YEAR, ANALYSIS_MONTH, 1)
    feb_end = pd.Timestamp(ANALYSIS_YEAR, ANALYSIS_MONTH + 1, 1)

    def get_status(row):
        spud = row.get("SPUD_DATE_parsed")
        comp = row.get("COMPLETION_DATE_parsed")

        if pd.isna(spud):
            return "NO_SPUD"  # Permitted but no drilling reported

        if spud >= feb_start and spud < feb_end:
            if pd.notna(comp) and comp < feb_end:
                return "SPUD_AND_COMPLETED_FEB"
            return "SPUD_IN_FEB"  # Started drilling in February

        if spud < feb_start:
            if pd.isna(comp):
                return "DRILLING_THROUGH_FEB"  # Started before, still going
            if comp >= feb_start:
                return "COMPLETING_IN_FEB"  # Drilling done, completing
            return "COMPLETED_BEFORE_FEB"  # Done before our period

        return "SPUD_AFTER_FEB"  # Not yet started as of Feb

    merged = merged.copy()
    merged["feb_status"] = merged.apply(get_status, axis=1)

    status_counts = merged["feb_status"].value_counts()
    logger.info(f"Well status classification:\n{status_counts.to_string()}")

    return merged


def identify_active_drilling_feb(merged: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Get wells that had active drilling rigs during February 2026.

    These are the wells where we expect to see a drilling rig in satellite imagery:
    - SPUD_IN_FEB: Started drilling in February
    - DRILLING_THROUGH_FEB: Started before Feb, still drilling
    """
    active = merged[merged["feb_status"].isin(["SPUD_IN_FEB", "DRILLING_THROUGH_FEB"])].copy()
    logger.info(f"Wells with active drilling rigs in Feb: {len(active)}")
    return active


def identify_completion_activity_feb(merged: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Get wells in completion phase during February.

    Important for the methodology: completion rigs look different from drilling
    rigs in satellite imagery. Completion activity involves frac spreads,
    pump trucks, sand storage - a wider footprint but different equipment.
    """
    completing = merged[merged["feb_status"].isin(["COMPLETING_IN_FEB", "SPUD_AND_COMPLETED_FEB"])].copy()
    logger.info(f"Wells with completion activity in Feb: {len(completing)}")
    return completing


def identify_satellite_verification_candidates(merged: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Get wells that are candidates for satellite verification.

    Two categories:
    1. NO_SPUD: Permitted but no spud date - satellite can check if pad is built
    2. Wells not in AFE Leaks: Coverage gaps worth investigating
    """
    candidates = merged[
        (merged["feb_status"] == "NO_SPUD") | (~merged["in_afe_leaks"])
    ].copy()
    logger.info(f"Satellite verification candidates: {len(candidates)}")
    return candidates


def load_afe_data(data_dir: str = None, use_sample: bool = True) -> pd.DataFrame:
    """Load AFE Leaks data."""
    if data_dir:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if csv_files:
            filepath = os.path.join(data_dir, csv_files[0])
            logger.info(f"Loading AFE data: {filepath}")
            return pd.read_csv(filepath)

    if use_sample:
        logger.info("Using sample AFE data for methodology development")
        return create_sample_afe_data()

    raise FileNotFoundError("No AFE data found and sample data disabled")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from rrc_permits import load_permits

    permits = load_permits()
    afe_data = load_afe_data()

    merged = crossref_permits_with_afe(permits, afe_data)
    merged = classify_well_status(merged)

    print(f"\n{'='*80}")
    print(f"AFE Leaks Cross-Reference Results")
    print(f"{'='*80}")

    active = identify_active_drilling_feb(merged)
    print(f"\nWells with DRILLING RIGS in February {ANALYSIS_YEAR}:")
    print(active[["PERMIT_NO", "LEASE_NAME", "OPERATOR_NAME", "SPUD_DATE",
                   "lat", "lon", "afe_amount", "feb_status"]].to_string(index=False))

    completing = identify_completion_activity_feb(merged)
    if len(completing) > 0:
        print(f"\nWells with COMPLETION activity in February (caution - not drilling rigs):")
        print(completing[["PERMIT_NO", "LEASE_NAME", "OPERATOR_NAME",
                          "SPUD_DATE", "COMPLETION_DATE"]].to_string(index=False))

    candidates = identify_satellite_verification_candidates(merged)
    print(f"\nSatellite verification candidates (no spud or not in AFE Leaks):")
    print(candidates[["PERMIT_NO", "LEASE_NAME", "OPERATOR_NAME", "lat", "lon",
                       "feb_status", "in_afe_leaks"]].to_string(index=False))
