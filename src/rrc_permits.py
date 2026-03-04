"""
Texas Railroad Commission (RRC) drilling permit data handler.

The RRC publishes several relevant datasets for download:
- Drilling Permit Master File (da800a): Contains permit details including
  operator, lease, well number, county, district, lat/long, well direction
- Drilling Permit Status File: Permit approval status and dates
- Completed Wells / W-2 data: Completion reports with spud dates

Data source: https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/

The drilling permit master file is a fixed-width text file. We parse it to extract
horizontal well permits in Reeves County with their surface hole locations.
"""

import os
import io
import zipfile
import logging
from datetime import datetime, date

import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    COUNTY, RRC_DISTRICT, WELL_DIRECTION,
    ANALYSIS_YEAR, ANALYSIS_MONTH, REEVES_COUNTY_BBOX,
)

logger = logging.getLogger(__name__)


# RRC county codes - Reeves County is 389
RRC_COUNTY_CODES = {
    "REEVES": "389",
}

# The RRC drilling permit master file (da800a) layout
# This is a fixed-width format. Key fields:
DA800A_COLUMNS = {
    "PERMIT_NO": (0, 6),
    "PERMIT_SEQUENCE": (6, 8),
    "DISTRICT": (8, 10),
    "COUNTY_CODE": (10, 13),
    "LEASE_NAME": (13, 45),
    "WELL_NO": (45, 51),
    "OPERATOR_NO": (51, 57),
    "OPERATOR_NAME": (57, 89),
    "FILING_PURPOSE": (89, 91),
    "PERMIT_APPROVAL_DATE": (91, 99),  # YYYYMMDD
    "WELL_DIRECTION": (99, 100),  # V=Vertical, H=Horizontal, D=Directional
    "TOTAL_DEPTH": (100, 105),
    "SURFACE_LATITUDE": (105, 115),
    "SURFACE_LONGITUDE": (115, 126),
    "BH_LATITUDE": (126, 136),  # Bottom hole
    "BH_LONGITUDE": (136, 147),
    "API_NO": (147, 155),
    "FIELD_NO": (155, 163),
    "FIELD_NAME": (163, 195),
}

# The RRC also publishes permit status / completion data that includes spud dates.
# The drilling permit status trailer file (da800b) has additional date fields:
DA800B_COLUMNS = {
    "PERMIT_NO": (0, 6),
    "PERMIT_SEQUENCE": (6, 8),
    "SPUD_DATE": (8, 16),       # YYYYMMDD - actual spud date if drilling has started
    "COMPLETION_DATE": (16, 24), # YYYYMMDD - from W-2 completion report
    "STATUS_CODE": (24, 26),     # Current permit status
    "TOTAL_DEPTH_ACTUAL": (26, 31),
}


def parse_rrc_permit_file(filepath: str) -> pd.DataFrame:
    """
    Parse the RRC drilling permit master fixed-width file.

    The RRC provides data in fixed-width text format. Each record contains
    permit details, well location, operator info, and well type.
    """
    records = []

    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            if len(line.strip()) < 100:
                continue
            record = {}
            for col_name, (start, end) in DA800A_COLUMNS.items():
                try:
                    record[col_name] = line[start:end].strip()
                except IndexError:
                    record[col_name] = ""
            records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} total permit records from {filepath}")
    return df


def parse_rrc_permit_status_file(filepath: str) -> pd.DataFrame:
    """
    Parse the RRC drilling permit status/trailer file (da800b).

    This file contains the spud date, completion date, and status code
    for permits. The spud date comes directly from the RRC records -
    operators report when they actually begin drilling (spud the well).
    """
    records = []

    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            if len(line.strip()) < 20:
                continue
            record = {}
            for col_name, (start, end) in DA800B_COLUMNS.items():
                try:
                    record[col_name] = line[start:end].strip()
                except IndexError:
                    record[col_name] = ""
            records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} permit status records from {filepath}")
    return df


def merge_permit_with_status(permits: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """
    Merge permit master records with status/trailer records to get spud dates.

    The RRC permit file itself may contain the spud date in the trailer records.
    This merges on PERMIT_NO + PERMIT_SEQUENCE to bring spud_date and
    completion_date into the permit records.
    """
    merged = permits.merge(
        status[["PERMIT_NO", "PERMIT_SEQUENCE", "SPUD_DATE", "COMPLETION_DATE", "STATUS_CODE"]],
        on=["PERMIT_NO", "PERMIT_SEQUENCE"],
        how="left",
    )

    # Parse date fields
    for col in ["SPUD_DATE", "COMPLETION_DATE"]:
        merged[col + "_parsed"] = pd.to_datetime(merged[col], format="%Y%m%d", errors="coerce")

    spud_count = merged["SPUD_DATE_parsed"].notna().sum()
    logger.info(f"Merged status data: {spud_count}/{len(merged)} permits have spud dates")
    return merged


def filter_reeves_county_horizontal(df: pd.DataFrame, year: int = None, month: int = None) -> pd.DataFrame:
    """
    Filter permits to Reeves County horizontal wells, optionally for a specific period.
    """
    county_code = RRC_COUNTY_CODES.get(COUNTY, "389")

    # Filter by county and well direction
    mask = (
        (df["COUNTY_CODE"] == county_code)
        & (df["WELL_DIRECTION"] == WELL_DIRECTION)
        & (df["DISTRICT"] == RRC_DISTRICT)
    )

    if year and month:
        # Filter by permit approval date
        def in_period(date_str):
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                return dt.year == year and dt.month == month
            except (ValueError, TypeError):
                return False

        mask = mask & df["PERMIT_APPROVAL_DATE"].apply(in_period)

    filtered = df[mask].copy()
    logger.info(
        f"Filtered to {len(filtered)} Reeves County horizontal permits"
        + (f" for {year}-{month:02d}" if year and month else "")
    )
    return filtered


def parse_coordinates(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert lat/long strings to numeric and create a GeoDataFrame.

    RRC coordinates are in decimal degrees (WGS84). Some records may have
    missing or invalid coordinates which we drop.
    """
    df = df.copy()

    df["lat"] = pd.to_numeric(df["SURFACE_LATITUDE"], errors="coerce")
    df["lon"] = pd.to_numeric(df["SURFACE_LONGITUDE"], errors="coerce")

    # RRC longitudes for Texas should be negative (western hemisphere)
    # Some files store them as positive; correct if needed
    df.loc[df["lon"] > 0, "lon"] = -df["lon"]

    # Validate coordinates are in Reeves County area
    valid = (
        (df["lat"].between(REEVES_COUNTY_BBOX["south"] - 0.1, REEVES_COUNTY_BBOX["north"] + 0.1))
        & (df["lon"].between(REEVES_COUNTY_BBOX["west"] - 0.1, REEVES_COUNTY_BBOX["east"] + 0.1))
    )

    df = df[valid & df["lat"].notna() & df["lon"].notna()].copy()

    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    logger.info(f"Created GeoDataFrame with {len(gdf)} geocoded permits")
    return gdf


def create_sample_permit_data() -> gpd.GeoDataFrame:
    """
    Create representative sample data based on known Reeves County permit patterns.

    This generates realistic permit records for methodology development when
    the actual RRC download isn't available. Based on public RRC data patterns
    for the Delaware Basin / Reeves County area.

    Real Reeves County horizontal wells cluster in several areas:
    - Northern Reeves (near Mentone): ~31.4°N, -103.6°W
    - Central Reeves (Pecos area): ~31.2°N, -103.5°W
    - Southern Reeves: ~31.0°N, -103.7°W
    """
    # Representative well locations based on typical Reeves County drilling areas.
    # Spud dates come from the RRC permit file itself (status/trailer records).
    # SPUD_DATE = None means the permit exists but no spud has been reported yet.
    wells = [
        # Northern Reeves - heavy Delaware Basin activity
        {"PERMIT_NO": "900001", "LEASE_NAME": "UNIVERSITY LANDS 45-08", "WELL_NO": "1H",
         "OPERATOR_NAME": "DIAMONDBACK ENERGY", "lat": 31.45, "lon": -103.58,
         "PERMIT_APPROVAL_DATE": "20260115", "API_NO": "38938901",
         "SPUD_DATE": "20260120", "COMPLETION_DATE": "20260228"},  # Spud Jan, completed Feb
        {"PERMIT_NO": "900002", "LEASE_NAME": "STATE ANTELOPE 22-15", "WELL_NO": "2H",
         "OPERATOR_NAME": "APACHE CORP", "lat": 31.42, "lon": -103.62,
         "PERMIT_APPROVAL_DATE": "20260118", "API_NO": "38938902",
         "SPUD_DATE": "20260125", "COMPLETION_DATE": None},  # Spud Jan, still drilling in Feb
        {"PERMIT_NO": "900003", "LEASE_NAME": "SIDEWINDER UNIT A", "WELL_NO": "3H",
         "OPERATOR_NAME": "CONOCOPHILLIPS", "lat": 31.48, "lon": -103.55,
         "PERMIT_APPROVAL_DATE": "20260120", "API_NO": "38938903",
         "SPUD_DATE": None, "COMPLETION_DATE": None},  # Permitted, no spud yet

        # Central Reeves - near Pecos
        {"PERMIT_NO": "900004", "LEASE_NAME": "PECOS VALLEY 31-42", "WELL_NO": "1AH",
         "OPERATOR_NAME": "OXY USA", "lat": 31.22, "lon": -103.50,
         "PERMIT_APPROVAL_DATE": "20260125", "API_NO": "38938904",
         "SPUD_DATE": "20260201", "COMPLETION_DATE": None},  # Spud Feb 1, actively drilling
        {"PERMIT_NO": "900005", "LEASE_NAME": "RED HILLS STATE", "WELL_NO": "4H",
         "OPERATOR_NAME": "CENTENNIAL RESOURCE", "lat": 31.25, "lon": -103.48,
         "PERMIT_APPROVAL_DATE": "20260128", "API_NO": "38938905",
         "SPUD_DATE": None, "COMPLETION_DATE": None},  # Permitted, no spud
        {"PERMIT_NO": "900006", "LEASE_NAME": "MUSTANG SPRINGS 15-22", "WELL_NO": "2H",
         "OPERATOR_NAME": "RING ENERGY", "lat": 31.18, "lon": -103.53,
         "PERMIT_APPROVAL_DATE": "20260201", "API_NO": "38938906",
         "SPUD_DATE": "20260205", "COMPLETION_DATE": None},  # Spud Feb 5

        # Active Feb drilling cluster
        {"PERMIT_NO": "900007", "LEASE_NAME": "DELAWARE MOUNTAIN A", "WELL_NO": "5H",
         "OPERATOR_NAME": "DIAMONDBACK ENERGY", "lat": 31.35, "lon": -103.65,
         "PERMIT_APPROVAL_DATE": "20260203", "API_NO": "38938907",
         "SPUD_DATE": "20260205", "COMPLETION_DATE": None},  # Spud Feb 5
        {"PERMIT_NO": "900008", "LEASE_NAME": "WOLFCAMP STATE 44-05", "WELL_NO": "1H",
         "OPERATOR_NAME": "DEVON ENERGY", "lat": 31.38, "lon": -103.60,
         "PERMIT_APPROVAL_DATE": "20260205", "API_NO": "38938908",
         "SPUD_DATE": "20260208", "COMPLETION_DATE": None},  # Spud Feb 8
        {"PERMIT_NO": "900009", "LEASE_NAME": "BONE SPRING RANCH", "WELL_NO": "3AH",
         "OPERATOR_NAME": "EOG RESOURCES", "lat": 31.30, "lon": -103.70,
         "PERMIT_APPROVAL_DATE": "20260208", "API_NO": "38938909",
         "SPUD_DATE": "20260210", "COMPLETION_DATE": None},  # Spud Feb 10
        {"PERMIT_NO": "900010", "LEASE_NAME": "GUADALUPE PASS 18-07", "WELL_NO": "2H",
         "OPERATOR_NAME": "APACHE CORP", "lat": 31.33, "lon": -103.55,
         "PERMIT_APPROVAL_DATE": "20260210", "API_NO": "38938910",
         "SPUD_DATE": None, "COMPLETION_DATE": None},  # Permitted, no spud
        {"PERMIT_NO": "900011", "LEASE_NAME": "RATTLESNAKE UNIT B", "WELL_NO": "6H",
         "OPERATOR_NAME": "PIONEER NATURAL RES", "lat": 31.40, "lon": -103.57,
         "PERMIT_APPROVAL_DATE": "20260212", "API_NO": "38938911",
         "SPUD_DATE": "20260214", "COMPLETION_DATE": None},  # Spud Feb 14
        {"PERMIT_NO": "900012", "LEASE_NAME": "TOYAH CREEK 28-33", "WELL_NO": "1H",
         "OPERATOR_NAME": "CONOCOPHILLIPS", "lat": 31.28, "lon": -103.63,
         "PERMIT_APPROVAL_DATE": "20260215", "API_NO": "38938912",
         "SPUD_DATE": "20260218", "COMPLETION_DATE": None},  # Spud Feb 18

        # Southern Reeves
        {"PERMIT_NO": "900013", "LEASE_NAME": "SOUTH PECOS UNIT", "WELL_NO": "4AH",
         "OPERATOR_NAME": "CHEVRON USA", "lat": 31.05, "lon": -103.72,
         "PERMIT_APPROVAL_DATE": "20260218", "API_NO": "38938913",
         "SPUD_DATE": "20260220", "COMPLETION_DATE": None},  # Spud Feb 20
        {"PERMIT_NO": "900014", "LEASE_NAME": "BALMORHEA STATE 12", "WELL_NO": "2H",
         "OPERATOR_NAME": "OXY USA", "lat": 31.02, "lon": -103.68,
         "PERMIT_APPROVAL_DATE": "20260220", "API_NO": "38938914",
         "SPUD_DATE": None, "COMPLETION_DATE": None},  # Permitted, no spud
        {"PERMIT_NO": "900015", "LEASE_NAME": "PHANTOM RANCH 41-04", "WELL_NO": "7H",
         "OPERATOR_NAME": "DIAMONDBACK ENERGY", "lat": 31.10, "lon": -103.75,
         "PERMIT_APPROVAL_DATE": "20260222", "API_NO": "38938915",
         "SPUD_DATE": "20260224", "COMPLETION_DATE": None},  # Spud Feb 24
    ]

    df = pd.DataFrame(wells)
    df["DISTRICT"] = RRC_DISTRICT
    df["COUNTY_CODE"] = RRC_COUNTY_CODES[COUNTY]
    df["WELL_DIRECTION"] = WELL_DIRECTION
    df["PERMIT_SEQUENCE"] = "00"
    df["TOTAL_DEPTH"] = "20000"
    df["OPERATOR_NO"] = ""
    df["FILING_PURPOSE"] = "W1"
    df["FIELD_NO"] = ""
    df["FIELD_NAME"] = "DELAWARE BASIN"
    df["SURFACE_LATITUDE"] = df["lat"].astype(str)
    df["SURFACE_LONGITUDE"] = df["lon"].astype(str)
    df["BH_LATITUDE"] = ""
    df["BH_LONGITUDE"] = ""

    # Parse spud/completion dates from the permit data itself
    df["SPUD_DATE_parsed"] = pd.to_datetime(df["SPUD_DATE"], format="%Y%m%d", errors="coerce")
    df["COMPLETION_DATE_parsed"] = pd.to_datetime(df["COMPLETION_DATE"], format="%Y%m%d", errors="coerce")

    geometry = [Point(row["lon"], row["lat"]) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    spud_count = gdf["SPUD_DATE_parsed"].notna().sum()
    no_spud = gdf["SPUD_DATE_parsed"].isna().sum()
    logger.info(
        f"Created sample dataset: {len(gdf)} permits, "
        f"{spud_count} with spud dates, {no_spud} without"
    )
    return gdf


def load_permits(data_dir: str = None, use_sample: bool = True) -> gpd.GeoDataFrame:
    """
    Load RRC permit data. Uses sample data for methodology development,
    or parses downloaded RRC files if available.

    If a status/trailer file exists alongside the permit master file,
    it will be merged to bring in spud dates and completion dates
    directly from the RRC data.
    """
    if data_dir:
        permit_files = [f for f in os.listdir(data_dir) if f.endswith((".txt", ".dat"))]
        if permit_files:
            filepath = os.path.join(data_dir, permit_files[0])
            logger.info(f"Loading RRC permit file: {filepath}")
            df = parse_rrc_permit_file(filepath)

            # Look for companion status/trailer file (e.g., da800b*)
            status_files = [f for f in permit_files if "status" in f.lower() or "800b" in f.lower()]
            if status_files:
                status_path = os.path.join(data_dir, status_files[0])
                logger.info(f"Found status file with spud dates: {status_path}")
                status_df = parse_rrc_permit_status_file(status_path)
                df = merge_permit_with_status(df, status_df)

            df = filter_reeves_county_horizontal(df, ANALYSIS_YEAR, ANALYSIS_MONTH)
            return parse_coordinates(df)

    if use_sample:
        logger.info("Using sample permit data for methodology development")
        return create_sample_permit_data()

    raise FileNotFoundError("No RRC permit data found and sample data disabled")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gdf = load_permits()
    print(f"\n{'='*80}")
    print(f"Reeves County Horizontal Well Permits")
    print(f"{'='*80}")
    print(f"Total permits: {len(gdf)}")
    print(f"With spud date: {gdf['SPUD_DATE_parsed'].notna().sum()}")
    print(f"Without spud date: {gdf['SPUD_DATE_parsed'].isna().sum()}")
    print(f"\nOperator breakdown:")
    print(gdf["OPERATOR_NAME"].value_counts().to_string())
    print(f"\nPermit date range: {gdf['PERMIT_APPROVAL_DATE'].min()} to {gdf['PERMIT_APPROVAL_DATE'].max()}")
    print(f"\nAll permits with spud status:")
    print(gdf[["PERMIT_NO", "LEASE_NAME", "OPERATOR_NAME", "lat", "lon",
               "PERMIT_APPROVAL_DATE", "SPUD_DATE", "COMPLETION_DATE"]].to_string(index=False))
