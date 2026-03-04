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
from datetime import datetime, date, timedelta

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
    Filter permits to Reeves County horizontal wells that are relevant to
    the analysis period.

    We want ALL permits for the county where either:
    1. Spud date >= analysis period start (spud during or after our window)
    2. Spud date is missing/null (no spud reported - could be drilling,
       could be not started, satellite needs to check)
    3. Spud date is before our window BUT no completion date yet
       (still drilling through our period)

    This is NOT just permits filed in the analysis month. A well permitted
    6 months ago could still be drilling now. We need the full picture of
    what could have an active rig during the period.
    """
    county_code = RRC_COUNTY_CODES.get(COUNTY, "389")

    # Base filter: Reeves County horizontal wells in District 08
    mask = (
        (df["COUNTY_CODE"] == county_code)
        & (df["WELL_DIRECTION"] == WELL_DIRECTION)
        & (df["DISTRICT"] == RRC_DISTRICT)
    )

    filtered = df[mask].copy()

    if year and month:
        period_start = f"{year}{month:02d}01"

        # Parse spud date for filtering
        if "SPUD_DATE_parsed" in filtered.columns:
            spud_col = "SPUD_DATE_parsed"
        else:
            filtered["_spud_parsed"] = pd.to_datetime(
                filtered.get("SPUD_DATE", pd.Series(dtype=str)),
                format="%Y%m%d", errors="coerce"
            )
            spud_col = "_spud_parsed"

        if "COMPLETION_DATE_parsed" in filtered.columns:
            comp_col = "COMPLETION_DATE_parsed"
        else:
            filtered["_comp_parsed"] = pd.to_datetime(
                filtered.get("COMPLETION_DATE", pd.Series(dtype=str)),
                format="%Y%m%d", errors="coerce"
            )
            comp_col = "_comp_parsed"

        period_start_ts = pd.Timestamp(year, month, 1)

        # Keep permits where:
        time_mask = (
            # No spud date at all - could be active, satellite needs to check
            filtered[spud_col].isna()
            # Spud on or after the start of our period
            | (filtered[spud_col] >= period_start_ts)
            # Spud before our period but not yet completed (still drilling)
            | (
                (filtered[spud_col] < period_start_ts)
                & (filtered[comp_col].isna())
            )
            # Spud before but completion falls in or after our period
            | (
                (filtered[spud_col] < period_start_ts)
                & (filtered[comp_col] >= period_start_ts)
            )
        )

        filtered = filtered[time_mask].copy()

        # Clean up temp columns
        for col in ["_spud_parsed", "_comp_parsed"]:
            if col in filtered.columns:
                filtered = filtered.drop(columns=[col])

    logger.info(
        f"Filtered to {len(filtered)} Reeves County horizontal permits "
        f"relevant to {year}-{month:02d}" if year and month
        else f"Filtered to {len(filtered)} Reeves County horizontal permits (all time)"
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

    Reeves County is one of the most active drilling counties in the US.
    At any given time there are typically 15-25 active rigs, and the total
    inventory of permits that are relevant (permitted, drilling, completing,
    recently completed) easily runs 80-120+.

    The sample covers the full spectrum you'd see pulling the actual RRC data:
    - Wells permitted months ago, already completed (done)
    - Wells spud in Dec/Jan, still drilling through Feb
    - Wells spud in Feb (new drilling starts)
    - Wells permitted but no spud date (backlog / not started)
    - Wells in completion (spud 30-60 days ago, rig released, frac crew on site)

    Locations spread across the major drilling areas in Reeves County:
    - Northern (near Mentone): ~31.35-31.55°N, heavy Diamondback/Apache area
    - Central (Pecos corridor): ~31.15-31.35°N, Oxy/ConocoPhillips/Devon
    - Southern: ~30.95-31.15°N, Chevron/EOG territory
    """
    import random
    random.seed(42)

    # Major operators and their approximate well counts / areas in Reeves
    operators = [
        ("DIAMONDBACK ENERGY", 18, 31.38, -103.60),
        ("APACHE CORP", 12, 31.42, -103.62),
        ("CONOCOPHILLIPS", 10, 31.30, -103.55),
        ("OXY USA", 10, 31.20, -103.50),
        ("EOG RESOURCES", 8, 31.10, -103.68),
        ("DEVON ENERGY", 7, 31.35, -103.58),
        ("CENTENNIAL RESOURCE", 6, 31.25, -103.48),
        ("PIONEER NATURAL RES", 5, 31.40, -103.57),
        ("RING ENERGY", 4, 31.18, -103.53),
        ("CHEVRON USA", 4, 31.05, -103.72),
        ("MARATHON OIL", 3, 31.28, -103.65),
        ("CIMAREX ENERGY", 3, 31.32, -103.70),
        ("MEWBOURNE OIL", 2, 31.45, -103.55),
        ("FASKEN OIL", 2, 31.48, -103.58),
        ("COLGATE ENERGY", 2, 31.15, -103.60),
    ]

    lease_prefixes = [
        "UNIVERSITY LANDS", "STATE", "PECOS VALLEY", "RED HILLS",
        "MUSTANG SPRINGS", "DELAWARE MOUNTAIN", "WOLFCAMP", "BONE SPRING",
        "GUADALUPE PASS", "RATTLESNAKE", "TOYAH CREEK", "PHANTOM RANCH",
        "BALMORHEA", "SOUTH PECOS", "SIDEWINDER", "APACHE DRAW",
        "SAND HILLS", "CEDAR LAKE", "MENTONE", "BELL CANYON",
        "CHERRY CANYON", "BRUSHY CANYON", "AVALON SHALE", "THIRD BONE",
        "UPPER WOLFCAMP", "LOWER WOLFCAMP", "SECOND BONE", "FIRST BONE",
    ]

    formations = ["WOLFCAMP A", "WOLFCAMP B", "BONE SPRING", "3RD BONE SPRING",
                   "2ND BONE SPRING", "AVALON", "WOLFCAMP D"]

    wells = []
    permit_counter = 900001
    api_counter = 38938001

    for operator_name, well_count, center_lat, center_lon in operators:
        for i in range(well_count):
            # Scatter wells around operator's center area
            lat = center_lat + random.uniform(-0.08, 0.08)
            lon = center_lon + random.uniform(-0.08, 0.08)
            # Clamp to Reeves County bounds
            lat = max(30.95, min(31.60, lat))
            lon = max(-104.15, min(-103.38, lon))

            lease = random.choice(lease_prefixes)
            section = random.randint(1, 48)
            block = random.randint(1, 60)
            well_no_num = random.randint(1, 8)
            well_suffix = random.choice(["H", "AH", "BH", "CH"])
            well_no = f"{well_no_num}{well_suffix}"

            # Permit dates spread over last 6 months
            days_ago_permit = random.randint(10, 180)
            permit_date = date(2026, 2, 28) - timedelta(days=days_ago_permit)
            permit_str = permit_date.strftime("%Y%m%d")

            # Determine spud/completion status
            # This is the realistic distribution for Reeves County:
            status_roll = random.random()

            if status_roll < 0.12:
                # ~12% completed before Feb (spud 60-120 days ago, completed)
                spud_days_ago = random.randint(60, 120)
                spud_d = date(2026, 2, 28) - timedelta(days=spud_days_ago)
                comp_d = spud_d + timedelta(days=random.randint(20, 35))
                spud_str = spud_d.strftime("%Y%m%d")
                comp_str = comp_d.strftime("%Y%m%d")
            elif status_roll < 0.25:
                # ~13% spud before Feb, completing in Feb (spud 35-60 days ago)
                spud_days_ago = random.randint(35, 60)
                spud_d = date(2026, 2, 28) - timedelta(days=spud_days_ago)
                comp_d = date(2026, 2, 1) + timedelta(days=random.randint(5, 25))
                spud_str = spud_d.strftime("%Y%m%d")
                comp_str = comp_d.strftime("%Y%m%d")
            elif status_roll < 0.40:
                # ~15% spud in Jan, still drilling through Feb (no completion)
                spud_d = date(2026, 1, 1) + timedelta(days=random.randint(5, 28))
                spud_str = spud_d.strftime("%Y%m%d")
                comp_str = None
            elif status_roll < 0.62:
                # ~22% spud in Feb (actively drilling)
                spud_d = date(2026, 2, 1) + timedelta(days=random.randint(0, 25))
                spud_str = spud_d.strftime("%Y%m%d")
                comp_str = None
            else:
                # ~38% permitted but NO spud date
                spud_str = None
                comp_str = None

            wells.append({
                "PERMIT_NO": str(permit_counter),
                "LEASE_NAME": f"{lease} {section:02d}-{block:02d}",
                "WELL_NO": well_no,
                "OPERATOR_NAME": operator_name,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "PERMIT_APPROVAL_DATE": permit_str,
                "API_NO": str(api_counter),
                "SPUD_DATE": spud_str,
                "COMPLETION_DATE": comp_str,
                "TARGET_FORMATION": random.choice(formations),
            })

            permit_counter += 1
            api_counter += 1

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
