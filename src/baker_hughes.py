"""
Baker Hughes rig count data handler.

Baker Hughes publishes weekly rig count data as Excel pivot tables at:
https://rigcount.bakerhughes.com/na-rig-count

The "Pivot Table" download includes county-level detail with fields:
Country, State, County, Basin, DrillFor, Location, Trajectory, etc.

Since there's no public API, we either:
1. Parse a downloaded Excel pivot table
2. Use representative data for methodology development

For Reeves County, we're looking at Permian Basin rigs, predominantly
horizontal wells targeting the Delaware Basin (Wolfcamp, Bone Spring).
"""

import os
import logging
from datetime import date, timedelta

import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import COUNTY, ANALYSIS_YEAR, ANALYSIS_MONTH

logger = logging.getLogger(__name__)


def parse_baker_hughes_pivot(filepath: str) -> pd.DataFrame:
    """
    Parse the Baker Hughes NA rig count pivot table Excel file.

    The pivot table has weekly columns with rig counts broken down by
    geography (state/county/basin) and well characteristics.
    """
    df = pd.read_excel(filepath, sheet_name=0, header=0)
    logger.info(f"Loaded Baker Hughes pivot table: {len(df)} rows")
    return df


def filter_reeves_county(df: pd.DataFrame) -> pd.DataFrame:
    """Filter Baker Hughes data to Reeves County."""
    # Column names may vary; try common patterns
    county_col = None
    for candidate in ["County", "COUNTY", "county"]:
        if candidate in df.columns:
            county_col = candidate
            break

    if county_col is None:
        logger.warning("No county column found in Baker Hughes data")
        return df

    return df[df[county_col].str.upper().str.contains("REEVES", na=False)]


def get_february_rig_weeks() -> list[date]:
    """Get the Friday dates (Baker Hughes release days) in February 2026."""
    weeks = []
    d = date(ANALYSIS_YEAR, ANALYSIS_MONTH, 1)
    end = date(ANALYSIS_YEAR, ANALYSIS_MONTH + 1, 1) if ANALYSIS_MONTH < 12 else date(ANALYSIS_YEAR + 1, 1, 1)

    while d < end:
        if d.weekday() == 4:  # Friday
            weeks.append(d)
        d += timedelta(days=1)

    return weeks


def create_sample_rig_count() -> pd.DataFrame:
    """
    Create representative Baker Hughes rig count data for Reeves County, Feb 2026.

    As of early 2026, the Permian Basin has ~242 active rigs total.
    Reeves County typically has 15-25 active rigs at any given time,
    making it one of the most active counties in the Permian.

    This sample reflects typical weekly fluctuation patterns.
    """
    weeks = get_february_rig_weeks()

    records = []
    # Typical Reeves County rig counts by week, reflecting real-world patterns
    # Mix of oil-directed horizontal wells targeting Delaware Basin formations
    rig_counts = [18, 19, 18, 20]  # Slight variation week to week

    operators = {
        "DIAMONDBACK ENERGY": [4, 4, 4, 5],
        "APACHE CORP": [3, 3, 3, 3],
        "CONOCOPHILLIPS": [2, 3, 2, 3],
        "OXY USA": [2, 2, 2, 2],
        "EOG RESOURCES": [2, 2, 2, 2],
        "DEVON ENERGY": [1, 1, 1, 1],
        "CENTENNIAL RESOURCE": [1, 1, 1, 1],
        "PIONEER NATURAL RES": [1, 1, 1, 1],
        "RING ENERGY": [1, 1, 1, 1],
        "CHEVRON USA": [1, 1, 1, 1],
    }

    for i, week_date in enumerate(weeks):
        if i >= len(rig_counts):
            break

        for operator, counts in operators.items():
            count = counts[i] if i < len(counts) else counts[-1]
            records.append({
                "Week": week_date,
                "Country": "UNITED STATES",
                "State": "TEXAS",
                "County": "REEVES",
                "Basin": "PERMIAN",
                "Sub_Basin": "DELAWARE",
                "DrillFor": "Oil",
                "Trajectory": "Horizontal",
                "Operator": operator,
                "Rig_Count": count,
            })

    df = pd.DataFrame(records)

    # Add weekly totals
    weekly_totals = df.groupby("Week")["Rig_Count"].sum().reset_index()
    weekly_totals.columns = ["Week", "Total_Rigs"]

    logger.info(
        f"Created sample rig count data: {len(weeks)} weeks, "
        f"avg {weekly_totals['Total_Rigs'].mean():.0f} rigs/week"
    )
    return df


def load_rig_counts(data_dir: str = None, use_sample: bool = True) -> pd.DataFrame:
    """Load Baker Hughes rig count data."""
    if data_dir:
        excel_files = [f for f in os.listdir(data_dir) if f.endswith((".xlsx", ".xls"))]
        if excel_files:
            filepath = os.path.join(data_dir, excel_files[0])
            logger.info(f"Loading Baker Hughes file: {filepath}")
            df = parse_baker_hughes_pivot(filepath)
            return filter_reeves_county(df)

    if use_sample:
        logger.info("Using sample rig count data for methodology development")
        return create_sample_rig_count()

    raise FileNotFoundError("No Baker Hughes data found and sample data disabled")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_rig_counts()
    print(f"\n{'='*80}")
    print(f"Baker Hughes Rig Count - Reeves County, February {ANALYSIS_YEAR}")
    print(f"{'='*80}")

    weekly = df.groupby("Week")["Rig_Count"].sum()
    print(f"\nWeekly rig counts:")
    for week, count in weekly.items():
        print(f"  {week}: {count} rigs")

    print(f"\nBy operator:")
    op_totals = df.groupby("Operator")["Rig_Count"].mean().sort_values(ascending=False)
    for op, avg in op_totals.items():
        print(f"  {op}: ~{avg:.0f} rigs/week")
