"""
Well status classification from RRC reported dates.

Uses spud dates and completion dates from the RRC permit data to classify
what was happening at each well during the analysis period. No external
data needed - the RRC is the authoritative source for these dates.
"""

import logging

import pandas as pd
import geopandas as gpd

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ANALYSIS_YEAR, ANALYSIS_MONTH

logger = logging.getLogger(__name__)


def classify_well_status(permits: gpd.GeoDataFrame, year: int = ANALYSIS_YEAR, month: int = ANALYSIS_MONTH) -> gpd.GeoDataFrame:
    """
    Classify each well's drilling status based on RRC-reported spud/completion dates.

    Categories:
    - SPUD_IN_PERIOD: Started drilling during the analysis month
    - DRILLING_THROUGH: Spud before the period, no completion yet (still drilling)
    - COMPLETING: Spud before the period, completion date falls in the period
    - COMPLETED_BEFORE: Both spud and completion before our window (done)
    - NO_SPUD: Permitted but no spud date reported
    """
    period_start = pd.Timestamp(year, month, 1)
    if month < 12:
        period_end = pd.Timestamp(year, month + 1, 1)
    else:
        period_end = pd.Timestamp(year + 1, 1, 1)

    def get_status(row):
        spud = row.get("SPUD_DATE_parsed")
        comp = row.get("COMPLETION_DATE_parsed")

        if pd.isna(spud):
            return "NO_SPUD"

        if spud >= period_start and spud < period_end:
            if pd.notna(comp) and comp < period_end:
                return "SPUD_AND_COMPLETED"
            return "SPUD_IN_PERIOD"

        if spud < period_start:
            if pd.isna(comp):
                return "DRILLING_THROUGH"
            if comp >= period_start:
                return "COMPLETING"
            return "COMPLETED_BEFORE"

        return "SPUD_AFTER"

    permits = permits.copy()
    permits["well_status"] = permits.apply(get_status, axis=1)

    status_counts = permits["well_status"].value_counts()
    logger.info(f"Well status classification:\n{status_counts.to_string()}")

    return permits


def get_active_drilling(permits: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Wells where a drilling rig should be present during the analysis period."""
    active = permits[permits["well_status"].isin(["SPUD_IN_PERIOD", "DRILLING_THROUGH"])].copy()
    logger.info(f"Active drilling rigs: {len(active)}")
    return active


def get_completion_activity(permits: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Wells in completion phase - NOT drilling rigs, different equipment on site."""
    completing = permits[permits["well_status"].isin(["COMPLETING", "SPUD_AND_COMPLETED"])].copy()
    logger.info(f"Completion activity: {len(completing)}")
    return completing


def get_no_spud(permits: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Permitted wells with no spud date - satellite verification candidates."""
    no_spud = permits[permits["well_status"] == "NO_SPUD"].copy()
    logger.info(f"No spud date (satellite check candidates): {len(no_spud)}")
    return no_spud
