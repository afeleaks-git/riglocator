"""
Change detection analysis for identifying drilling rig activity from Sentinel-2 imagery.

Methodology:
1. NDVI differencing: Compare vegetation index between before/during periods.
   Active drill pads show NDVI drops (vegetation cleared for pad construction).

2. Brightness change: Bare earth and equipment are brighter than natural terrain.
   New pads show increased brightness in visible bands.

3. Spatial filtering: Real drill pads are 1-4 acres (~4000-16000 m²).
   At 10m resolution that's 40-160 pixels. Filter out smaller noise.

4. Proximity filtering: We only look within buffer zones around known permit
   locations, drastically reducing false positives.

CRITICAL: Distinguishing drilling rigs from completion rigs:
- Drilling phase: Single tall mast/derrick, smaller footprint, typically
  20-30 days for horizontal wells in the Permian
- Completion phase: Frac spread with many pump trucks, sand storage, wider
  footprint, 15-25 days typical
- The temporal sequence matters: if a well was spud 30+ days ago and still
  shows activity, it may be in completion rather than drilling
- We use the spud date from the RRC permit data to estimate which phase
  a well is likely in at the time of each satellite pass

Confidence scoring:
- HIGH: NDVI drop + brightness increase + within permit buffer + spud date
  consistent with drilling timeline
- MEDIUM: Change detected within permit buffer but no spud date to confirm
- LOW: Change detected but ambiguous (could be completion, could be
  pad construction without drilling yet)
"""

import logging
from datetime import date, timedelta

import numpy as np
import geopandas as gpd
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    NDVI_CHANGE_THRESHOLD, BRIGHTNESS_CHANGE_THRESHOLD,
    MIN_PAD_AREA_PIXELS, BUFFER_RADIUS_M,
)

logger = logging.getLogger(__name__)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)

    For Reeves County desert landscape:
    - Natural terrain: NDVI ~0.05-0.15 (sparse desert scrub)
    - Well pads (bare earth): NDVI ~-0.05 to 0.05
    - Irrigated agriculture: NDVI ~0.3-0.6
    """
    denom = nir + red
    # Avoid division by zero
    ndvi = np.where(denom > 0, (nir - red) / denom, 0)
    return ndvi.astype(np.float32)


def compute_brightness(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Compute average visible brightness.
    Used to detect newly exposed bare earth and equipment.
    """
    return ((red + green + blue) / 3.0).astype(np.float32)


def compute_bare_soil_index(red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Compute a bare soil index to detect newly graded/cleared areas.
    BSI = ((Red + Blue) - (NIR + Green)) / ((Red + Blue) + (NIR + Green))

    Higher values indicate bare soil (well pads, roads).
    """
    # Simplified version using available bands
    numerator = (red + blue) - nir
    denominator = (red + blue) + nir
    bsi = np.where(denominator > 0, numerator / denominator, 0)
    return bsi.astype(np.float32)


def detect_pad_changes(
    before_scene: dict,
    during_scene: dict,
) -> dict:
    """
    Run change detection between before and during scenes.

    Returns change maps:
    - ndvi_change: Negative values = vegetation loss (potential pad construction)
    - brightness_change: Positive values = new bare earth/equipment
    - combined_score: Weighted combination for overall change confidence
    """
    # Compute indices for both periods
    ndvi_before = compute_ndvi(before_scene["red"], before_scene["nir"])
    ndvi_during = compute_ndvi(during_scene["red"], during_scene["nir"])

    bright_before = compute_brightness(before_scene["red"], before_scene["green"], before_scene["blue"])
    bright_during = compute_brightness(during_scene["red"], during_scene["green"], during_scene["blue"])

    # Change maps
    ndvi_change = ndvi_during - ndvi_before  # Negative = vegetation loss
    brightness_change = bright_during - bright_before  # Positive = brighter

    # Combined change score (higher = more likely a new drill pad)
    # Invert NDVI change so positive = more change
    combined = (-ndvi_change * 0.6) + (brightness_change * 0.4)

    return {
        "ndvi_before": ndvi_before,
        "ndvi_during": ndvi_during,
        "ndvi_change": ndvi_change,
        "brightness_before": bright_before,
        "brightness_during": bright_during,
        "brightness_change": brightness_change,
        "combined_score": combined,
        "transform": during_scene["transform"],
        "crs": during_scene["crs"],
    }


def extract_well_signatures(
    change_maps: dict,
    wells: gpd.GeoDataFrame,
    buffer_pixels: int = 20,
) -> gpd.GeoDataFrame:
    """
    Extract change detection metrics within each well's buffer zone.

    For each well location, we sample the change maps within the buffer
    and compute statistics that indicate whether activity is present.
    """
    transform = change_maps["transform"]
    ndvi_change = change_maps["ndvi_change"]
    brightness_change = change_maps["brightness_change"]
    combined = change_maps["combined_score"]

    results = []
    height, width = ndvi_change.shape

    for idx, well in wells.iterrows():
        # Convert well lat/lon to pixel coordinates
        col, row = ~transform * (well.geometry.x, well.geometry.y)
        col, row = int(col), int(row)

        if not (0 <= row < height and 0 <= col < width):
            continue

        # Extract buffer window
        r_min = max(0, row - buffer_pixels)
        r_max = min(height, row + buffer_pixels)
        c_min = max(0, col - buffer_pixels)
        c_max = min(width, col + buffer_pixels)

        ndvi_patch = ndvi_change[r_min:r_max, c_min:c_max]
        bright_patch = brightness_change[r_min:r_max, c_min:c_max]
        combined_patch = combined[r_min:r_max, c_min:c_max]

        # Compute metrics
        results.append({
            "PERMIT_NO": well.get("PERMIT_NO"),
            "LEASE_NAME": well.get("LEASE_NAME"),
            "OPERATOR_NAME": well.get("OPERATOR_NAME"),
            "lat": well.get("lat"),
            "lon": well.get("lon"),
            "SPUD_DATE": well.get("SPUD_DATE"),
            "ndvi_change_mean": float(np.mean(ndvi_patch)),
            "ndvi_change_min": float(np.min(ndvi_patch)),
            "brightness_change_mean": float(np.mean(bright_patch)),
            "brightness_change_max": float(np.max(bright_patch)),
            "combined_score_mean": float(np.mean(combined_patch)),
            "combined_score_max": float(np.max(combined_patch)),
            # Count pixels exceeding thresholds
            "disturbed_pixel_count": int(np.sum(ndvi_patch < NDVI_CHANGE_THRESHOLD)),
            "bright_pixel_count": int(np.sum(bright_patch > BRIGHTNESS_CHANGE_THRESHOLD)),
        })

    result_df = gpd.GeoDataFrame(
        pd.DataFrame(results),
        geometry=[well.geometry for _, well in wells.iterrows()
                  if _check_bounds(well, transform, height, width)],
        crs=wells.crs,
    )
    return result_df


def _check_bounds(well, transform, height, width):
    col, row = ~transform * (well.geometry.x, well.geometry.y)
    return 0 <= int(row) < height and 0 <= int(col) < width


def classify_detections(
    detections: gpd.GeoDataFrame,
    analysis_date: date = None,
) -> gpd.GeoDataFrame:
    """
    Classify each well's detection into confidence levels and likely activity type.

    Confidence levels:
    - HIGH: Strong change signal + spud date consistent with active drilling
    - MEDIUM: Change detected but timing is ambiguous
    - LOW: Weak or ambiguous signal

    Activity type:
    - DRILLING_RIG: Consistent with active drilling (spud < 30 days ago)
    - COMPLETION_RIG: Well was spud > 30 days ago, likely in completion
    - PAD_CONSTRUCTION: Change detected but no spud date yet
    - NO_ACTIVITY: No significant change detected
    """
    detections = detections.copy()

    if analysis_date is None:
        analysis_date = date(2026, 2, 28)

    classifications = []
    confidences = []

    for _, row in detections.iterrows():
        ndvi_drop = row["ndvi_change_mean"] < NDVI_CHANGE_THRESHOLD
        brightness_up = row["brightness_change_mean"] > BRIGHTNESS_CHANGE_THRESHOLD
        significant_disturbance = row["disturbed_pixel_count"] > MIN_PAD_AREA_PIXELS * 0.3
        spud_date = pd.to_datetime(row.get("SPUD_DATE"), format="%Y%m%d", errors="coerce")

        if not (ndvi_drop or brightness_up or significant_disturbance):
            classifications.append("NO_ACTIVITY")
            confidences.append("LOW")
            continue

        # There IS change - now classify what kind
        if pd.notna(spud_date):
            days_since_spud = (analysis_date - spud_date.date()).days

            if 0 <= days_since_spud <= 35:
                # Typical horizontal well drilling time: 15-30 days
                classifications.append("DRILLING_RIG")
                confidences.append("HIGH")
            elif days_since_spud > 35:
                # Likely moved to completion phase
                classifications.append("COMPLETION_RIG")
                confidences.append("HIGH")
            else:
                # Spud date is in the future - pad construction?
                classifications.append("PAD_CONSTRUCTION")
                confidences.append("MEDIUM")
        else:
            # No spud date - we see change but can't confirm phase
            if significant_disturbance and (ndvi_drop and brightness_up):
                classifications.append("PROBABLE_DRILLING")
                confidences.append("MEDIUM")
            else:
                classifications.append("PAD_CONSTRUCTION")
                confidences.append("LOW")

    detections["activity_class"] = classifications
    detections["confidence"] = confidences

    logger.info(
        f"Classification results:\n"
        + detections["activity_class"].value_counts().to_string()
    )

    return detections


def deduplicate_by_pad(
    detections: gpd.GeoDataFrame,
    pad_radius_m: float = 250.0,
) -> gpd.GeoDataFrame:
    """
    Group nearby wells into pads and keep one detection per pad.

    In the Permian, multiple horizontal wells are often drilled from the
    same surface pad (e.g., 2-4 wells targeting different zones like
    Wolfcamp A, Wolfcamp B, Bone Spring). A single rig services all
    wells on the pad sequentially. Counting each well as a separate rig
    would overcount.

    Groups wells within pad_radius_m of each other, then keeps the
    highest-confidence detection per group as the pad representative.
    """
    if len(detections) == 0:
        return detections

    # Project to UTM for metric distances
    det_utm = detections.to_crs("EPSG:32613")

    # Simple spatial clustering: assign each well to a pad group
    pad_ids = [-1] * len(det_utm)
    next_pad = 0

    for i, (idx_i, row_i) in enumerate(det_utm.iterrows()):
        if pad_ids[i] >= 0:
            continue
        # Start a new pad group
        pad_ids[i] = next_pad
        for j, (idx_j, row_j) in enumerate(det_utm.iterrows()):
            if j <= i or pad_ids[j] >= 0:
                continue
            dist = row_i.geometry.distance(row_j.geometry)
            if dist <= pad_radius_m:
                pad_ids[j] = next_pad
        next_pad += 1

    detections = detections.copy()
    detections["pad_group"] = pad_ids

    # Count wells per pad for reporting
    pad_sizes = detections.groupby("pad_group").size()
    multi_well_pads = (pad_sizes > 1).sum()
    if multi_well_pads > 0:
        logger.info(
            f"Pad deduplication: {len(detections)} wells -> {next_pad} pads "
            f"({multi_well_pads} multi-well pads)"
        )

    # Rank detections: prefer HIGH confidence DRILLING_RIG, then by combined score
    confidence_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    activity_rank = {
        "DRILLING_RIG": 5, "PROBABLE_DRILLING": 4,
        "COMPLETION_RIG": 3, "PAD_CONSTRUCTION": 2, "NO_ACTIVITY": 1,
    }
    detections["_conf_rank"] = detections["confidence"].map(confidence_rank).fillna(0)
    detections["_act_rank"] = detections["activity_class"].map(activity_rank).fillna(0)

    # Keep the best detection per pad
    deduped = (
        detections
        .sort_values(["_act_rank", "_conf_rank", "combined_score_max"], ascending=False)
        .groupby("pad_group")
        .first()
        .reset_index()
    )

    # Add count of wells on this pad
    deduped["wells_on_pad"] = deduped["pad_group"].map(pad_sizes)

    # Clean up temp columns
    deduped = deduped.drop(columns=["_conf_rank", "_act_rank"])

    # Restore as GeoDataFrame
    deduped = gpd.GeoDataFrame(deduped, geometry="geometry", crs=detections.crs)

    logger.info(
        f"After dedup: {len(deduped)} unique pads "
        f"(was {len(detections)} individual wells)"
    )
    return deduped


def estimate_rig_vs_completion(
    detections: gpd.GeoDataFrame,
    rig_count_target: int,
) -> dict:
    """
    Compare detected drilling activity against Baker Hughes rig count.

    This is a key validation step: if Baker Hughes says 18 rigs are running
    in Reeves County, we should detect roughly that many HIGH-confidence
    DRILLING_RIG classifications. Large discrepancies indicate either:
    - Our detection threshold is wrong
    - Some rigs are on wells not in our permit set
    - Some detected changes are actually completion activity, not drilling
    """
    drilling_high = detections[
        (detections["activity_class"] == "DRILLING_RIG")
        & (detections["confidence"] == "HIGH")
    ]
    drilling_all = detections[
        detections["activity_class"].isin(["DRILLING_RIG", "PROBABLE_DRILLING"])
    ]
    completion = detections[detections["activity_class"] == "COMPLETION_RIG"]

    return {
        "baker_hughes_rig_count": rig_count_target,
        "high_confidence_drilling_rigs": len(drilling_high),
        "all_probable_drilling": len(drilling_all),
        "completion_activity": len(completion),
        "detection_rate": len(drilling_high) / max(rig_count_target, 1),
        "interpretation": (
            f"Baker Hughes reports {rig_count_target} rigs. "
            f"We detect {len(drilling_high)} high-confidence drilling sites "
            f"and {len(completion)} completion sites. "
            f"Detection rate: {len(drilling_high)/max(rig_count_target,1):.0%}."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from rrc_permits import load_permits
    from satellite import create_synthetic_scene, get_analysis_periods
    from well_status import classify_well_status, get_active_drilling

    wells = load_permits()
    wells = classify_well_status(wells)
    active_drilling = get_active_drilling(wells)

    (before_start, before_end), (during_start, during_end) = get_analysis_periods()

    before_scene = create_synthetic_scene(wells, before_start)
    during_scene = create_synthetic_scene(wells, during_start, active_wells=active_drilling)

    changes = detect_pad_changes(before_scene, during_scene)
    detections = extract_well_signatures(changes, wells)
    classified = classify_detections(detections)

    print(f"\n{'='*80}")
    print(f"Change Detection Results")
    print(f"{'='*80}")
    print(classified[["PERMIT_NO", "LEASE_NAME", "OPERATOR_NAME", "SPUD_DATE",
                       "ndvi_change_mean", "brightness_change_mean",
                       "activity_class", "confidence"]].to_string(index=False))

    validation = estimate_rig_vs_completion(classified, rig_count_target=18)
    print(f"\n{validation['interpretation']}")
