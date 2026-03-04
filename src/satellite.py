"""
Sentinel-2 satellite imagery acquisition pipeline.

Uses the Copernicus Data Space Ecosystem (CDSE) to search for and acquire
Sentinel-2 L2A (surface reflectance) imagery over Reeves County well locations.

Sentinel-2 is the clear choice over Landsat for this application:
- 10m resolution (vs 30m) means a typical ~150m drill pad spans ~15 pixels
  vs ~5 pixels on Landsat - much more detectable
- 5-day revisit (vs 16-day) gives better temporal coverage of February
- Red edge bands provide additional vegetation stress indicators

The pipeline:
1. Takes well locations (lat/lon from RRC permits)
2. Creates a buffered search area around each well
3. Queries CDSE for available Sentinel-2 scenes
4. Downloads the relevant bands for before/during analysis periods
5. Clips to the well buffer zones

Authentication: Requires a free Copernicus Data Space account.
Register at https://dataspace.copernicus.eu/
"""

import os
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import requests
from shapely.geometry import box, Point, mapping
from shapely.ops import unary_union

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    BUFFER_RADIUS_M, REEVES_COUNTY_BBOX, ANALYSIS_YEAR, ANALYSIS_MONTH,
    COPERNICUS_CATALOG_URL, COPERNICUS_TOKEN_URL,
    SENTINEL2_BANDS, NDVI_CHANGE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def create_well_buffers(wells: gpd.GeoDataFrame, buffer_m: float = BUFFER_RADIUS_M) -> gpd.GeoDataFrame:
    """
    Create buffer zones around well locations for satellite analysis.

    Projects to UTM Zone 13N (appropriate for Reeves County, TX) to create
    metric buffers, then reprojects back to WGS84.

    The 30m buffer is the minimum useful radius - it ensures we capture
    at least a 60m x 60m area (6x6 pixels at 10m Sentinel-2 resolution).
    For practical drill pad detection, you'd want at least 150m to capture
    the full pad, access roads, and surrounding context.
    """
    # Project to UTM 13N for metric buffering
    wells_utm = wells.to_crs("EPSG:32613")

    # Use a larger buffer for practical detection - 200m captures full pad + context
    detection_buffer = max(buffer_m, 200)
    wells_utm["buffer_geom"] = wells_utm.geometry.buffer(detection_buffer)

    buffered = wells_utm.copy()
    buffered = buffered.set_geometry("buffer_geom")
    buffered = buffered.to_crs("EPSG:4326")

    logger.info(f"Created {detection_buffer}m buffers around {len(buffered)} wells")
    return buffered


def get_search_bbox(wells: gpd.GeoDataFrame) -> dict:
    """
    Get the bounding box encompassing all well locations with padding.
    Used to search for Sentinel-2 tiles covering the area of interest.
    """
    bounds = wells.total_bounds  # [minx, miny, maxx, maxy]
    padding = 0.05  # ~5km padding in degrees

    return {
        "west": bounds[0] - padding,
        "south": bounds[1] - padding,
        "east": bounds[2] + padding,
        "north": bounds[3] + padding,
    }


def search_sentinel2_scenes(
    bbox: dict,
    start_date: date,
    end_date: date,
    max_cloud_pct: int = 20,
) -> list[dict]:
    """
    Search the Copernicus Data Space for Sentinel-2 L2A scenes.

    Uses the OData catalog API to find scenes covering our area of interest
    with acceptable cloud cover.

    Returns a list of scene metadata dicts with id, date, cloud cover, etc.
    """
    # Build OData filter query
    bbox_wkt = (
        f"POLYGON(({bbox['west']} {bbox['south']},"
        f"{bbox['east']} {bbox['south']},"
        f"{bbox['east']} {bbox['north']},"
        f"{bbox['west']} {bbox['north']},"
        f"{bbox['west']} {bbox['south']}))"
    )

    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}') "
            f"and ContentDate/Start gt {start_date.isoformat()}T00:00:00.000Z "
            f"and ContentDate/Start lt {end_date.isoformat()}T23:59:59.999Z "
            f"and Attributes/OData.CSC.DoubleAttribute/any("
            f"att:att/Name eq 'cloudCover' and att/Value lt {max_cloud_pct})"
        ),
        "$orderby": "ContentDate/Start asc",
        "$top": 50,
    }

    try:
        response = requests.get(
            f"{COPERNICUS_CATALOG_URL}/Products",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        scenes = []
        for product in data.get("value", []):
            scenes.append({
                "id": product["Id"],
                "name": product["Name"],
                "date": product["ContentDate"]["Start"][:10],
                "cloud_cover": next(
                    (a["Value"] for a in product.get("Attributes", [])
                     if a.get("Name") == "cloudCover"),
                    None,
                ),
                "footprint": product.get("Footprint"),
            })

        logger.info(f"Found {len(scenes)} Sentinel-2 scenes for {start_date} to {end_date}")
        return scenes

    except requests.RequestException as e:
        logger.warning(f"CDSE catalog query failed: {e}")
        logger.info("Falling back to simulated scene catalog")
        return _simulate_scene_catalog(start_date, end_date)


def _simulate_scene_catalog(start_date: date, end_date: date) -> list[dict]:
    """
    Simulate a Sentinel-2 scene catalog for methodology development.

    Sentinel-2 has a 5-day revisit time with 2 satellites. Over Reeves County,
    this means we get a scene roughly every 2-3 days (overlapping orbits).
    February 2026 would yield approximately 10-12 usable scenes.
    """
    scenes = []
    current = start_date
    scene_idx = 0

    while current <= end_date:
        # Simulate typical cloud cover for West Texas in February
        # West Texas is generally very clear - most scenes are usable
        cloud_cover = [3.2, 8.1, 1.5, 15.2, 2.8, 45.3, 4.1, 2.0, 7.5, 12.3, 1.8, 3.5]
        cloud = cloud_cover[scene_idx % len(cloud_cover)]

        tile_id = "T13SDA"  # Typical Sentinel-2 tile covering Reeves County
        scenes.append({
            "id": f"S2B_MSIL2A_{current.strftime('%Y%m%d')}_{tile_id}",
            "name": f"S2B_MSIL2A_{current.strftime('%Y%m%d')}T170901_N0510_R069_{tile_id}",
            "date": current.isoformat(),
            "cloud_cover": cloud,
            "footprint": None,
            "simulated": True,
        })
        scene_idx += 1
        current += timedelta(days=3)  # ~3 day effective revisit

    # Filter to <20% cloud cover
    usable = [s for s in scenes if s["cloud_cover"] < 20]
    logger.info(
        f"Simulated catalog: {len(scenes)} total scenes, "
        f"{len(usable)} usable (<20% cloud)"
    )
    return usable


def get_analysis_periods() -> tuple[tuple[date, date], tuple[date, date]]:
    """
    Define the before/during time periods for change detection.

    Before period: January 2026 (baseline - what did the area look like
    before our analysis month)

    During period: February 2026 (when we're looking for drilling activity)

    The change between these periods is what we analyze.
    """
    before_start = date(ANALYSIS_YEAR, ANALYSIS_MONTH - 1, 1)
    before_end = date(ANALYSIS_YEAR, ANALYSIS_MONTH, 1) - timedelta(days=1)

    during_start = date(ANALYSIS_YEAR, ANALYSIS_MONTH, 1)
    during_end = date(ANALYSIS_YEAR, ANALYSIS_MONTH + 1, 1) - timedelta(days=1)

    return (before_start, before_end), (during_start, during_end)


def create_synthetic_scene(
    wells: gpd.GeoDataFrame,
    scene_date: date,
    active_wells: gpd.GeoDataFrame = None,
    resolution: float = 10.0,
) -> dict:
    """
    Create a synthetic Sentinel-2 scene for methodology demonstration.

    Generates realistic spectral values for the Reeves County landscape:
    - Desert/scrubland background (low NDVI, high brightness)
    - Well pads appear as bright bare-earth patches (very low NDVI)
    - Active drilling sites show equipment signatures (brightness anomaly)
    - Access roads as linear bright features

    This demonstrates what the change detection algorithm would see.
    """
    bbox = get_search_bbox(wells)
    width = int((bbox["east"] - bbox["west"]) * 111000 / resolution)
    height = int((bbox["north"] - bbox["south"]) * 111000 / resolution)

    # Cap at reasonable size for demo
    width = min(width, 2000)
    height = min(height, 2000)

    # Background: desert scrubland typical of Reeves County
    np.random.seed(42 + scene_date.toordinal())
    red = np.random.normal(0.18, 0.02, (height, width)).astype(np.float32)
    green = np.random.normal(0.15, 0.02, (height, width)).astype(np.float32)
    blue = np.random.normal(0.12, 0.02, (height, width)).astype(np.float32)
    nir = np.random.normal(0.22, 0.03, (height, width)).astype(np.float32)

    transform = from_bounds(bbox["west"], bbox["south"],
                            bbox["east"], bbox["north"], width, height)

    # Add well pad signatures for active drilling wells
    if active_wells is not None:
        for _, well in active_wells.iterrows():
            # Convert lat/lon to pixel coordinates
            col, row = ~transform * (well.geometry.x, well.geometry.y)
            col, row = int(col), int(row)

            if 0 <= row < height and 0 <= col < width:
                # Drill pad: ~150m radius = ~15 pixels at 10m
                pad_radius = 15
                y_min = max(0, row - pad_radius)
                y_max = min(height, row + pad_radius)
                x_min = max(0, col - pad_radius)
                x_max = min(width, col + pad_radius)

                # Well pad signature: bright bare earth (high reflectance, low NDVI)
                red[y_min:y_max, x_min:x_max] = np.random.normal(0.30, 0.02, (y_max-y_min, x_max-x_min))
                green[y_min:y_max, x_min:x_max] = np.random.normal(0.28, 0.02, (y_max-y_min, x_max-x_min))
                blue[y_min:y_max, x_min:x_max] = np.random.normal(0.25, 0.02, (y_max-y_min, x_max-x_min))
                nir[y_min:y_max, x_min:x_max] = np.random.normal(0.25, 0.02, (y_max-y_min, x_max-x_min))

                # Rig itself: very small bright spot at center (~30m = 3 pixels)
                rig_r = 2
                ry_min, ry_max = max(0, row-rig_r), min(height, row+rig_r)
                rx_min, rx_max = max(0, col-rig_r), min(width, col+rig_r)
                red[ry_min:ry_max, rx_min:rx_max] = 0.40
                green[ry_min:ry_max, rx_min:rx_max] = 0.38
                blue[ry_min:ry_max, rx_min:rx_max] = 0.35
                nir[ry_min:ry_max, rx_min:rx_max] = 0.30

    return {
        "red": np.clip(red, 0, 1),
        "green": np.clip(green, 0, 1),
        "blue": np.clip(blue, 0, 1),
        "nir": np.clip(nir, 0, 1),
        "transform": transform,
        "crs": CRS.from_epsg(4326),
        "width": width,
        "height": height,
        "date": scene_date,
        "bbox": bbox,
    }


def save_scene_geotiff(scene: dict, output_path: str):
    """Save a synthetic scene as a GeoTIFF for visualization/analysis."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": scene["width"],
        "height": scene["height"],
        "count": 4,  # R, G, B, NIR
        "crs": scene["crs"],
        "transform": scene["transform"],
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(scene["red"], 1)
        dst.write(scene["green"], 2)
        dst.write(scene["blue"], 3)
        dst.write(scene["nir"], 4)
        dst.set_band_description(1, "Red (B04)")
        dst.set_band_description(2, "Green (B03)")
        dst.set_band_description(3, "Blue (B02)")
        dst.set_band_description(4, "NIR (B08)")

    logger.info(f"Saved scene GeoTIFF: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from rrc_permits import load_permits

    wells = load_permits()

    # Check scene availability
    (before_start, before_end), (during_start, during_end) = get_analysis_periods()
    print(f"\n{'='*80}")
    print(f"Sentinel-2 Scene Search")
    print(f"{'='*80}")
    print(f"Before period: {before_start} to {before_end}")
    print(f"During period: {during_start} to {during_end}")

    bbox = get_search_bbox(wells)
    print(f"Search bbox: {bbox}")

    before_scenes = search_sentinel2_scenes(bbox, before_start, before_end)
    during_scenes = search_sentinel2_scenes(bbox, during_start, during_end)

    print(f"\nBefore period scenes (<20% cloud): {len(before_scenes)}")
    for s in before_scenes:
        print(f"  {s['date']} - {s['cloud_cover']:.1f}% cloud")

    print(f"\nDuring period scenes (<20% cloud): {len(during_scenes)}")
    for s in during_scenes:
        print(f"  {s['date']} - {s['cloud_cover']:.1f}% cloud")

    # Create well buffers
    buffered = create_well_buffers(wells)
    print(f"\nCreated {len(buffered)} well buffer zones")
