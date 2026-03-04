"""
Real Sentinel-2 imagery download via Sentinel Hub Process API.

Replaces the synthetic scene generation with actual satellite data
from the Copernicus Data Space / Sentinel Hub service.

Uses OAuth2 client credentials flow to authenticate, then requests
Sentinel-2 L2A (surface reflectance) bands via the Process API.
"""

import io
import logging
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    SH_TOKEN_URL, SH_PROCESS_URL,
    SH_CLIENT_ID, SH_CLIENT_SECRET,
    REEVES_COUNTY_BBOX,
)

logger = logging.getLogger(__name__)

# Cache the token so we don't re-auth every call
_token_cache = {"token": None, "expires_at": 0}


def get_sh_token() -> str:
    """
    Get an OAuth2 access token from Sentinel Hub using client credentials.
    Caches the token until it expires.
    """
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    logger.info("Requesting Sentinel Hub access token...")
    resp = requests.post(
        SH_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": SH_CLIENT_ID,
            "client_secret": SH_CLIENT_SECRET,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 300)

    logger.info("Sentinel Hub token acquired successfully")
    return _token_cache["token"]


# Evalscript that returns 4 bands as float32 + SCL for cloud masking
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B04", "B03", "B02", "B08", "SCL"],
      units: "REFLECTANCE"
    }],
    output: {
      bands: 5,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02, sample.B08, sample.SCL];
}
"""


def download_sentinel2_scene(
    bbox: dict,
    start_date: date,
    end_date: date,
    resolution: float = 10.0,
    max_cloud_pct: int = 30,
) -> Optional[dict]:
    """
    Download real Sentinel-2 L2A bands from Sentinel Hub Process API.

    Args:
        bbox: dict with west, south, east, north
        start_date: start of time window
        end_date: end of time window
        resolution: pixel size in meters (default 10m)
        max_cloud_pct: maximum cloud coverage percentage

    Returns:
        dict with red, green, blue, nir arrays + metadata, or None on failure
    """
    token = get_sh_token()

    # Calculate output dimensions from bbox and resolution
    # Approximate degrees to meters at this latitude (~31°N)
    lat_center = (bbox["south"] + bbox["north"]) / 2
    m_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
    m_per_deg_lat = 110540

    width_m = (bbox["east"] - bbox["west"]) * m_per_deg_lon
    height_m = (bbox["north"] - bbox["south"]) * m_per_deg_lat

    width_px = int(width_m / resolution)
    height_px = int(height_m / resolution)

    # Sentinel Hub has a limit of ~2500x2500 per request at 10m
    # For larger areas, we need to either reduce resolution or tile
    max_dim = 2500
    if width_px > max_dim or height_px > max_dim:
        scale = max(width_px, height_px) / max_dim
        width_px = int(width_px / scale)
        height_px = int(height_px / scale)
        effective_res = resolution * scale
        logger.info(f"Area too large for {resolution}m, using {effective_res:.1f}m ({width_px}x{height_px}px)")
    else:
        logger.info(f"Requesting {width_px}x{height_px}px at {resolution}m resolution")

    payload = {
        "input": {
            "bounds": {
                "bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date.isoformat()}T00:00:00Z",
                            "to": f"{end_date.isoformat()}T23:59:59Z",
                        },
                        "maxCloudCoverage": max_cloud_pct,
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": width_px,
            "height": height_px,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": EVALSCRIPT,
    }

    logger.info(
        f"Downloading Sentinel-2 for {start_date} to {end_date}, "
        f"bbox=[{bbox['west']:.3f},{bbox['south']:.3f},{bbox['east']:.3f},{bbox['north']:.3f}]"
    )

    # Retry with exponential backoff
    for attempt in range(4):
        try:
            resp = requests.post(
                SH_PROCESS_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "image/tiff",
                },
                timeout=120,
            )

            if resp.status_code == 401:
                # Token expired, refresh
                _token_cache["token"] = None
                token = get_sh_token()
                continue

            resp.raise_for_status()

            # Parse the returned GeoTIFF
            with rasterio.open(io.BytesIO(resp.content)) as src:
                data = src.read()  # shape: (5, H, W) -> R, G, B, NIR, SCL

                red = data[0]
                green = data[1]
                blue = data[2]
                nir = data[3]
                scl = data[4]

                # Cloud masking using Scene Classification Layer
                # SCL values: 0=no_data, 1=saturated, 2=dark, 3=shadow,
                # 4=vegetation, 5=bare_soil, 6=water, 7=unclassified,
                # 8=cloud_medium, 9=cloud_high, 10=cirrus, 11=snow
                cloud_mask = np.isin(scl.astype(int), [0, 1, 3, 8, 9, 10])
                cloud_pct = cloud_mask.sum() / cloud_mask.size * 100

                logger.info(
                    f"Downloaded {src.width}x{src.height}px scene, "
                    f"cloud/shadow mask: {cloud_pct:.1f}%"
                )

                # Replace cloudy pixels with NaN
                red = np.where(cloud_mask, np.nan, red).astype(np.float32)
                green = np.where(cloud_mask, np.nan, green).astype(np.float32)
                blue = np.where(cloud_mask, np.nan, blue).astype(np.float32)
                nir = np.where(cloud_mask, np.nan, nir).astype(np.float32)

                transform = from_bounds(
                    bbox["west"], bbox["south"],
                    bbox["east"], bbox["north"],
                    src.width, src.height,
                )

                return {
                    "red": red,
                    "green": green,
                    "blue": blue,
                    "nir": nir,
                    "scl": scl,
                    "cloud_mask": cloud_mask,
                    "cloud_pct": cloud_pct,
                    "transform": transform,
                    "crs": CRS.from_epsg(4326),
                    "width": src.width,
                    "height": src.height,
                    "date_range": (start_date, end_date),
                    "bbox": bbox,
                    "synthetic": False,
                }

        except requests.RequestException as e:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Sentinel Hub request failed (attempt {attempt+1}/4): {e}")
            if attempt < 3:
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error("All Sentinel Hub attempts failed")
                return None

    return None


def download_scene_pair(
    bbox: dict,
    before_start: date,
    before_end: date,
    during_start: date,
    during_end: date,
    resolution: float = 10.0,
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Download before and during scene pair for change detection.
    Uses mosaicking (least cloud cover) to get the best composite for each period.
    """
    logger.info("Downloading BEFORE period scene...")
    before = download_sentinel2_scene(bbox, before_start, before_end, resolution)

    logger.info("Downloading DURING period scene...")
    during = download_sentinel2_scene(bbox, during_start, during_end, resolution)

    if before is not None and during is not None:
        logger.info(
            f"Both scenes downloaded successfully. "
            f"Before: {before['width']}x{before['height']}px ({before['cloud_pct']:.1f}% cloud), "
            f"During: {during['width']}x{during['height']}px ({during['cloud_pct']:.1f}% cloud)"
        )
    else:
        failed = []
        if before is None:
            failed.append("before")
        if during is None:
            failed.append("during")
        logger.error(f"Failed to download: {', '.join(failed)} scene(s)")

    return before, during


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    # Quick test: download a small area around a known well location
    test_bbox = {
        "west": -103.8,
        "south": 31.2,
        "east": -103.7,
        "north": 31.3,
    }

    print("Testing Sentinel Hub authentication and download...")
    scene = download_sentinel2_scene(
        test_bbox,
        start_date=date(2025, 11, 1),
        end_date=date(2025, 11, 30),
        resolution=10.0,
    )

    if scene:
        print(f"SUCCESS: Downloaded {scene['width']}x{scene['height']}px scene")
        print(f"  Red range: {np.nanmin(scene['red']):.4f} - {np.nanmax(scene['red']):.4f}")
        print(f"  NIR range: {np.nanmin(scene['nir']):.4f} - {np.nanmax(scene['nir']):.4f}")
        print(f"  Cloud coverage: {scene['cloud_pct']:.1f}%")
    else:
        print("FAILED: Could not download scene")
