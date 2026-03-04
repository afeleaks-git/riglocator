"""
Configuration for the Reeves County Permian Basin rig locator pipeline.
"""

# Geographic focus
COUNTY = "REEVES"
STATE = "TX"
RRC_DISTRICT = "08"  # Permian Basin district

# Reeves County approximate bounding box (WGS84)
REEVES_COUNTY_BBOX = {
    "west": -104.17,
    "south": 30.94,
    "east": -103.37,
    "north": 31.61,
}

# Time period of interest
ANALYSIS_YEAR = 2026
ANALYSIS_MONTH = 2  # February

# Well filtering
WELL_DIRECTION = "H"  # Horizontal wells

# Buffer radius around well locations for satellite analysis (meters)
BUFFER_RADIUS_M = 30

# Satellite imagery
SENTINEL2_BANDS = ["B04", "B03", "B02", "B08"]  # R, G, B, NIR at 10m
LANDSAT_BANDS = ["B4", "B3", "B2", "B5"]  # R, G, B, NIR at 30m

# Change detection thresholds
# NDVI drop > this suggests ground disturbance (pad construction / active drilling)
NDVI_CHANGE_THRESHOLD = -0.15
# Brightness increase > this suggests bare earth / equipment
BRIGHTNESS_CHANGE_THRESHOLD = 0.10

# Minimum pad size in pixels (to filter noise)
# At 10m resolution, a typical drill pad ~100x100m = ~10x10 pixels = 100 pixels
MIN_PAD_AREA_PIXELS = 50

# RRC data source
RRC_PERMIT_URL = "https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/"
RRC_DRILLING_PERMIT_MASTER = "https://mft.rrc.texas.gov/link/7a5577fc-c140-4757-b8d0-07edbb3d261f"

# Baker Hughes rig count
BAKER_HUGHES_URL = "https://rigcount.bakerhughes.com/na-rig-count"

# Copernicus Data Space (Sentinel-2)
COPERNICUS_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
COPERNICUS_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# Sentinel Hub Process API (for actual imagery download)
# Set SH_CLIENT_ID and SH_CLIENT_SECRET as environment variables, or
# they fall back to the defaults below for development.
import os as _os
SH_TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
SH_PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"
SH_CLIENT_ID = _os.environ.get("SH_CLIENT_ID", "sh-fcaefd7b-0d8c-49b0-bd4f-f6b1eccc9a14")
SH_CLIENT_SECRET = _os.environ.get("SH_CLIENT_SECRET", "MYtX5ZWGdT4nnApKxUbU8EdwusYBriY6")
