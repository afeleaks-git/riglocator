"""
6-month backtest: Jul-Dec 2025, Reeves County horizontal wells.

For each month, builds a watchlist of wells that COULD have a rig on them,
downloads per-pad Sentinel-2 time series at 10m resolution, and detects
MaxBright spikes indicating rig arrival. Compares detections against
ground-truth spud dates and Baker Hughes rig counts.

Multi-well pad logic:
  In the Permian, a single rig drills 2-4 wells sequentially from the same
  surface pad (~250m radius). We cluster wells into pads, order by spud date
  (or permit date), and assign each well a ~20-day drilling window.
"""

import os
import sys
import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import SH_CLIENT_ID, SH_CLIENT_SECRET, REEVES_COUNTY_BBOX

logger = logging.getLogger(__name__)

SPIKE_THRESHOLD = 0.04
PAD_RADIUS_M = 250.0
WELL_BUFFER_PX = 3
TILE_SIZE_DEG = 0.006
DRILL_DAYS = 20
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
S2_REVISIT_DAYS = 5


def load_all_wells():
    """Load all Reeves County H-wells from the DB with spud/completion dates."""
    from rrc_permits import load_permits_from_db
    frames = []
    for month in range(7, 13):
        try:
            gdf = load_permits_from_db(2025, month)
            frames.append(gdf)
        except Exception as e:
            logger.warning(f"Failed to load 2025-{month:02d}: {e}")
    if not frames:
        raise RuntimeError("Could not load any wells from DB")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("SPUD_DATE_parsed", na_position="last")
    combined = combined.drop_duplicates(subset=["PERMIT_NO"], keep="first")
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    logger.info(f"Loaded {len(combined)} unique wells across Jul-Dec 2025")
    return combined


def cluster_to_pads(wells):
    """Cluster wells within PAD_RADIUS_M into pads."""
    utm = wells.to_crs("EPSG:32613")
    pad_ids = [-1] * len(utm)
    next_pad = 0
    coords = np.array([(g.x, g.y) for g in utm.geometry])
    for i in range(len(coords)):
        if pad_ids[i] >= 0:
            continue
        pad_ids[i] = next_pad
        for j in range(i + 1, len(coords)):
            if pad_ids[j] >= 0:
                continue
            dist = np.sqrt((coords[i][0] - coords[j][0])**2 +
                           (coords[i][1] - coords[j][1])**2)
            if dist <= PAD_RADIUS_M:
                pad_ids[j] = next_pad
        next_pad += 1
    wells = wells.copy()
    wells["pad_id"] = pad_ids
    pad_centroids = wells.groupby("pad_id").agg(
        pad_lon=("lon", "mean"),
        pad_lat=("lat", "mean"),
    )
    wells = wells.merge(pad_centroids, on="pad_id")
    n_multi = (wells.groupby("pad_id").size() > 1).sum()
    logger.info(f"Clustered {len(wells)} wells into {next_pad} pads ({n_multi} multi-well)")
    return wells


def assign_pad_drilling_windows(wells):
    """
    For multi-well pads, estimate when each well was being drilled.
    Sort pad wells by spud date (or permit date fallback). Each well gets
    ~DRILL_DAYS. One rig services the whole pad sequentially.
    """
    wells = wells.copy()
    wells["est_drill_start"] = pd.NaT
    wells["est_drill_end"] = pd.NaT
    wells["drill_order_on_pad"] = 0
    for pad_id, group in wells.groupby("pad_id"):
        sort_df = group[["SPUD_DATE_parsed", "PERMIT_APPROVAL_DATE"]].copy()
        sort_df["_sort_date"] = sort_df["SPUD_DATE_parsed"]
        no_spud = sort_df["_sort_date"].isna()
        if no_spud.any():
            sort_df.loc[no_spud, "_sort_date"] = pd.to_datetime(
                group.loc[no_spud, "PERMIT_APPROVAL_DATE"],
                format="%Y%m%d", errors="coerce"
            )
        sorted_idxs = sort_df.sort_values("_sort_date", na_position="last").index.tolist()
        prev_start = None
        for order, idx in enumerate(sorted_idxs):
            spud = wells.loc[idx, "SPUD_DATE_parsed"]
            if pd.notna(spud):
                start = spud
            elif prev_start is not None:
                start = prev_start + timedelta(days=DRILL_DAYS)
            else:
                start = pd.NaT
            wells.loc[idx, "est_drill_start"] = start
            if pd.notna(start):
                wells.loc[idx, "est_drill_end"] = start + timedelta(days=DRILL_DAYS)
            wells.loc[idx, "drill_order_on_pad"] = order
            if pd.notna(start):
                prev_start = start
    has_window = wells["est_drill_start"].notna().sum()
    logger.info(f"Assigned drilling windows to {has_window}/{len(wells)} wells")
    return wells


def build_watchlist(wells, year, month):
    """Build the watchlist for a given month."""
    month_start = pd.Timestamp(year, month, 1)
    if month == 12:
        month_end = pd.Timestamp(year, 12, 31)
    else:
        month_end = pd.Timestamp(year, month + 1, 1) - timedelta(days=1)
    spud = wells["SPUD_DATE_parsed"]
    comp = wells["COMPLETION_DATE_parsed"]
    permit = pd.to_datetime(wells["PERMIT_APPROVAL_DATE"], format="%Y%m%d", errors="coerce")
    mask = (
        (spud >= month_start)
        | (spud.isna() & (permit <= month_end))
        | ((spud < month_start) & comp.isna())
        | ((spud < month_start) & (comp >= month_start))
    )
    watchlist = wells[mask].copy()
    spud_count = watchlist["SPUD_DATE_parsed"].notna().sum()
    logger.info(f"Watchlist {year}-{month:02d}: {len(watchlist)} wells ({spud_count} with spud dates)")
    return watchlist


def get_ground_truth(wells, year, month):
    """Determine which pads actually had a rig during this month."""
    month_start = pd.Timestamp(year, month, 1)
    if month == 12:
        month_end = pd.Timestamp(year, 12, 31)
    else:
        month_end = pd.Timestamp(year, month + 1, 1) - timedelta(days=1)
    active = {}
    for pad_id, group in wells.groupby("pad_id"):
        drilling_wells = []
        for idx, row in group.iterrows():
            ds = row["est_drill_start"]
            de = row["est_drill_end"]
            if pd.notna(ds) and pd.notna(de):
                if ds <= month_end and de >= month_start:
                    drilling_wells.append(idx)
        if drilling_wells:
            active[pad_id] = drilling_wells
    return active


def get_scene_dates(year, month):
    """Generate approximate Sentinel-2 scene dates for a month (5-day cadence)."""
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=S2_REVISIT_DAYS)
    return dates


def download_pad_scene(pad_lon, pad_lat, scene_date, half_size=TILE_SIZE_DEG / 2):
    """Download a small Sentinel-2 tile centered on a pad for a single date."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"{pad_lon:.5f}_{pad_lat:.5f}_{scene_date.isoformat()}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}
    from sentinel_hub import get_sh_token, EVALSCRIPT
    from config.settings import SH_PROCESS_URL
    bbox = {
        "west": pad_lon - half_size,
        "south": pad_lat - half_size,
        "east": pad_lon + half_size,
        "north": pad_lat + half_size,
    }
    width_px = 60
    height_px = 60
    token = get_sh_token()
    payload = {
        "input": {
            "bounds": {
                "bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{(scene_date - timedelta(days=2)).isoformat()}T00:00:00Z",
                        "to": f"{(scene_date + timedelta(days=3)).isoformat()}T00:00:00Z",
                    },
                    "maxCloudCoverage": 40,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": width_px,
            "height": height_px,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    import requests
    import io
    import rasterio
    for attempt in range(3):
        try:
            resp = requests.post(
                SH_PROCESS_URL,
                json=payload,
                headers={"Authorization": f"Bearer {token}", "Accept": "image/tiff"},
                timeout=60,
            )
            if resp.status_code == 401:
                from sentinel_hub import _token_cache
                _token_cache["token"] = None
                token = get_sh_token()
                continue
            resp.raise_for_status()
            with rasterio.open(io.BytesIO(resp.content)) as src:
                data = src.read()
                result = {
                    "red": data[0].astype(np.float32),
                    "green": data[1].astype(np.float32),
                    "blue": data[2].astype(np.float32),
                    "nir": data[3].astype(np.float32),
                    "scl": data[4].astype(np.float32),
                }
            np.savez_compressed(cache_path, **result)
            return result
        except Exception as e:
            logger.debug(f"Download attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None


def extract_center_metrics(scene):
    """Extract brightness/NDVI at the center of a pad tile."""
    if scene is None:
        return None
    h, w = scene["red"].shape
    cy, cx = h // 2, w // 2
    r = WELL_BUFFER_PX
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    red = scene["red"][y0:y1, x0:x1]
    green = scene["green"][y0:y1, x0:x1]
    blue = scene["blue"][y0:y1, x0:x1]
    nir = scene["nir"][y0:y1, x0:x1]
    scl = scene["scl"][y0:y1, x0:x1]
    cloud_mask = np.isin(scl.astype(int), [0, 1, 3, 8, 9, 10])
    valid = ~cloud_mask
    if valid.sum() < 3:
        return None
    brightness = (red + green + blue) / 3.0
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0)
    return {
        "brightness_mean": float(np.nanmean(brightness[valid])),
        "brightness_max": float(np.nanmax(brightness[valid])),
        "ndvi_mean": float(np.nanmean(ndvi[valid])),
        "valid_pixels": int(valid.sum()),
    }


def detect_rig_from_timeseries(timeseries, baseline_brightness):
    """Detect rig arrival from a time series of per-scene metrics."""
    if not timeseries or baseline_brightness is None:
        return {"detected": False, "max_spike": 0, "n_observations": 0}
    max_spike = 0.0
    spike_date = None
    for obs in timeseries:
        if obs is None:
            continue
        spike = obs["brightness_max"] - baseline_brightness
        if spike > max_spike:
            max_spike = spike
            spike_date = obs.get("date")
    detected = max_spike >= SPIKE_THRESHOLD
    return {
        "detected": detected,
        "max_spike": max_spike,
        "spike_date": spike_date,
        "baseline_brightness": baseline_brightness,
        "n_observations": len([t for t in timeseries if t is not None]),
    }


def run_backtest(months=None):
    """Run the full 6-month backtest."""
    if months is None:
        months = [(2025, m) for m in range(7, 13)]

    hdr = "#" * 80
    eq = "=" * 80
    print()
    print(hdr)
    print("# BACKTEST: Rig Detection vs Ground Truth Spud Dates")
    m0 = months[0]
    m1 = months[-1]
    print(f"# Reeves County, {m0[0]}-{m0[1]:02d} through {m1[0]}-{m1[1]:02d}")
    print(hdr)

    print()
    print("Loading wells from database...")
    all_wells = load_all_wells()
    print(f"  {len(all_wells)} unique wells loaded")

    all_wells = cluster_to_pads(all_wells)
    n_pads = all_wells["pad_id"].nunique()
    n_saved = len(all_wells) - n_pads
    print(f"  {n_pads} unique pads (saves ~{n_saved} duplicate satellite pulls)")

    all_wells = assign_pad_drilling_windows(all_wells)

    from baker_hughes import load_rig_counts

    monthly_results = []

    for year, month in months:
        print()
        print(eq)
        print(f"  MONTH: {year}-{month:02d}")
        print(eq)

        watchlist = build_watchlist(all_wells, year, month)
        n_wl_pads = watchlist["pad_id"].nunique()
        print(f"  Watchlist: {len(watchlist)} wells on {n_wl_pads} pads")

        truth = get_ground_truth(all_wells, year, month)
        truth_pads = set(truth.keys())
        truth_wells = sum(len(v) for v in truth.values())
        print(f"  Ground truth: {len(truth_pads)} pads with active drilling ({truth_wells} wells)")

        pad_info = (
            watchlist
            .groupby("pad_id")
            .agg(pad_lon=("pad_lon", "first"), pad_lat=("pad_lat", "first"))
        )

        if month == 1:
            baseline_date = date(year - 1, 12, 25)
        else:
            baseline_date = date(year, month, 1) - timedelta(days=5)

        scene_dates = get_scene_dates(year, month)
        n_calls = len(pad_info) * (len(scene_dates) + 1)
        print(f"  Downloading {len(pad_info)} pad tiles x {len(scene_dates)+1} dates = {n_calls} API calls")

        detected_pads = set()

        for pad_id, row in pad_info.iterrows():
            plon, plat = row["pad_lon"], row["pad_lat"]
            baseline_scene = download_pad_scene(plon, plat, baseline_date)
            baseline_metrics = extract_center_metrics(baseline_scene)
            baseline_bright = baseline_metrics["brightness_max"] if baseline_metrics else None
            ts = []
            for sd in scene_dates:
                scene = download_pad_scene(plon, plat, sd)
                metrics = extract_center_metrics(scene)
                if metrics:
                    metrics["date"] = sd
                ts.append(metrics)
            detection = detect_rig_from_timeseries(ts, baseline_bright)
            if detection["detected"]:
                detected_pads.add(pad_id)

        tp = detected_pads & truth_pads
        fp = detected_pads - truth_pads
        fn = truth_pads - detected_pads
        precision = len(tp) / max(len(detected_pads), 1)
        recall = len(tp) / max(len(truth_pads), 1)

        try:
            bh = load_rig_counts(use_sample=True, year=year, month=month)
            bh_avg = bh.groupby("Week")["Rig_Count"].sum().mean()
        except Exception:
            bh_avg = float("nan")

        result = {
            "year": year,
            "month": month,
            "watchlist_wells": len(watchlist),
            "watchlist_pads": len(pad_info),
            "truth_active_pads": len(truth_pads),
            "detected_pads": len(detected_pads),
            "TP": len(tp),
            "FP": len(fp),
            "FN": len(fn),
            "precision": precision,
            "recall": recall,
            "baker_hughes_avg": bh_avg,
        }
        monthly_results.append(result)

        print()
        print("  Results:")
        print(f"    Detected:     {len(detected_pads)} pads with rig activity")
        print(f"    Ground truth: {len(truth_pads)} pads actually drilling")
        print(f"    TP={len(tp)}  FP={len(fp)}  FN={len(fn)}")
        print(f"    Precision: {precision:.0%}  Recall: {recall:.0%}")
        print(f"    Baker Hughes avg: {bh_avg:.0f} rigs")
        print(f"    Our estimate:     {len(detected_pads)} rigs (satellite)")
        print(f"    RRC spud-based:   {len(truth_pads)} rigs (ground truth)")

    print()
    print()
    print(hdr)
    print("# BACKTEST SUMMARY")
    print(hdr)

    summary_df = pd.DataFrame(monthly_results)
    print()
    print(summary_df.to_string(index=False))

    avg_p = summary_df["precision"].mean()
    avg_r = summary_df["recall"].mean()
    avg_bh = summary_df["baker_hughes_avg"].mean()
    avg_det = summary_df["detected_pads"].mean()
    print(f"  Average precision: {avg_p:.0%}")
    print(f"  Average recall:    {avg_r:.0%}")
    print(f"  Average BH rigs:   {avg_bh:.0f}")
    print(f"  Average detected:  {avg_det:.0f}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    details_path = os.path.join(output_dir, "backtest_results.csv")
    summary_df.to_csv(details_path, index=False)
    print(f"  Results saved to {details_path}")

    return summary_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    run_backtest()
