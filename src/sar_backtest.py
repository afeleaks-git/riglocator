"""
1-month SAR backtest for Dec 2025.
Downloads Sentinel-1 VV/VH for all watchlist pads, computes
vv_contrast and delta_contrast, sweeps thresholds for precision/recall.
"""

import numpy as np
import io
import time
import requests
import sys
import os
import logging
from datetime import date, timedelta

import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING)

from config.settings import SH_TOKEN_URL, SH_PROCESS_URL, SH_CLIENT_ID, SH_CLIENT_SECRET
from backtest import load_all_wells, cluster_to_pads, assign_pad_drilling_windows, get_ground_truth, build_watchlist

SAR_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sar_cache")

SAR_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH"], units: "LINEAR_POWER" }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}"""

_token_cache = {"token": None, "expires_at": 0}

def get_token():
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    resp = requests.post(SH_TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": SH_CLIENT_ID,
        "client_secret": SH_CLIENT_SECRET,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 300)
    return _token_cache["token"]


def download_sar_cached(lon, lat, scene_date, token, half=0.003):
    """Download SAR tile with NPZ caching."""
    os.makedirs(SAR_CACHE_DIR, exist_ok=True)
    cache_key = f"sar_{lon:.5f}_{lat:.5f}_{scene_date.isoformat()}"
    cache_path = os.path.join(SAR_CACHE_DIR, f"{cache_key}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        result = {k: float(d[k]) for k in d.files}
        d.close()
        if result.get("valid_pct", 0) < 1:
            return None
        return result

    bbox = [lon - half, lat - half, lon + half, lat + half]
    t_from = (scene_date - timedelta(days=6)).isoformat() + "T00:00:00Z"
    t_to = (scene_date + timedelta(days=6)).isoformat() + "T23:59:59Z"
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": t_from, "to": t_to},
                    "acquisitionMode": "IW",
                    "polarization": "DV",
                    "resolution": "HIGH",
                },
                "processing": {
                    "orthorectify": True,
                    "backCoeff": "GAMMA0_TERRAIN",
                },
            }],
        },
        "output": {
            "width": 60, "height": 60,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": SAR_EVALSCRIPT,
    }
    token = get_token()  # refresh if needed
    resp = requests.post(SH_PROCESS_URL, json=payload,
        headers={"Authorization": f"Bearer {token}", "Accept": "image/tiff"}, timeout=120)
    if resp.status_code == 401:
        _token_cache["token"] = None
        token = get_token()
        resp = requests.post(SH_PROCESS_URL, json=payload,
            headers={"Authorization": f"Bearer {token}", "Accept": "image/tiff"}, timeout=120)
    if resp.status_code != 200:
        np.savez(cache_path, valid_pct=0)
        return None
    with rasterio.open(io.BytesIO(resp.content)) as src:
        data = src.read()
    vv, vh = data[0], data[1]
    valid = vv > 0
    if valid.sum() < 100:
        np.savez(cache_path, valid_pct=0)
        return None
    vv_db = 10 * np.log10(vv + 1e-10)
    vh_db = 10 * np.log10(vh + 1e-10)
    cy, cx = 30, 30
    r = 3
    center_vv = vv_db[cy-r:cy+r, cx-r:cx+r]
    center_vh = vh_db[cy-r:cy+r, cx-r:cx+r]
    bg_mask = np.ones_like(vv_db, dtype=bool)
    bg_mask[cy-r:cy+r, cx-r:cx+r] = False
    bg_mask &= valid
    result = {
        "vv_mean": float(np.mean(vv_db[valid])),
        "vh_mean": float(np.mean(vh_db[valid])),
        "vv_max": float(np.max(vv_db[valid])),
        "vv_center": float(np.mean(center_vv)),
        "vv_bg": float(np.mean(vv_db[bg_mask])),
        "vv_contrast": float(np.mean(center_vv) - np.mean(vv_db[bg_mask])),
        "vh_contrast": float(np.mean(center_vh) - np.mean(vh_db[bg_mask])),
        "vv_std": float(np.std(vv_db[valid])),
        "valid_pct": float(valid.sum() / valid.size * 100),
    }
    np.savez(cache_path, **result)
    return result


def run_sar_backtest(year=2025, month=12):
    """Run 1-month SAR backtest with threshold sweep."""
    print(f"=== SAR BACKTEST {year}-{month:02d} ===")

    # Load and prep wells
    wells = load_all_wells()
    wells = cluster_to_pads(wells)
    wells = assign_pad_drilling_windows(wells)

    # Ground truth: which pads had a rig this month?
    truth = get_ground_truth(wells, year, month)
    truth_pad_ids = set(truth.keys())
    print(f"Ground truth: {len(truth_pad_ids)} active pads")

    # Get all unique pads from watchlist
    watchlist = build_watchlist(wells, year, month)
    watch_pads = watchlist.groupby("pad_id").agg(
        pad_lon=("pad_lon", "first"), pad_lat=("pad_lat", "first"),
    ).reset_index()
    print(f"Watchlist pads: {len(watch_pads)}")

    # Scene dates: ~12 day revisit for S1
    scene_dates = []
    d = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    while d <= end:
        scene_dates.append(d)
        d += timedelta(days=12)
    # Also get a baseline scene from the prior month
    baseline_date = date(year, month, 1) - timedelta(days=15)
    print(f"Scene dates: {scene_dates}")
    print(f"Baseline date: {baseline_date}")

    # Download SAR for each pad x scene_date
    pad_metrics = {}  # pad_id -> {max_contrast, max_delta_contrast, ...}
    total = len(watch_pads)
    for i, (_, row) in enumerate(watch_pads.iterrows()):
        pid = row["pad_id"]
        lon, lat = row["pad_lon"], row["pad_lat"]

        # Download baseline
        base = download_sar_cached(lon, lat, baseline_date, None)
        base_contrast = base["vv_contrast"] if base else 0

        # Download each scene date
        contrasts = []
        delta_contrasts = []
        vv_maxes = []
        for sd in scene_dates:
            scene = download_sar_cached(lon, lat, sd, None)
            if scene is None:
                continue
            contrasts.append(scene["vv_contrast"])
            delta_contrasts.append(scene["vv_contrast"] - base_contrast)
            vv_maxes.append(scene["vv_max"])
            time.sleep(0.2)

        if not contrasts:
            continue

        pad_metrics[pid] = {
            "max_contrast": max(contrasts),
            "mean_contrast": np.mean(contrasts),
            "max_delta_contrast": max(delta_contrasts),
            "mean_delta_contrast": np.mean(delta_contrasts),
            "max_vv_max": max(vv_maxes),
            "is_active": pid in truth_pad_ids,
        }

        status = "ACTIVE" if pid in truth_pad_ids else "inactive"
        mc = pad_metrics[pid]["max_contrast"]
        mdc = pad_metrics[pid]["max_delta_contrast"]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] pad {pid} ({status}): max_contrast={mc:+.1f}dB delta={mdc:+.1f}dB")

    print(f"Downloaded SAR for {len(pad_metrics)} pads")

    # Threshold sweep
    print()
    print("=== THRESHOLD SWEEP ===")
    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    print(f"{"Metric":<20} {"Thresh":>6} {"TP":>4} {"FP":>4} {"FN":>4} {"Prec":>6} {"Rec":>6} {"F1":>6}")
    print("-" * 56)

    best_f1 = 0
    best_cfg = ""
    for metric_name in ["max_contrast", "mean_contrast", "max_delta_contrast"]:
        for thresh in thresholds:
            tp = fp = fn = 0
            for pid, m in pad_metrics.items():
                detected = m[metric_name] >= thresh
                actual = m["is_active"]
                if detected and actual:
                    tp += 1
                elif detected and not actual:
                    fp += 1
                elif not detected and actual:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = f"{metric_name} >= {thresh}"
            print(f"{metric_name:<20} {thresh:>6} {tp:>4} {fp:>4} {fn:>4} {prec:>6.0%} {rec:>6.0%} {f1:>6.2f}")
        print()

    print(f"Best F1 = {best_f1:.2f} at {best_cfg}")


if __name__ == "__main__":
    run_sar_backtest()
