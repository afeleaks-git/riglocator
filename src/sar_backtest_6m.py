"""
6-month SAR backtest: Jul-Dec 2025.
Uses Sentinel-1 VV delta_contrast to detect drilling rigs.
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

RAW_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "sar_raw")

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


def download_raw_sar(lon, lat, scene_date, half=0.003):
    """Download raw VV/VH raster and cache as NPZ."""
    os.makedirs(RAW_CACHE, exist_ok=True)
    key = f"raw_{lon:.5f}_{lat:.5f}_{scene_date.isoformat()}"
    path = os.path.join(RAW_CACHE, f"{key}.npz")
    if os.path.exists(path):
        d = np.load(path)
        if "vv" in d.files:
            vv, vh = d["vv"], d["vh"]
            d.close()
            return vv, vh
        d.close()
        return None, None
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
    token = get_token()
    resp = requests.post(SH_PROCESS_URL, json=payload,
        headers={"Authorization": f"Bearer {token}", "Accept": "image/tiff"}, timeout=120)
    if resp.status_code == 401:
        _token_cache["token"] = None
        token = get_token()
        resp = requests.post(SH_PROCESS_URL, json=payload,
            headers={"Authorization": f"Bearer {token}", "Accept": "image/tiff"}, timeout=120)
    if resp.status_code != 200:
        np.savez(path, empty=True)
        return None, None
    with rasterio.open(io.BytesIO(resp.content)) as src:
        data = src.read()
    vv, vh = data[0], data[1]
    if (vv > 0).sum() < 100:
        np.savez(path, empty=True)
        return None, None
    np.savez(path, vv=vv, vh=vh)
    return vv, vh


def compute_metrics(vv, vh, center_r=3):
    valid = vv > 0
    if valid.sum() < 100:
        return None
    vv_db = 10 * np.log10(vv + 1e-10)
    cy, cx = vv.shape[0] // 2, vv.shape[1] // 2
    r = center_r
    center_vv = vv_db[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
    bg_mask = np.ones_like(vv_db, dtype=bool)
    bg_mask[max(0,cy-r):cy+r, max(0,cx-r):cx+r] = False
    bg_mask &= valid
    if bg_mask.sum() < 50:
        return None
    return {
        "vv_contrast": float(np.mean(center_vv) - np.mean(vv_db[bg_mask])),
        "vv_max": float(np.max(vv_db[valid])),
        "vv_mean": float(np.mean(vv_db[valid])),
    }


def run_month(wells, year, month, quiet=False):
    """Run SAR backtest for a single month. Returns pad_data list."""
    truth = get_ground_truth(wells, year, month)
    truth_ids = set(truth.keys())
    watchlist = build_watchlist(wells, year, month)
    watch_pads = watchlist.groupby("pad_id").agg(
        pad_lon=("pad_lon", "first"), pad_lat=("pad_lat", "first"),
    ).reset_index()

    if not quiet:
        print(f"  {year}-{month:02d}: {len(truth_ids)} active pads, {len(watch_pads)} watchlist pads")

    # SAR scene dates (~12 day revisit)
    scene_dates = []
    d = date(year, month, 1)
    if month == 12:
        end_d = date(year, 12, 31)
    else:
        end_d = date(year, month + 1, 1) - timedelta(days=1)
    while d <= end_d:
        scene_dates.append(d)
        d += timedelta(days=12)
    baseline_date = date(year, month, 1) - timedelta(days=15)

    # Download tiles
    total = len(watch_pads)
    raw_data = {}
    for i, (_, row) in enumerate(watch_pads.iterrows()):
        pid = row["pad_id"]
        lon, lat = row["pad_lon"], row["pad_lat"]
        for sd in [baseline_date] + scene_dates:
            vv, vh = download_raw_sar(lon, lat, sd)
            if vv is not None:
                raw_data[(pid, sd)] = (vv, vh)
            time.sleep(0.1)
        if not quiet and (i+1) % 20 == 0:
            print(f"    [{i+1}/{total}] downloaded")

    # Compute per-pad metrics
    pad_data = []
    for _, row in watch_pads.iterrows():
        pid = row["pad_id"]
        base_key = (pid, baseline_date)
        base_contrast = 0
        if base_key in raw_data:
            bm = compute_metrics(raw_data[base_key][0], raw_data[base_key][1])
            if bm:
                base_contrast = bm["vv_contrast"]
        contrasts = []
        delta_contrasts = []
        for sd in scene_dates:
            sk = (pid, sd)
            if sk not in raw_data:
                continue
            sm = compute_metrics(raw_data[sk][0], raw_data[sk][1])
            if sm is None:
                continue
            contrasts.append(sm["vv_contrast"])
            delta_contrasts.append(sm["vv_contrast"] - base_contrast)
        entry = {"pad_id": pid, "is_active": pid in truth_ids}
        if contrasts:
            entry["max_contrast"] = max(contrasts)
            entry["max_delta_contrast"] = max(delta_contrasts)
        pad_data.append(entry)
    return pad_data, len(truth_ids)


def score(pad_data, metric, thresh):
    tp = fp = fn = 0
    for pm in pad_data:
        val = pm.get(metric)
        if val is None:
            if pm["is_active"]:
                fn += 1
            continue
        detected = val >= thresh
        if detected and pm["is_active"]:
            tp += 1
        elif detected and not pm["is_active"]:
            fp += 1
        elif not detected and pm["is_active"]:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
    return tp, fp, fn, prec, rec, f1


def main():
    print("=== 6-MONTH SAR BACKTEST: Jul-Dec 2025 ===")
    print()

    wells = load_all_wells()
    wells = cluster_to_pads(wells)
    wells = assign_pad_drilling_windows(wells)
    print(f"Loaded {len(wells)} wells")
    print()

    # Run each month
    monthly_data = {}
    for month in range(7, 13):
        print(f"--- Processing {2025}-{month:02d} ---")
        pad_data, n_truth = run_month(wells, 2025, month)
        monthly_data[month] = (pad_data, n_truth)
        print(f"    Done: {len(pad_data)} pads evaluated")
        print()

    # Results table per month at key thresholds
    thresholds = [5, 7, 10, 15]
    metric = "max_delta_contrast"

    print("=" * 90)
    print("RESULTS BY MONTH (max_delta_contrast)")
    print("=" * 90)
    hdr = f"{"Month":>7} {"Truth":>6} {"Thresh":>7} {"TP":>4} {"FP":>4} {"FN":>4} {"Prec":>6} {"Rec":>6} {"F1":>6}"
    print(hdr)
    print("-" * 55)

    for month in range(7, 13):
        pad_data, n_truth = monthly_data[month]
        for t in thresholds:
            tp, fp, fn, prec, rec, f1 = score(pad_data, metric, t)
            label = f"2025-{month:02d}"
            print(f"{label:>7} {n_truth:>6} {t:>7} {tp:>4} {fp:>4} {fn:>4} {prec:>6.0%} {rec:>6.0%} {f1:>6.2f}")
        print()

    # Aggregate across all months
    print("=" * 90)
    print("AGGREGATE ACROSS ALL 6 MONTHS")
    print("=" * 90)
    all_data = []
    for month in range(7, 13):
        all_data.extend(monthly_data[month][0])

    print(f"Total pad-months evaluated: {len(all_data)}")
    total_truth = sum(monthly_data[m][1] for m in range(7, 13))
    print(f"Total active pad-months: {total_truth}")
    print()

    sweep_thresholds = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    print(f"{"Metric":<22} {"Thresh":>7} {"TP":>4} {"FP":>4} {"FN":>4} {"Prec":>6} {"Rec":>6} {"F1":>6}")
    print("-" * 55)

    best_f1 = 0
    best_cfg = ""
    for m in ["max_delta_contrast", "max_contrast"]:
        for t in sweep_thresholds:
            tp, fp, fn, prec, rec, f1 = score(all_data, m, t)
            print(f"{m:<22} {t:>7} {tp:>4} {fp:>4} {fn:>4} {prec:>6.0%} {rec:>6.0%} {f1:>6.2f}")
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = f"{m} >= {t}"
        print()

    print(f"Best aggregate F1 = {best_f1:.2f} at {best_cfg}")

    # Save results CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for month in range(7, 13):
        pad_data, n_truth = monthly_data[month]
        for t in sweep_thresholds:
            tp, fp, fn, prec, rec, f1 = score(pad_data, "max_delta_contrast", t)
            rows.append({"month": f"2025-{month:02d}", "threshold": t,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "sar_backtest_6m_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
