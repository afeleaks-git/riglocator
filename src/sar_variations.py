"""
SAR method variations for Dec 2025 backtest.
Tests: different center window sizes, SAR+optical combination.
Caches raw VV/VH rasters to avoid re-downloading.
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
OPTICAL_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

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


def compute_sar_metrics(vv, vh, center_r=3):
    """Compute SAR metrics with configurable center window radius."""
    valid = vv > 0
    if valid.sum() < 100:
        return None
    vv_db = 10 * np.log10(vv + 1e-10)
    vh_db = 10 * np.log10(vh + 1e-10)
    cy, cx = vv.shape[0] // 2, vv.shape[1] // 2
    r = center_r
    center_vv = vv_db[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
    center_vh = vh_db[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
    bg_mask = np.ones_like(vv_db, dtype=bool)
    bg_mask[max(0,cy-r):cy+r, max(0,cx-r):cx+r] = False
    bg_mask &= valid
    if bg_mask.sum() < 50:
        return None
    return {
        "vv_mean": float(np.mean(vv_db[valid])),
        "vv_max": float(np.max(vv_db[valid])),
        "vv_center": float(np.mean(center_vv)),
        "vv_bg": float(np.mean(vv_db[bg_mask])),
        "vv_contrast": float(np.mean(center_vv) - np.mean(vv_db[bg_mask])),
        "vh_contrast": float(np.mean(center_vh) - np.mean(vh_db[bg_mask])),
        "vv_std": float(np.std(vv_db[valid])),
    }


def load_optical_metrics(lon, lat, scene_dates):
    """Load cached optical brightness from the S2 cache."""
    brightnesses = []
    for sd in scene_dates:
        key = f"{lon:.5f}_{lat:.5f}_{sd.isoformat()}"
        path = os.path.join(OPTICAL_CACHE, f"{key}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        if "red" not in d.files:
            d.close()
            continue
        red = d["red"]
        if np.nanmax(red) < 0.01:
            d.close()
            continue
        green = d["green"]
        blue = d["blue"]
        nir = d["nir"]
        brightness = (red + green + blue + nir) / 4.0
        brightnesses.append(float(np.nanmax(brightness)))
        d.close()
    if not brightnesses:
        return None
    return {"max_brightness": max(brightnesses), "mean_brightness": np.mean(brightnesses)}


def sweep_threshold(pad_data, metric_name, thresholds):
    """Sweep a single metric and return best config."""
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        for pm in pad_data:
            val = pm.get(metric_name)
            if val is None:
                if pm["is_active"]:
                    fn += 1
                continue
            detected = val >= t
            if detected and pm["is_active"]:
                tp += 1
            elif detected and not pm["is_active"]:
                fp += 1
            elif not detected and pm["is_active"]:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
        rows.append((metric_name, t, tp, fp, fn, prec, rec, f1))
    return rows


def sweep_combined(pad_data, sar_metric, sar_thresh, opt_metric, opt_thresh, mode="AND"):
    """Sweep a combined SAR+optical detector."""
    tp = fp = fn = 0
    for pm in pad_data:
        sv = pm.get(sar_metric)
        ov = pm.get(opt_metric)
        if sv is None:
            if pm["is_active"]:
                fn += 1
            continue
        sar_det = sv >= sar_thresh
        opt_det = ov is not None and ov >= opt_thresh
        if mode == "AND":
            detected = sar_det and opt_det
        else:
            detected = sar_det or opt_det
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
    year, month = 2025, 12
    print(f"=== SAR VARIATIONS BACKTEST {year}-{month:02d} ===")

    wells = load_all_wells()
    wells = cluster_to_pads(wells)
    wells = assign_pad_drilling_windows(wells)
    truth = get_ground_truth(wells, year, month)
    truth_ids = set(truth.keys())

    watchlist = build_watchlist(wells, year, month)
    watch_pads = watchlist.groupby("pad_id").agg(
        pad_lon=("pad_lon", "first"), pad_lat=("pad_lat", "first"),
    ).reset_index()
    print(f"Truth: {len(truth_ids)} active pads, Watchlist: {len(watch_pads)} pads")

    # SAR scene dates (12-day revisit)
    sar_dates = [date(2025,12,1), date(2025,12,13), date(2025,12,25)]
    baseline_date = date(2025, 11, 16)
    # Optical scene dates (5-day revisit)
    opt_dates = []
    d = date(2025, 12, 1)
    while d <= date(2025, 12, 31):
        opt_dates.append(d)
        d += timedelta(days=5)
    opt_baseline_dates = [date(2025,11,1), date(2025,11,11), date(2025,11,21)]

    # Download all raw SAR tiles
    print(f"Downloading SAR tiles ({len(watch_pads)} pads x {len(sar_dates)+1} dates)...")
    raw_data = {}  # (pad_id, date) -> (vv, vh)
    total = len(watch_pads)
    for i, (_, row) in enumerate(watch_pads.iterrows()):
        pid = row["pad_id"]
        lon, lat = row["pad_lon"], row["pad_lat"]
        for sd in [baseline_date] + sar_dates:
            vv, vh = download_raw_sar(lon, lat, sd)
            if vv is not None:
                raw_data[(pid, sd)] = (vv, vh)
            time.sleep(0.15)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{total}] downloaded")

    print(f"Total cached SAR tiles: {len(raw_data)}")

    # Test different center window radii
    window_radii = [2, 3, 5, 7, 10]
    sar_thresholds = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

    print()
    print("=" * 80)
    print("PART 1: CENTER WINDOW SIZE SWEEP")
    print("=" * 80)

    best_overall_f1 = 0
    best_overall_cfg = ""

    for radius in window_radii:
        window_px = radius * 2
        # At ~10m/px, window covers radius*2*10 = window_px*10 meters diameter
        window_m = window_px * 10
        print(f"--- Window radius={radius}px ({window_m}m diameter) ---")

        # Compute metrics for each pad
        pad_data = []
        for _, row in watch_pads.iterrows():
            pid = row["pad_id"]
            # Baseline metrics
            base_key = (pid, baseline_date)
            base_contrast = 0
            if base_key in raw_data:
                bm = compute_sar_metrics(raw_data[base_key][0], raw_data[base_key][1], radius)
                if bm:
                    base_contrast = bm["vv_contrast"]

            # Scene metrics
            contrasts = []
            delta_contrasts = []
            for sd in sar_dates:
                sk = (pid, sd)
                if sk not in raw_data:
                    continue
                sm = compute_sar_metrics(raw_data[sk][0], raw_data[sk][1], radius)
                if sm is None:
                    continue
                contrasts.append(sm["vv_contrast"])
                delta_contrasts.append(sm["vv_contrast"] - base_contrast)

            entry = {"pad_id": pid, "is_active": pid in truth_ids}
            if contrasts:
                entry["max_contrast"] = max(contrasts)
                entry["max_delta_contrast"] = max(delta_contrasts)
            pad_data.append(entry)

        # Sweep thresholds for this window size
        for metric in ["max_contrast", "max_delta_contrast"]:
            rows = sweep_threshold(pad_data, metric, sar_thresholds)
            best = max(rows, key=lambda x: x[7])
            bname, bt, btp, bfp, bfn, bprec, brec, bf1 = best
            print(f"  {metric}: best F1={bf1:.2f} at >={bt}dB (P={bprec:.0%} R={brec:.0%} TP={btp} FP={bfp} FN={bfn})")
            if bf1 > best_overall_f1:
                best_overall_f1 = bf1
                best_overall_cfg = f"r={radius} {metric}>={bt}"

    print(f"Best overall: F1={best_overall_f1:.2f} at {best_overall_cfg}")

    # PART 2: SAR + Optical combination
    print()
    print("=" * 80)
    print("PART 2: SAR + OPTICAL COMBINATION")
    print("=" * 80)

    # Use r=3 (original) for SAR, load optical
    pad_data_combo = []
    opt_found = 0
    for _, row in watch_pads.iterrows():
        pid = row["pad_id"]
        lon, lat = row["pad_lon"], row["pad_lat"]

        # SAR metrics (r=3)
        base_key = (pid, baseline_date)
        base_contrast = 0
        if base_key in raw_data:
            bm = compute_sar_metrics(raw_data[base_key][0], raw_data[base_key][1], 3)
            if bm:
                base_contrast = bm["vv_contrast"]
        contrasts = []
        delta_contrasts = []
        for sd in sar_dates:
            sk = (pid, sd)
            if sk not in raw_data:
                continue
            sm = compute_sar_metrics(raw_data[sk][0], raw_data[sk][1], 3)
            if sm is None:
                continue
            contrasts.append(sm["vv_contrast"])
            delta_contrasts.append(sm["vv_contrast"] - base_contrast)

        entry = {"pad_id": pid, "is_active": pid in truth_ids}
        if contrasts:
            entry["max_contrast"] = max(contrasts)
            entry["max_delta_contrast"] = max(delta_contrasts)

        # Optical metrics from existing cache
        opt = load_optical_metrics(lon, lat, opt_dates)
        if opt:
            entry["max_brightness"] = opt["max_brightness"]
            opt_found += 1

        # Optical baseline
        opt_base = load_optical_metrics(lon, lat, opt_baseline_dates)
        if opt and opt_base:
            entry["delta_brightness"] = opt["max_brightness"] - opt_base["max_brightness"]

        pad_data_combo.append(entry)

    print(f"Pads with optical data: {opt_found}/{len(watch_pads)}")

    # Sweep SAR-only for comparison
    print()
    print("SAR-only (r=3) baseline:")
    for metric in ["max_contrast", "max_delta_contrast"]:
        rows = sweep_threshold(pad_data_combo, metric, sar_thresholds)
        best = max(rows, key=lambda x: x[7])
        _, bt, btp, bfp, bfn, bp, br, bf = best
        print(f"  {metric}: best F1={bf:.2f} at >={bt}dB (P={bp:.0%} R={br:.0%})")

    # Sweep combined SAR+optical
    print()
    print("Combined SAR + Optical (AND mode = both must trigger):")
    print(f"{"Config":<45} {"TP":>4} {"FP":>4} {"FN":>4} {"Prec":>6} {"Rec":>6} {"F1":>6}")
    print("-" * 71)

    best_combo_f1 = 0
    best_combo_cfg = ""
    sar_sweeps = [("max_delta_contrast", t) for t in [3, 5, 7, 10]]
    opt_sweeps = [("max_brightness", t) for t in [0.3, 0.35, 0.4, 0.45, 0.5]]
    opt_sweeps += [("delta_brightness", t) for t in [0.02, 0.03, 0.04, 0.05]]

    for sm, st in sar_sweeps:
        for om, ot in opt_sweeps:
            for mode in ["AND", "OR"]:
                tp, fp, fn, prec, rec, f1 = sweep_combined(pad_data_combo, sm, st, om, ot, mode)
                cfg = f"{sm}>={st} {mode} {om}>={ot}"
                if f1 > best_combo_f1:
                    best_combo_f1 = f1
                    best_combo_cfg = cfg
                if f1 >= 0.55:
                    print(f"{cfg:<45} {tp:>4} {fp:>4} {fn:>4} {prec:>6.0%} {rec:>6.0%} {f1:>6.2f}")

    print()
    print(f"Best combined F1 = {best_combo_f1:.2f} at {best_combo_cfg}")


if __name__ == "__main__":
    main()
