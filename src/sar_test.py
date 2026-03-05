"""
Sentinel-1 SAR test: compare radar backscatter for active vs inactive drilling pads.
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
from backtest import load_all_wells, cluster_to_pads, assign_pad_drilling_windows, get_ground_truth


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


def get_token():
    resp = requests.post(SH_TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": SH_CLIENT_ID,
        "client_secret": SH_CLIENT_SECRET,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_sar(lon, lat, scene_date, token, half=0.003):
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
    resp = requests.post(SH_PROCESS_URL, json=payload,
        headers={"Authorization": f"Bearer {token}", "Accept": "image/tiff"}, timeout=120)
    if resp.status_code != 200:
        print(f"    SAR error {resp.status_code}: {resp.text[:200]}")
        return None
    with rasterio.open(io.BytesIO(resp.content)) as src:
        data = src.read()
    vv, vh = data[0], data[1]
    valid = vv > 0
    if valid.sum() < 100:
        return None
    vv_db = 10 * np.log10(vv + 1e-10)
    vh_db = 10 * np.log10(vh + 1e-10)
    cr = vh / (vv + 1e-10)
    cy, cx = 30, 30
    r = 3
    center_vv = vv_db[cy - r:cy + r, cx - r:cx + r]
    center_vh = vh_db[cy - r:cy + r, cx - r:cx + r]
    bg_mask = np.ones_like(vv_db, dtype=bool)
    bg_mask[cy - r:cy + r, cx - r:cx + r] = False
    bg_mask &= valid
    return {
        "vv_mean": np.mean(vv_db[valid]), "vh_mean": np.mean(vh_db[valid]),
        "vv_max": np.max(vv_db[valid]), "vh_max": np.max(vh_db[valid]),
        "vv_center": np.mean(center_vv), "vh_center": np.mean(center_vh),
        "vv_bg": np.mean(vv_db[bg_mask]), "vh_bg": np.mean(vh_db[bg_mask]),
        "vv_contrast": np.mean(center_vv) - np.mean(vv_db[bg_mask]),
        "vh_contrast": np.mean(center_vh) - np.mean(vh_db[bg_mask]),
        "cr_mean": np.mean(cr[valid]), "cr_center": np.mean(cr[cy-r:cy+r, cx-r:cx+r]),
        "vv_std": np.std(vv_db[valid]),
        "valid_pct": valid.sum() / valid.size * 100,
    }


def main():
    token = get_token()
    print("Token acquired")
    wells = load_all_wells()
    wells = cluster_to_pads(wells)
    wells = assign_pad_drilling_windows(wells)
    dec_truth = get_ground_truth(wells, 2025, 12)
    active_pad_ids = set(dec_truth.keys())
    pad_info = wells.groupby("pad_id").agg(
        pad_lon=("pad_lon", "first"), pad_lat=("pad_lat", "first"),
    ).reset_index()
    active_pads = pad_info[pad_info["pad_id"].isin(active_pad_ids)]
    inactive_pads = pad_info[~pad_info["pad_id"].isin(active_pad_ids)]
    print(f"Active pads: {len(active_pads)}, Inactive pads: {len(inactive_pads)}")
    np.random.seed(42)
    i_idx = np.random.choice(len(inactive_pads), min(15, len(inactive_pads)), replace=False)
    a_sample = active_pads
    i_sample = inactive_pads.iloc[i_idx]
    scene_date = date(2025, 12, 15)
    baseline_date = date(2025, 11, 15)
    print(f"Downloading SAR for {len(a_sample)} active + {len(i_sample)} inactive pads...")
    results = []
    for label, sample in [("ACTIVE", a_sample), ("INACTIVE", i_sample)]:
        for _, row in sample.iterrows():
            lon, lat = row["pad_lon"], row["pad_lat"]
            cur = download_sar(lon, lat, scene_date, token)
            base = download_sar(lon, lat, baseline_date, token)
            if cur is None:
                print(f"  {label} ({lon:.4f},{lat:.4f}): NO SAR DATA")
                continue
            delta_vv = cur["vv_mean"] - (base["vv_mean"] if base else cur["vv_mean"])
            delta_contrast = cur["vv_contrast"] - (base["vv_contrast"] if base else 0)
            results.append({
                "label": label, "lon": lon, "lat": lat,
                "vv_mean": cur["vv_mean"], "vh_mean": cur["vh_mean"],
                "vv_max": cur["vv_max"], "vh_max": cur["vh_max"],
                "vv_center": cur["vv_center"], "vv_bg": cur["vv_bg"],
                "vv_contrast": cur["vv_contrast"], "vh_contrast": cur["vh_contrast"],
                "cr_mean": cur["cr_mean"], "cr_center": cur["cr_center"],
                "vv_std": cur["vv_std"],
                "delta_vv": delta_vv, "delta_contrast": delta_contrast,
            })
            vvm = cur["vv_mean"]
            vvx = cur["vv_max"]
            vvc = cur["vv_contrast"]
            print(f"  {label} ({lon:.4f},{lat:.4f}): VV={vvm:.1f}dB max={vvx:.1f}dB contrast={vvc:+.1f}dB")
            time.sleep(0.3)
    print_summary(results)


def print_summary(results):
    print()
    print("=" * 80)
    print("SENTINEL-1 SAR COMPARISON: ACTIVE vs INACTIVE PADS (Dec 2025)")
    print("=" * 80)
    active_r = [r for r in results if r["label"] == "ACTIVE"]
    inactive_r = [r for r in results if r["label"] == "INACTIVE"]
    print(f"Samples: {len(active_r)} active, {len(inactive_r)} inactive")
    if not active_r or not inactive_r:
        print("NOT ENOUGH DATA")
        return
    print()
    hdr = f"{"Metric":<16} {"Active":>10} {"Inactive":>10} {"Diff":>10} {"Effect(d)":>10} {"Signal?":>10}"
    print(hdr)
    print("-" * 66)
    metrics = ["vv_mean", "vh_mean", "vv_max", "vh_max", "vv_center", "vv_bg",
        "vv_contrast", "vh_contrast", "cr_mean", "cr_center", "vv_std",
        "delta_vv", "delta_contrast"]
    for metric in metrics:
        a_vals = np.array([r[metric] for r in active_r])
        i_vals = np.array([r[metric] for r in inactive_r])
        a_mean = np.mean(a_vals)
        i_mean = np.mean(i_vals)
        diff = a_mean - i_mean
        pooled_std = np.sqrt((np.std(a_vals)**2 + np.std(i_vals)**2) / 2) + 1e-10
        d = abs(diff) / pooled_std
        sig = "YES!" if d > 0.8 else "YES" if d > 0.5 else "maybe" if d > 0.3 else "no"
        print(f"{metric:<16} {a_mean:>10.2f} {i_mean:>10.2f} {diff:>+10.2f} {d:>10.2f} {sig:>10}")
    print()
    print("--- VV Contrast (center - background) per pad ---")
    print("ACTIVE pads:")
    for r in sorted(active_r, key=lambda x: -x["vv_contrast"]):
        vc = r["vv_contrast"]
        print(f"  ({r["lon"]:.4f},{r["lat"]:.4f}): {vc:+.2f} dB")
    print("INACTIVE pads:")
    for r in sorted(inactive_r, key=lambda x: -x["vv_contrast"]):
        vc = r["vv_contrast"]
        print(f"  ({r["lon"]:.4f},{r["lat"]:.4f}): {vc:+.2f} dB")


if __name__ == "__main__":
    main()
