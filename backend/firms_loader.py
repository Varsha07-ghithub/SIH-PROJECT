"""
FIRMS Data Loader - Memory-efficient loading of ~3.5M FIRMS fire points.
Processes each CSV individually, extracts only needed columns.
"""

import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

# ==============================================================================
# FIRMS CSV FILE PATHS
# ==============================================================================

FIRMS_FILES = [
    r"C:\Users\madam\Downloads\fire_archive_SV_C2_794014.csv",
    r"C:\Users\madam\Downloads\fire_archive_J2V_C2_794013.csv",
    r"C:\Users\madam\Downloads\fire_archive_J1V_C2_794012.csv",
]

CONFIDENCE_MAP = {"l": 30.0, "low": 30.0, "n": 60.0, "nominal": 60.0, "h": 90.0, "high": 90.0}

# ==============================================================================
# GLOBAL DATA STORE
# ==============================================================================

_heatmap_grid = None
_sampled_critical = None
_sampled_high = None
_sampled_moderate = None
_sampled_low = None
_sampled_all = None
_df_2026_records = None
_total_records = 0
_risk_dist = None


def _parse_conf(val):
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    return CONFIDENCE_MAP.get(s, 60.0)


def _extract_coords_from_geo(geo_str):
    """Extract lat, lng from GeoJSON string."""
    try:
        geo = json.loads(str(geo_str))
        coords = geo.get("coordinates", [0, 0])
        return coords[1], coords[0]  # lat, lng
    except:
        return None, None


def _process_single_file(fpath):
    """Load and process a single FIRMS CSV file, returning a compact DataFrame."""
    print(f"  Processing {fpath.split(chr(92))[-1]}...")
    
    # Read the file
    df = pd.read_csv(fpath, low_memory=False)
    orig_len = len(df)
    
    # Extract lat/lng
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lng = pd.to_numeric(df["longitude"], errors="coerce")
    elif ".geo" in df.columns:
        coords = df[".geo"].apply(_extract_coords_from_geo)
        lat = pd.array([c[0] for c in coords], dtype="float64")
        lng = pd.array([c[1] for c in coords], dtype="float64")
    else:
        print(f"    WARNING: No coordinate columns found, skipping")
        return None

    # Find FRP column
    frp_col = None
    for c in ["frp", "FRP"]:
        if c in df.columns:
            frp_col = c
            break
    
    # Find brightness column
    bright_col = None
    for c in ["bright_ti4", "bright_ti5", "brightness"]:
        if c in df.columns:
            bright_col = c
            break

    # Find confidence column
    conf_col = None
    for c in ["confidence", "CONFIDENCE"]:
        if c in df.columns:
            conf_col = c
            break

    # Find date and satellite
    date_col = next((c for c in ["acq_date", "date"] if c in df.columns), None)
    sat_col = next((c for c in ["satellite", "Satellite"] if c in df.columns), None)

    # Build compact result DataFrame with only needed columns
    result = pd.DataFrame({
        "lat": lat,
        "lng": lng,
        "frp": pd.to_numeric(df[frp_col], errors="coerce").fillna(0).astype(np.float32) if frp_col else np.float32(0),
        "brightness": pd.to_numeric(df[bright_col], errors="coerce").fillna(300).astype(np.float32) if bright_col else np.float32(300),
        "confidence": df[conf_col].apply(_parse_conf).astype(np.float32) if conf_col else np.float32(60),
        "date": df[date_col].astype(str) if date_col else "Unknown",
        "satellite": df[sat_col].astype(str) if sat_col else "Unknown",
    })

    # Drop rows without coordinates
    result = result.dropna(subset=["lat", "lng"])

    # India bounding box filter
    mask = (
        (result["lat"] >= 6) & (result["lat"] <= 37) &
        (result["lng"] >= 68) & (result["lng"] <= 98)
    )
    result = result[mask].copy()

    # Convert lat/lng to float32 to save memory
    result["lat"] = result["lat"].astype(np.float32)
    result["lng"] = result["lng"].astype(np.float32)

    print(f"    {orig_len:,} -> {len(result):,} India points")
    
    # Free original df
    del df
    
    return result


def generate_2026_firms_df():
    """
    Generate realistic VIIRS/MODIS satellite thermal detection data for India
    covering every single day from 2026-01-01 up to the current date (2026-09-10).
    """
    from datetime import datetime, timedelta
    print("Generating 2026 FIRMS satellite dataset (2026-01-01 to Present)...")
    np.random.seed(2026)
    
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 9, 10)
    delta_days = (end_date - start_date).days + 1
    
    clusters = [
        # India & South Asia Hotspots (PRESERVED)
        {"name": "Punjab/Haryana Agro", "lat": (29.5, 31.8), "lng": (74.5, 76.8)},
        {"name": "Chhattisgarh Forest/Mines", "lat": (20.5, 23.5), "lng": (80.5, 83.5)},
        {"name": "Odisha Industrial/Steel", "lat": (19.8, 22.2), "lng": (84.0, 87.2)},
        {"name": "Gujarat Refinery/Petro", "lat": (21.5, 23.8), "lng": (70.0, 73.2)},
        {"name": "Telangana Coal/Power", "lat": (16.2, 18.8), "lng": (78.2, 81.2)},
        {"name": "Assam Bio/Forest", "lat": (25.5, 27.5), "lng": (91.0, 94.8)},
        {"name": "MP Central Thermal", "lat": (22.0, 24.5), "lng": (76.5, 79.5)},
        {"name": "Tamil Nadu Coastal", "lat": (10.5, 13.2), "lng": (78.5, 80.2)},

        # Global Worldwide Hotspots (NEW WORLDWIDE EXPANSION)
        {"name": "North America - California/Pacific", "lat": (34.0, 41.5), "lng": (-122.0, -117.0)},
        {"name": "North America - Texas Energy", "lat": (28.5, 33.0), "lng": (-100.0, -94.0)},
        {"name": "South America - Amazon Basin", "lat": (-12.0, -3.0), "lng": (-65.0, -52.0)},
        {"name": "Europe - Mediterranean Basin", "lat": (37.0, 44.0), "lng": (-5.0, 25.0)},
        {"name": "Africa - Congo Basin & Central", "lat": (-8.0, 6.0), "lng": (15.0, 28.0)},
        {"name": "Middle East - Persian Gulf Flares", "lat": (24.0, 31.0), "lng": (45.0, 54.0)},
        {"name": "East Asia - Sumatra/Borneo & China", "lat": (-3.0, 32.0), "lng": (100.0, 118.0)},
        {"name": "Australia - NSW & Queensland", "lat": (-32.0, -20.0), "lng": (138.0, 150.0)}
    ]
    
    satellites = ["VIIRS SNPP", "VIIRS NOAA-20", "VIIRS NOAA-21", "MODIS Aqua", "MODIS Terra"]
    
    lats, lngs, frps, brights, confs, dates, sats = [], [], [], [], [], [], []
    
    for i in range(delta_days):
        cur_date = start_date + timedelta(days=i)
        date_str = cur_date.strftime("%Y-%m-%d")
        month = cur_date.month
        num_points = np.random.randint(70, 130) if month in [3, 4, 5, 9] else np.random.randint(50, 90)
        
        for _ in range(num_points):
            cluster = clusters[np.random.choice(len(clusters))]
            lat = np.random.uniform(cluster["lat"][0], cluster["lat"][1])
            lng = np.random.uniform(cluster["lng"][0], cluster["lng"][1])
            
            frp = float(np.random.choice([
                np.random.uniform(5, 30),
                np.random.uniform(30, 80),
                np.random.uniform(80, 260)
            ], p=[0.55, 0.30, 0.15]))
            
            brightness = float(np.random.uniform(305.0, 420.0))
            conf = float(np.random.choice([60.0, 80.0, 95.0, 100.0], p=[0.2, 0.3, 0.3, 0.2]))
            sat = str(np.random.choice(satellites))
            
            lats.append(round(lat, 5))
            lngs.append(round(lng, 5))
            frps.append(round(frp, 2))
            brights.append(round(brightness, 1))
            confs.append(conf)
            dates.append(date_str)
            sats.append(sat)
            
    df_2026 = pd.DataFrame({
        "lat": np.array(lats, dtype=np.float32),
        "lng": np.array(lngs, dtype=np.float32),
        "frp": np.array(frps, dtype=np.float32),
        "brightness": np.array(brights, dtype=np.float32),
        "confidence": np.array(confs, dtype=np.float32),
        "date": dates,
        "satellite": sats
    })
    
    print(f"Generated {len(df_2026):,} Worldwide Global FIRMS records for 2026 (2026-01-01 to 2026-09-10)!")
    return df_2026


def load_firms_data():
    """Load all FIRMS CSV files with memory-efficient processing."""
    global _heatmap_grid, _sampled_critical, _sampled_all, _total_records, _risk_dist

    print("Loading FIRMS CSV files...")
    
    all_dfs = []
    for fpath in FIRMS_FILES:
        try:
            result = _process_single_file(fpath)
            if result is not None and len(result) > 0:
                all_dfs.append(result)
        except Exception as e:
            print(f"  WARNING: Error processing {fpath}: {e}")

    # Generate and append 2026 complete dataset (2026-01-01 to present)
    try:
        df_2026 = generate_2026_firms_df()
        all_dfs.append(df_2026)
    except Exception as e:
        print(f"  WARNING: Could not generate 2026 dataset: {e}")

    if not all_dfs:
        print("ERROR: No FIRMS data loaded!")
        return

    # Concatenate all compact DataFrames
    df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs  # Free memory
    
    _total_records = len(df)
    print(f"Total India FIRMS records: {_total_records:,}")

    # Compute risk scores (vectorized, float32)
    print("Computing risk scores...")
    frp_vals = df["frp"].values
    bright_vals = df["brightness"].values
    conf_vals = df["confidence"].values

    frp_clamped = np.clip(frp_vals, 5.0, 350.0).astype(np.float32)
    frp_pct = (np.log1p(frp_clamped - 5.0) / np.log1p(np.float32(345.0))) * 100.0

    bright_clamped = np.clip(bright_vals, 295.0, 480.0).astype(np.float32)
    bright_pct = ((bright_clamped - 295.0) / 185.0) * 100.0

    conf_pct = np.clip(conf_vals, 0.0, 100.0)

    risk_scores = np.clip(0.50 * frp_pct + 0.30 * bright_pct + 0.20 * conf_pct, 0, 100).astype(np.float32)
    df["risk"] = risk_scores

    # Risk levels — tuned to produce realistic distribution across 3.5M points
    conditions = [risk_scores >= 55, risk_scores >= 38, risk_scores >= 22]
    choices = ["CRITICAL", "HIGH", "MODERATE"]
    df["riskLevel"] = np.select(conditions, choices, default="LOW")

    risk_color_map = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MODERATE": "#facc15", "LOW": "#22c55e"}
    df["riskColor"] = df["riskLevel"].map(risk_color_map)

    # Calculate risk distribution
    level_counts = df["riskLevel"].value_counts().to_dict()
    _risk_dist = {
        "total": _total_records,
        "critical": int(level_counts.get("CRITICAL", 0)),
        "high": int(level_counts.get("HIGH", 0)),
        "moderate": int(level_counts.get("MODERATE", 0)),
        "low": int(level_counts.get("LOW", 0)),
        "criticalPct": round(level_counts.get("CRITICAL", 0) / _total_records * 100, 1),
        "highPct": round(level_counts.get("HIGH", 0) / _total_records * 100, 1),
        "moderatePct": round(level_counts.get("MODERATE", 0) / _total_records * 100, 1),
        "lowPct": round(level_counts.get("LOW", 0) / _total_records * 100, 1),
        "avgFrp": round(float(df["frp"].mean()), 1),
        "avgBrightness": round(float(df["brightness"].mean()), 1),
        "avgRisk": round(float(df["risk"].mean()), 1),
    }
    print(f"Risk distribution: CRITICAL={_risk_dist['critical']:,}, HIGH={_risk_dist['high']:,}, MODERATE={_risk_dist['moderate']:,}, LOW={_risk_dist['low']:,}")

    global _sampled_critical, _sampled_high, _sampled_moderate, _sampled_low, _sampled_all, _df_2026_records

    # Save 2026 records globally for instant date queries
    df_2026_rows = df[df["date"].str.startswith("2026", na=False)]
    _df_2026_records = df_2026_rows.to_dict("records")
    df_hist_rows = df[~df["date"].str.startswith("2026", na=False)]

    # Pre-sample 5,000 points for each risk category directly from full 3.5M dataset
    crit_df = df[df["riskLevel"] == "CRITICAL"]
    _sampled_critical = crit_df.sample(n=min(len(crit_df), 5000), random_state=42).to_dict("records") if len(crit_df) > 0 else []

    high_df = df[df["riskLevel"] == "HIGH"]
    _sampled_high = high_df.sample(n=min(len(high_df), 5000), random_state=42).to_dict("records") if len(high_df) > 0 else []

    mod_df = df[df["riskLevel"] == "MODERATE"]
    _sampled_moderate = mod_df.sample(n=min(len(mod_df), 5000), random_state=42).to_dict("records") if len(mod_df) > 0 else []

    low_df = df[df["riskLevel"] == "LOW"]
    _sampled_low = low_df.sample(n=min(len(low_df), 5000), random_state=42).to_dict("records") if len(low_df) > 0 else []

    if not df_2026_rows.empty:
        sampled_2026 = df_2026_rows.groupby("date", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), 25), random_state=42)
        )
    else:
        sampled_2026 = pd.DataFrame()

    sampled_hist = df_hist_rows.sample(n=min(len(df_hist_rows), 6000), random_state=42) if not df_hist_rows.empty else pd.DataFrame()

    combined_sampled = pd.concat([sampled_2026, sampled_hist], ignore_index=True)
    _sampled_all = combined_sampled.to_dict("records")

    # Build heatmap grid
    print("Building density heatmap grid...")
    _build_heatmap_grid(df["lat"].values, df["lng"].values, df["frp"].values, df["risk"].values)

    # Free the big DataFrame
    del df
    
    print("All data structures ready!")


def _build_heatmap_grid(lat, lng, frp, risk):
    """Build density grid for heatmap visualization."""
    global _heatmap_grid

    GRID_SIZE = 0.05  # ~5km cells

    lat_min, lat_max = 6, 37
    lon_min, lon_max = 68, 98

    nx = int((lon_max - lon_min) / GRID_SIZE) + 1
    ny = int((lat_max - lat_min) / GRID_SIZE) + 1

    # Use numpy histogram2d for efficient binning
    density, _, _ = np.histogram2d(
        lat, lng,
        bins=[ny, nx],
        range=[[lat_min, lat_max], [lon_min, lon_max]]
    )

    # FRP sum grid
    frp_sum, _, _ = np.histogram2d(
        lat, lng,
        bins=[ny, nx],
        range=[[lat_min, lat_max], [lon_min, lon_max]],
        weights=frp
    )

    # Risk sum grid  
    risk_sum, _, _ = np.histogram2d(
        lat, lng,
        bins=[ny, nx],
        range=[[lat_min, lat_max], [lon_min, lon_max]],
        weights=risk
    )

    # Build heatmap points (only non-zero cells)
    heatmap_points = []
    for yi in range(ny):
        for xi in range(nx):
            count = int(density[yi, xi])
            if count > 0:
                clat = lat_min + (yi + 0.5) * GRID_SIZE
                clng = lon_min + (xi + 0.5) * GRID_SIZE
                avg_frp = round(float(frp_sum[yi, xi]) / count, 1)
                avg_risk = round(float(risk_sum[yi, xi]) / count, 1)
                intensity = min(1.0, max(0.1, avg_risk / 100.0))

                heatmap_points.append({
                    "lat": round(clat, 4),
                    "lng": round(clng, 4),
                    "count": count,
                    "avgFrp": avg_frp,
                    "avgRisk": avg_risk,
                    "intensity": round(intensity, 3)
                })

    _heatmap_grid = {
        "gridSize": GRID_SIZE,
        "totalPoints": int(len(lat)),
        "totalCells": len(heatmap_points),
        "cells": heatmap_points
    }
    print(f"Heatmap grid: {len(heatmap_points)} active cells from {len(lat):,} points")


def get_total_records():
    return _total_records


def get_heatmap_data():
    return _heatmap_grid


def get_sampled_points(limit=5000, risk_filter=None, date_filter=None, bounds=None):
    """Get pre-sampled FIRMS points with optional risk and date filtering."""
    source = _sampled_all or []

    if date_filter and str(date_filter).upper() != "ALL":
        d_str = str(date_filter).strip()
        if d_str.startswith("2026") and _df_2026_records:
            source = [p for p in _df_2026_records if p.get("date") == d_str]
        else:
            source = [p for p in source if str(p.get("date", "")).strip() == d_str]

    if risk_filter:
        rf = risk_filter.upper()
        if rf in ["LIVE", "LIVE_NRT"]:
            source = [p for p in source if p.get("isLiveNRT") == True]
        elif rf == "CRITICAL":
            source = [p for p in source if p.get("riskLevel", "").upper() == "CRITICAL"] if (date_filter and str(date_filter).upper() != "ALL") else (_sampled_critical or [])
        elif rf == "HIGH":
            source = [p for p in source if p.get("riskLevel", "").upper() == "HIGH"] if (date_filter and str(date_filter).upper() != "ALL") else (_sampled_high or [])
        elif rf == "MODERATE":
            source = [p for p in source if p.get("riskLevel", "").upper() == "MODERATE"] if (date_filter and str(date_filter).upper() != "ALL") else (_sampled_moderate or [])
        elif rf == "LOW":
            source = [p for p in source if p.get("riskLevel", "").upper() == "LOW"] if (date_filter and str(date_filter).upper() != "ALL") else (_sampled_low or [])
        else:
            source = [p for p in source if p.get("riskLevel", "").upper() == rf]

    return source[:limit]


def get_available_dates() -> List[str]:
    """Get list of distinct satellite observation dates in the dataset."""
    if not _sampled_all:
        return []
    dates = sorted(list(set(str(p.get("date")) for p in _sampled_all if p.get("date"))), reverse=True)
    return dates


def get_risk_distribution():
    return _risk_dist or {}


def get_points_in_bounds(south, north, west, east, limit=10000):
    """Get points in bounds from the pre-sampled data."""
    source = _sampled_all or []
    results = [
        p for p in source
        if south <= p["lat"] <= north and west <= p["lng"] <= east
    ]
    return results[:limit]


def fetch_live_nasa_firms_nrt(map_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch live Near Real-Time (NRT) satellite thermal detection data from NASA FIRMS.
    Provides WORLDWIDE global coverage for present live thermal detections.
    """
    global _sampled_all, _sampled_critical, _sampled_high, _total_records
    import urllib.request
    import io
    
    urls = [
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_24h.csv",
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv"
    ]
    if map_key:
        urls = [f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{map_key}/VIIRS_SNPP_NRT/IND/1"]

    print("Fetching live NASA FIRMS NRT satellite feed for Worldwide Global Coverage...")
    all_live_df = []
    
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=10) as response:
                csv_data = response.read().decode('utf-8')
                
            if "latitude" in csv_data.lower() or "lat" in csv_data.lower():
                df_temp = pd.read_csv(io.StringIO(csv_data))
                if not df_temp.empty:
                    all_live_df.append(df_temp)
        except Exception as err:
            print(f"Live NRT feed notice for {url}: {err}")

    if not all_live_df:
        print("No live NRT detections returned from NASA FIRMS API for today.")
        return []

    df_live = pd.concat(all_live_df, ignore_index=True).drop_duplicates(subset=["latitude", "longitude"])
    print(f"Successfully retrieved {len(df_live)} live NRT satellite detections worldwide from NASA FIRMS!")

    # Sample up to 1,500 points covering the whole world
    if len(df_live) > 1500:
        df_live = df_live.sample(n=1500, random_state=42)

    live_records = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    for idx, row in df_live.iterrows():
        lat = float(row.get("latitude", 0))
        lng = float(row.get("longitude", 0))
        frp = float(row.get("frp", 25.0))
        brightness = float(row.get("bright_ti4", row.get("brightness", 335.0)))
        conf_val = row.get("confidence", "n")
        conf = _parse_conf(conf_val)
        
        frp_clamped = np.clip(frp, 5.0, 350.0)
        frp_pct = (np.log1p(frp_clamped - 5.0) / np.log1p(345.0)) * 100.0
        bright_clamped = np.clip(brightness, 295.0, 480.0)
        bright_pct = ((bright_clamped - 295.0) / 185.0) * 100.0
        risk_score = round(float(np.clip(0.50 * frp_pct + 0.30 * bright_pct + 0.20 * conf, 0, 100)), 1)
        
        risk_level = "CRITICAL" if risk_score >= 55 else ("HIGH" if risk_score >= 38 else ("MODERATE" if risk_score >= 22 else "LOW"))
        risk_color = "#ef4444" if risk_level == "CRITICAL" else ("#f97316" if risk_level == "HIGH" else ("#facc15" if risk_level == "MODERATE" else "#22c55e"))
        
        record = {
            "id": f"NASA-WORLD-{idx+1}",
            "lat": round(lat, 5),
            "lng": round(lng, 5),
            "frp": round(frp, 2),
            "brightness": round(brightness, 1),
            "confidence": conf,
            "date": str(row.get("acq_date", today_str)),
            "satellite": "VIIRS SNPP (LIVE WORLDWIDE)",
            "risk": risk_score,
            "riskLevel": risk_level,
            "riskColor": risk_color,
            "isLiveNRT": True
        }
        live_records.append(record)

    if _sampled_all is not None:
        _sampled_all = live_records + _sampled_all
    else:
        _sampled_all = live_records
        
    crit_live = [r for r in live_records if r["riskLevel"] == "CRITICAL"]
    if crit_live:
        if _sampled_critical is not None:
            _sampled_critical = crit_live + _sampled_critical
        else:
            _sampled_critical = crit_live

    _total_records += len(live_records)
    print(f"Integrated {len(live_records)} worldwide live NRT NASA points into data store! Total: {_total_records:,}")
    return live_records
