"""
Google Earth Engine - Dynamic World Land Cover Integration
Uses the real GEE API to fetch 10m resolution land cover data.
Falls back to heuristic estimation if GEE auth is not available.
"""

import sys
import traceback
from typing import Dict, Any

# Dynamic World class definitions
DW_CLASSES = {
    0: {"name": "Water", "color": "#419BDF", "desc": "Rivers, lakes, coastal waters"},
    1: {"name": "Trees / Forest", "color": "#397D49", "desc": "Dense tree canopy, forests"},
    2: {"name": "Grass", "color": "#88B053", "desc": "Natural grasslands, pastures"},
    3: {"name": "Flooded Vegetation", "color": "#7A87C6", "desc": "Wetlands, marshes, mangroves"},
    4: {"name": "Agriculture / Crops", "color": "#E49635", "desc": "Croplands, paddy fields"},
    5: {"name": "Shrub and Scrub", "color": "#DFC35A", "desc": "Sparse bushes, scrubland"},
    6: {"name": "Built Area", "color": "#C4281B", "desc": "Industrial zones, urban areas"},
    7: {"name": "Bare Ground", "color": "#A59B8F", "desc": "Exposed soil, rocks, quarries"},
    8: {"name": "Snow and Ice", "color": "#B39FE1", "desc": "Snow, glaciers, alpine ice"},
}

GEE_PROJECT = "rosy-phalanx-507408-c2"
_ee_initialized = False
_ee_auth_failed = False
_ee = None

def init_earth_engine():
    """Initialize Google Earth Engine with project credentials."""
    global _ee_initialized, _ee, _ee_auth_failed

    if _ee_initialized or _ee_auth_failed:
        return _ee_initialized

    try:
        import ee
        _ee = ee

        # Try to authenticate and initialize
        try:
            ee.Initialize(project=GEE_PROJECT)
            _ee_initialized = True
            print(f"Google Earth Engine initialized with project: {GEE_PROJECT}")
            return True
        except Exception:
            # Try with default credentials (do not call ee.Authenticate() as it blocks)
            try:
                # Without Authenticate(), this will just fail if no default credentials exist,
                # falling back gracefully to the heuristic estimator.
                ee.Initialize(project=GEE_PROJECT)
                _ee_initialized = True
                print(f"Google Earth Engine initialized after auth with project: {GEE_PROJECT}")
                return True
            except Exception as auth_err:
                _ee_auth_failed = True
                print(f"GEE auth failed: {auth_err}")
                print("Using heuristic land cover estimation instead.")
                return False
    except ImportError:
        _ee_auth_failed = True
        print("earthengine-api not installed. Using heuristic land cover estimation.")
        return False


def get_landcover_gee(lat: float, lng: float, radius_km: float = 2.0) -> Dict[str, Any]:
    """
    Get real Dynamic World land cover data from Google Earth Engine.
    Uses the 10m resolution Dynamic World dataset.
    """
    global _ee, _ee_initialized

    if not _ee_initialized:
        init_earth_engine()

    if _ee_initialized and _ee is not None:
        try:
            ee = _ee
            point = ee.Geometry.Point([lng, lat])
            buffer = point.buffer(radius_km * 1000)

            # Get Dynamic World composite (most recent)
            dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                .filterBounds(point) \
                .filterDate("2023-01-01", "2024-12-31") \
                .select("label") \
                .mode()

            # Sample the land cover at the point
            result = dw.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=buffer,
                scale=10,
                maxPixels=1e6
            ).getInfo()

            label = result.get("label", None)

            # Also get class probabilities
            dw_probs = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                .filterBounds(point) \
                .filterDate("2023-01-01", "2024-12-31") \
                .select(["water", "trees", "grass", "flooded_vegetation",
                         "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]) \
                .mean()

            probs_result = dw_probs.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=10,
                maxPixels=1e6
            ).getInfo()

            # Build distribution
            band_names = ["water", "trees", "grass", "flooded_vegetation",
                          "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]

            distribution = []
            for i, band in enumerate(band_names):
                pct = round((probs_result.get(band, 0.0) or 0.0) * 100.0, 1)
                distribution.append({
                    "classId": i,
                    "name": DW_CLASSES[i]["name"],
                    "color": DW_CLASSES[i]["color"],
                    "percentage": pct
                })

            distribution.sort(key=lambda x: x["percentage"], reverse=True)
            primary_id = distribution[0]["classId"] if distribution else (label if label is not None else 4)

            return {
                "source": "Google Earth Engine (Dynamic World 10m)",
                "project": GEE_PROJECT,
                "primaryClassId": primary_id,
                "primaryClassName": DW_CLASSES[primary_id]["name"],
                "primaryColor": DW_CLASSES[primary_id]["color"],
                "primaryDescription": DW_CLASSES[primary_id]["desc"],
                "distribution": distribution,
                "coordinates": {"lat": lat, "lng": lng},
                "radiusKm": radius_km,
            }

        except Exception as e:
            print(f"GEE query failed: {e}")
            traceback.print_exc()

    # Fallback: heuristic estimation
    return _estimate_landcover_heuristic(lat, lng)


def _estimate_landcover_heuristic(lat: float, lng: float) -> Dict[str, Any]:
    """Heuristic land cover estimation based on geographic location in India."""
    probs = [0.0] * 9

    # 1. Check for proximity to known industrial facilities first
    try:
        import math
        from model import FACILITIES_DB
        
        min_dist = float('inf')
        for f in FACILITIES_DB:
            dx = (f["lng"] - lng) * math.cos(math.radians(lat))
            dy = (f["lat"] - lat)
            dist = math.sqrt(dx*dx + dy*dy) * 111.0 # approx km
            if dist < min_dist:
                min_dist = dist
                
        if min_dist < 2.5:
            # Industrial site override
            probs[7] = 0.45  # Bare ground
            probs[6] = 0.35  # Built-up
            probs[5] = 0.10  # Shrub
            probs[1] = 0.05  # Trees
            probs[4] = 0.05  # Crops
            
    except ImportError:
        pass

    if sum(probs) == 0:
        # Indo-Gangetic Plain (agricultural heartland)
        if 24 <= lat <= 32 and 76 <= lng <= 88:
            probs[4] = 0.65  # Crops
            probs[6] = 0.15  # Built
            probs[2] = 0.08  # Grass
            probs[1] = 0.07  # Trees
            probs[0] = 0.05  # Water

        # Western Rajasthan (arid)
        elif lat > 24 and lng < 74:
            probs[7] = 0.55  # Bare Ground
            probs[5] = 0.30  # Shrub
            probs[4] = 0.05  # Crops
            probs[6] = 0.10  # Built
            
        # Northeast India (forest dense)
        elif lat > 22 and lng > 88:
            probs[1] = 0.65  # Trees
            probs[4] = 0.10  # Crops
        probs[2] = 0.10  # Grass
        probs[3] = 0.10  # Flooded Vegetation
        probs[0] = 0.05  # Water

    # Central Indian Forests (MP / Chhattisgarh / Odisha belt)
    elif 19 <= lat <= 24 and 79 <= lng <= 85:
        probs[1] = 0.55  # Trees
        probs[4] = 0.15  # Crops
        probs[5] = 0.15  # Shrub
        probs[2] = 0.10  # Grass
        probs[7] = 0.05  # Bare

    # Western Ghats
    elif 10 <= lat <= 20 and 73 <= lng <= 76.5:
        probs[1] = 0.60  # Trees
        probs[4] = 0.15  # Crops
        probs[2] = 0.15  # Grass
        probs[6] = 0.05  # Built
        probs[0] = 0.05  # Water

    # Deccan Plateau / Central India
    elif 15 <= lat <= 24 and 74 <= lng <= 79:
        probs[4] = 0.35  # Crops
        probs[5] = 0.30  # Shrub
        probs[7] = 0.15  # Bare
        probs[2] = 0.10  # Grass
        probs[1] = 0.10  # Trees

    # Himalayan (high altitude)
    elif lat > 32:
        probs[8] = 0.40  # Snow
        probs[7] = 0.30  # Bare
        probs[1] = 0.15  # Trees
        probs[2] = 0.10  # Grass
        probs[5] = 0.05  # Shrub

    # Coastal areas
    elif (lat < 11) or (lng > 84 and lat < 21):
        probs[0] = 0.30  # Water
        probs[4] = 0.25  # Crops
        probs[1] = 0.20  # Trees
        probs[3] = 0.15  # Flooded Veg
        probs[6] = 0.10  # Built

    # Eastern Ghats / South Interior (AP / TN / Karnataka forests and scrub)
    elif 11 <= lat <= 18 and 76.5 <= lng <= 81:
        probs[1] = 0.45  # Trees
        probs[5] = 0.30  # Shrub
        probs[4] = 0.10  # Crops
        probs[2] = 0.10  # Grass
        probs[7] = 0.05  # Bare

    # Default (generic India)
    else:
        probs[4] = 0.30  # Crops
        probs[1] = 0.30  # Trees
        probs[5] = 0.20  # Shrub
        probs[2] = 0.10  # Grass
        probs[6] = 0.10  # Built

    # Normalize and add deterministic variance based on coordinates
    import math
    import numpy as np
    
    # Generate pseudo-random deterministic variance (-0.15 to +0.15) for each band
    variance = [
        math.sin(lat * 12.3 + lng * 4.5 + i * 1.7) * 0.15 
        for i in range(9)
    ]
    
    # Apply variance and ensure no negative probabilities
    adjusted_probs = [max(0.01, p + v if p > 0 else 0) for i, (p, v) in enumerate(zip(probs, variance))]
    
    total_p = sum(adjusted_probs)
    normalized = [round((p / total_p) * 100.0, 1) for p in adjusted_probs]

    # Ensure it exactly sums to 100.0 (adjust the highest probability)
    primary_id = int(np.argmax(normalized))
    diff = round(100.0 - sum(normalized), 1)
    normalized[primary_id] = round(normalized[primary_id] + diff, 1)

    distribution = [
        {"classId": i, "name": DW_CLASSES[i]["name"], "color": DW_CLASSES[i]["color"], "percentage": normalized[i]}
        for i in range(9) if normalized[i] > 0
    ]
    distribution.sort(key=lambda x: x["percentage"], reverse=True)

    return {
        "source": "Heuristic Estimation (GEE auth required for real data)",
        "project": GEE_PROJECT,
        "primaryClassId": distribution[0]["classId"],
        "primaryClassName": distribution[0]["name"],
        "primaryColor": distribution[0]["color"],
        "primaryDescription": DW_CLASSES[distribution[0]["classId"]]["desc"],
        "distribution": distribution,
        "coordinates": {"lat": lat, "lng": lng},
        "radiusKm": 2.0,
    }
