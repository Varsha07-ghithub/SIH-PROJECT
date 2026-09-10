"""
FastAPI Backend Server for Industrial Fire Intelligence & Dynamic World Integration
Serves REST API at /api/* and mounts the frontend at /
Now loads ALL ~3.5M real FIRMS points from CSV and integrates Google Earth Engine.
"""

import os
import random
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .model import (
    ThermalRiskModel,
    LandCoverEngine,
    EventClassifier,
    generate_initial_events,
    find_nearest_facility,
    calculate_haversine_distance,
    FACILITIES_DB,
    DYNAMIC_WORLD_CLASSES
)

from .firms_loader import (
    load_firms_data,
    get_total_records,
    get_heatmap_data,
    get_sampled_points,
    get_risk_distribution,
    get_points_in_bounds,
    fetch_live_nasa_firms_nrt,
    get_available_dates,
)

from .landcover_gee import (
    init_earth_engine,
    get_landcover_gee,
    DW_CLASSES,
    GEE_PROJECT,
)

app = FastAPI(
    title="Geothermal AI & Dynamic World Fire Intelligence API",
    description="Backend API powering the Industrial Fire Intelligence Dashboard with ~3.5M real NASA FIRMS points & Google Earth Engine Dynamic World Land Cover",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for generated events and state
EVENTS_STORE: List[Dict[str, Any]] = generate_initial_events(120)

ALERTS_STORE: List[Dict[str, Any]] = [
    {
        "id": "ALT-1001",
        "title": "Critical Thermal Spike near Eastern Petro Refinery",
        "level": "CRITICAL",
        "color": "#ef4444",
        "facility": "Eastern Petro Refinery",
        "targetEvent": "TF-10293",
        "timestamp": "12 mins ago",
        "status": "ACTIVE",
        "threshold": "FRP > 180 MW"
    },
    {
        "id": "ALT-1002",
        "title": "Persistent Flaring Detected at South Steel Works",
        "level": "HIGH",
        "color": "#f97316",
        "facility": "South Steel Works",
        "targetEvent": "TF-10295",
        "timestamp": "45 mins ago",
        "status": "ACTIVE",
        "threshold": "Recurrence > 20 in 7d"
    },
    {
        "id": "ALT-1003",
        "title": "Agricultural Burning Boundary Alert",
        "level": "MODERATE",
        "color": "#facc15",
        "facility": "Northern Mining Zone",
        "targetEvent": "TF-10301",
        "timestamp": "2 hours ago",
        "status": "ACKNOWLEDGED",
        "threshold": "Proximity < 3 km"
    }
]

REPORTS_STORE: List[Dict[str, Any]] = [
    {
        "id": "REP-2026-001",
        "eventId": "TF-10293",
        "title": "Industrial Thermal Anomaly Investigation - Eastern Petro",
        "riskLevel": "CRITICAL",
        "riskScore": 91.4,
        "author": "Thermal AI Autonomous Inspector",
        "date": "2026-09-09",
        "status": "COMPLETED",
        "summary": "High-intensity thermal source detected on Built Area land cover within 0.8 km of Eastern Petro Refinery."
    },
]

# ==============================================================================
# STARTUP: Load real FIRMS data
# ==============================================================================

@app.on_event("startup")
def startup_event():
    print("\n" + "="*70)
    print("LOADING REAL FIRMS DATA (~3.5M points)...")
    print("="*70)
    load_firms_data()
    print("\nInitializing Google Earth Engine...")
    init_earth_engine()
    print("="*70)
    print("STARTUP COMPLETE!")
    print("="*70 + "\n")

# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class PredictRequest(BaseModel):
    lat: float
    lng: float
    frp: Optional[float] = 85.0
    brightness: Optional[float] = 340.0
    confidence: Optional[Any] = 85.0
    satellite: Optional[str] = "VIIRS"
    sensor: Optional[str] = "Suomi NPP"
    persistence_override: Optional[str] = None

class LandCoverRequest(BaseModel):
    lat: float
    lng: float
    radius_km: Optional[float] = 2.0

class AlertCreateRequest(BaseModel):
    title: str
    facility: str
    level: str
    threshold: str

class ReportCreateRequest(BaseModel):
    eventId: str
    notes: Optional[str] = ""

# ==============================================================================
# FIRMS MAP DATA ENDPOINTS (ALL ~3.5M POINTS)
# ==============================================================================

@app.get("/api/firms/heatmap")
def get_firms_heatmap():
    """Returns heatmap grid data for all ~3.5M FIRMS points."""
    data = get_heatmap_data()
    if not data:
        raise HTTPException(status_code=503, detail="FIRMS data not loaded yet")
    return data

@app.get("/api/firms/points")
def get_firms_points(
    limit: int = Query(5000, ge=100, le=50000),
    risk_level: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    south: Optional[float] = Query(None),
    north: Optional[float] = Query(None),
    west: Optional[float] = Query(None),
    east: Optional[float] = Query(None),
):
    """Returns sampled FIRMS points with risk data for map markers."""
    bounds = None
    if all(v is not None for v in [south, north, west, east]):
        bounds = {"south": south, "north": north, "west": west, "east": east}

    points = get_sampled_points(limit=limit, risk_filter=risk_level, date_filter=date, bounds=bounds)
    return {"total": get_total_records(), "returned": len(points), "points": points}

@app.get("/api/firms/dates")
def get_firms_dates():
    """Returns list of distinct observation dates available in the dataset."""
    dates = get_available_dates()
    return {"total": len(dates), "dates": dates}

@app.api_route("/api/firms/live-sync", methods=["GET", "POST"])
def sync_live_firms(map_key: Optional[str] = Query(None)):
    """Fetch and integrate live NRT satellite points from NASA FIRMS API."""
    live_points = fetch_live_nasa_firms_nrt(map_key=map_key)
    return {
        "status": "success",
        "livePointsFetched": len(live_points),
        "totalRecords": get_total_records(),
        "livePoints": live_points[:25]
    }

@app.get("/api/firms/bounds")
def get_firms_in_bounds(
    south: float = Query(...),
    north: float = Query(...),
    west: float = Query(...),
    east: float = Query(...),
    limit: int = Query(10000, ge=100, le=50000),
):
    """Returns FIRMS points within specific geographic bounds (for zoom-level detail)."""
    points = get_points_in_bounds(south, north, west, east, limit)
    return {"total": get_total_records(), "returned": len(points), "points": points}

@app.get("/api/firms/stats")
def get_firms_stats():
    """Returns risk distribution statistics across ALL FIRMS points."""
    dist = get_risk_distribution()
    dist["totalRecords"] = get_total_records()
    dist["dataSource"] = "NASA FIRMS (SV-C2, J2V-C2, J1V-C2)"
    return dist

# ==============================================================================
# GOOGLE EARTH ENGINE LAND COVER ENDPOINTS
# ==============================================================================

@app.post("/api/landcover")
def get_landcover(req: LandCoverRequest):
    """Get Dynamic World land cover data from Google Earth Engine for given coordinates."""
    result = get_landcover_gee(req.lat, req.lng, req.radius_km)
    return result

@app.get("/api/landcover/point")
def get_landcover_at_point(
    lat: float = Query(...),
    lng: float = Query(...),
    radius_km: float = Query(2.0),
):
    """GET endpoint for Dynamic World land cover at a point."""
    return get_landcover_gee(lat, lng, radius_km)

# ==============================================================================
# EXISTING REST API ENDPOINTS
# ==============================================================================

@app.get("/api/health")
def get_health():
    """Health check and model status."""
    return {
        "status": "ONLINE",
        "model": "Dynamic World & FIRMS Industrial Intelligence Model v2.0",
        "geeProject": GEE_PROJECT,
        "landcoverClassesCount": len(DW_CLASSES),
        "totalFirmsRecords": get_total_records(),
        "totalEventsLoaded": len(EVENTS_STORE),
        "totalFacilitiesMonitored": len(FACILITIES_DB),
        "dataFiles": [
            "fire_archive_SV_C2_794014.csv",
            "fire_archive_J2V_C2_794013.csv",
            "fire_archive_J1V_C2_794012.csv"
        ],
        "serverTimestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
def get_stats():
    """Returns top-level dashboard metrics and KPIs."""
    total = len(EVENTS_STORE)
    critical_count = sum(1 for e in EVENTS_STORE if e["riskLevel"] == "CRITICAL")
    high_count = sum(1 for e in EVENTS_STORE if e["riskLevel"] == "HIGH")
    moderate_count = sum(1 for e in EVENTS_STORE if e["riskLevel"] == "MODERATE")
    low_count = sum(1 for e in EVENTS_STORE if e["riskLevel"] == "LOW")

    avg_frp = round(sum(e["frp"] for e in EVENTS_STORE) / max(1, total), 1)
    avg_temp = round(sum(e["temperature"] for e in EVENTS_STORE) / max(1, total), 1)

    firms_dist = get_risk_distribution()

    return {
        "totalEvents": total,
        "totalFirmsRecords": get_total_records(),
        "critical": critical_count,
        "high": high_count,
        "moderate": moderate_count,
        "low": low_count,
        "averageFRP": avg_frp,
        "averageTemperature": avg_temp,
        "facilitiesMonitored": len(FACILITIES_DB),
        "activeAlerts": len([a for a in ALERTS_STORE if a["status"] == "ACTIVE"]),
        "firmsDistribution": firms_dist,
    }

@app.get("/api/events")
def get_events(
    q: Optional[str] = Query(None),
    classification: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    facility: Optional[str] = Query(None),
    limit: Optional[int] = Query(120, ge=1, le=500),
    offset: Optional[int] = Query(0, ge=0)
):
    """Returns paginated and filtered FIRMS events."""
    results = EVENTS_STORE
    if q:
        query = q.lower().strip()
        results = [e for e in results if query in e["id"].lower() or query in e["classification"].lower() or query in e["facility"].lower()]
    if classification:
        results = [e for e in results if e["classification"].lower() == classification.lower()]
    if risk_level:
        results = [e for e in results if e["riskLevel"].upper() == risk_level.upper()]
    if facility:
        results = [e for e in results if facility.lower() in e["facility"].lower()]

    paginated = results[offset : offset + limit]
    return {"total": len(results), "offset": offset, "limit": limit, "events": paginated}

@app.get("/api/events/{event_id}")
def get_event_detail(event_id: str):
    for e in EVENTS_STORE:
        if e["id"].lower() == event_id.lower():
            return e
    raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

@app.post("/api/predict")
def predict_fire_risk(req: PredictRequest):
    """Run real-time inference using the Dynamic World & Thermal Risk Model."""
    geo_context = find_nearest_facility(req.lat, req.lng)
    
    # Get dynamic GEE land cover (or heuristic fallback)
    gee_lc = get_landcover_gee(req.lat, req.lng)
    
    risk_data = ThermalRiskModel.compute_risk(req.frp, req.brightness, req.confidence)

    # Persistence / Recurrence calculation
    if req.persistence_override == "HIGH":
        simulated_recurrence = random.randint(26, 42)
    elif req.persistence_override == "MEDIUM":
        simulated_recurrence = random.randint(12, 24)
    elif req.persistence_override == "LOW":
        simulated_recurrence = random.randint(3, 8)
    else:
        # Dynamic calculation based on FIRMS spatial density + thermal characteristics + facility distance
        nearby_count = 0
        from backend.firms_loader import _sampled_all
        if _sampled_all:
            nearby_count = sum(1 for p in _sampled_all if abs(p.get('lat', 0) - req.lat) < 0.3 and abs(p.get('lng', 0) - req.lng) < 0.3)
        
        base_rec = min(20, nearby_count * 2)
        frp_val = req.frp or 50.0
        frp_factor = int(frp_val / 6.0)
        
        conf_val = 80.0
        try:
            if req.confidence is not None:
                conf_val = float(req.confidence)
        except Exception:
            pass
        conf_factor = int(conf_val / 20.0)
        
        dist = geo_context["distanceKm"]
        fac_factor = 20 if dist <= 1.5 else (12 if dist <= 5.0 else (5 if dist <= 25.0 else 0))
        
        simulated_recurrence = max(3, min(65, base_rec + frp_factor + conf_factor + fac_factor))

    ai_res = EventClassifier.classify(
        frp=req.frp, brightness=req.brightness,
        confidence=float(req.confidence if isinstance(req.confidence, (int, float)) else 80),
        dist_km=geo_context["distanceKm"], landcover_id=gee_lc["primaryClassId"],
        detections_count=simulated_recurrence
    )

    new_id = f"TF-CUSTOM-{len(EVENTS_STORE) + 1}"
    prediction_result = {
        "id": new_id, "lat": req.lat, "lng": req.lng,
        "classification": ai_res["classification"], "confidence": ai_res["confidence"],
        "risk": risk_data["riskScore"], "riskLevel": risk_data["riskLevel"],
        "riskColor": risk_data["riskColor"], "riskBadge": risk_data["riskBadge"],
        "frp": req.frp, "temperature": req.brightness,
        "satellite": req.satellite, "sensor": req.sensor,
        "facility": geo_context["facility"], "facilityType": geo_context["facilityType"],
        "distance": geo_context["distanceStr"], "facilityDistanceKm": geo_context["distanceKm"],
        "landCover": gee_lc["primaryClassName"], "landCoverId": gee_lc["primaryClassId"],
        "landCoverColor": gee_lc["primaryColor"], "landCoverSource": gee_lc["source"],
        "landCoverDistribution": gee_lc["distribution"],
        "persistence": "HIGH" if simulated_recurrence >= 25 else ("MEDIUM" if simulated_recurrence >= 10 else "LOW"),
        "detections24h": max(1, int(simulated_recurrence * 0.15)),
        "detections7d": max(2, int(simulated_recurrence * 0.5)),
        "detections30d": simulated_recurrence,
        "probabilities": ai_res["probabilities"],
        "explanation": ai_res["explanation"],
        "factors": ai_res["factors"],
        "riskBreakdownBars": ai_res["riskBreakdownBars"],
        "timestamp": "Just now"
    }
    EVENTS_STORE.insert(0, prediction_result)
    return prediction_result

@app.post("/api/analyze-landcover")
def analyze_landcover(req: LandCoverRequest):
    """Detailed Dynamic World land cover from GEE around custom coordinates."""
    geo_context = find_nearest_facility(req.lat, req.lng)
    gee_lc = get_landcover_gee(req.lat, req.lng, req.radius_km)

    return {
        "coordinates": {"lat": req.lat, "lng": req.lng},
        "radiusKm": req.radius_km,
        "nearestFacility": geo_context["facility"],
        "distanceToFacility": geo_context["distanceStr"],
        "source": gee_lc["source"],
        "dominantClass": gee_lc["primaryClassName"],
        "dominantClassId": gee_lc["primaryClassId"],
        "dominantColor": gee_lc["primaryColor"],
        "distribution": gee_lc["distribution"],
    }

@app.get("/api/facilities")
def get_facilities():
    results = []
    for f in FACILITIES_DB:
        events_near = [e for e in EVENTS_STORE if calculate_haversine_distance(e["lat"], e["lng"], f["lat"], f["lng"]) <= 5.0]
        avg_risk = round(sum(e["risk"] for e in events_near) / max(1, len(events_near)), 1) if events_near else f["risk"]
        results.append({
            **f, "events": len(events_near) if events_near else f["events"],
            "currentRisk": avg_risk,
            "status": "CRITICAL RISK" if avg_risk >= 80 else ("HIGH RISK" if avg_risk >= 60 else "MONITORED")
        })
    return results

@app.get("/api/persistence")
def get_persistence():
    persistent_events = [e for e in EVENTS_STORE if e["detections30d"] >= 20]
    persistent_events.sort(key=lambda x: x["detections30d"], reverse=True)
    return {
        "totalPersistent": len(persistent_events),
        "averageDetections": round(sum(e["detections30d"] for e in persistent_events) / max(1, len(persistent_events)), 1),
        "highPersistenceSources": persistent_events[:25]
    }

@app.get("/api/risk-summary")
def get_risk_summary():
    total = len(EVENTS_STORE)
    critical = [e for e in EVENTS_STORE if e["riskLevel"] == "CRITICAL"]
    high = [e for e in EVENTS_STORE if e["riskLevel"] == "HIGH"]
    moderate = [e for e in EVENTS_STORE if e["riskLevel"] == "MODERATE"]
    low = [e for e in EVENTS_STORE if e["riskLevel"] == "LOW"]
    return {
        "total": total, "critical": len(critical), "high": len(high),
        "moderate": len(moderate), "low": len(low),
        "topCriticalEvents": critical[:10]
    }

@app.get("/api/alerts")
def get_alerts():
    return ALERTS_STORE

@app.post("/api/alerts")
def create_alert(req: AlertCreateRequest):
    new_alert = {
        "id": f"ALT-{1000 + len(ALERTS_STORE) + 1}", "title": req.title,
        "facility": req.facility, "level": req.level.upper(),
        "color": "#ef4444" if req.level.upper() == "CRITICAL" else ("#f97316" if req.level.upper() == "HIGH" else "#facc15"),
        "targetEvent": "TF-CUSTOM", "timestamp": "Just now", "status": "ACTIVE", "threshold": req.threshold
    }
    ALERTS_STORE.insert(0, new_alert)
    return new_alert

@app.get("/api/reports")
def get_reports():
    return REPORTS_STORE

@app.post("/api/reports")
def create_report(req: ReportCreateRequest):
    target = next((e for e in EVENTS_STORE if e["id"].lower() == req.eventId.lower()), EVENTS_STORE[0])
    new_report = {
        "id": f"REP-2026-{str(len(REPORTS_STORE) + 1).zfill(3)}",
        "eventId": target["id"],
        "title": f"Investigation Report - {target['facility']} ({target['classification']})",
        "riskLevel": target["riskLevel"], "riskScore": target["risk"],
        "author": "Thermal AI Autonomous Inspector", "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "COMPLETED",
        "summary": f"{target['classification']} at {target['lat']}, {target['lng']}. FRP: {target['frp']} MW. {req.notes or target['explanation']}"
    }
    REPORTS_STORE.insert(0, new_report)
    return new_report

# ==============================================================================
# FRONTEND STATIC FILES SERVING
# ==============================================================================

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "front end"))

if os.path.exists(FRONTEND_DIR):
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
