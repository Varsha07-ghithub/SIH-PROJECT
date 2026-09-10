"""
Industrial Fire Intelligence & Dynamic World Land Cover Model
Extracted and enhanced from landcover integration.ipynb
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

# ==============================================================================
# 1. DYNAMIC WORLD LAND COVER DEFINITIONS & PALETTES
# ==============================================================================

DYNAMIC_WORLD_CLASSES = {
    0: {
        "id": 0,
        "name": "Water",
        "color": "#419BDF",
        "description": "Rivers, lakes, coastal waters, reservoirs"
    },
    1: {
        "id": 1,
        "name": "Trees / Forest",
        "color": "#397D49",
        "description": "Dense tree canopy, primary and secondary forests"
    },
    2: {
        "id": 2,
        "name": "Grass",
        "color": "#88B053",
        "description": "Natural grasslands, pastures, open vegetation"
    },
    3: {
        "id": 3,
        "name": "Flooded Vegetation",
        "color": "#7A87C6",
        "description": "Wetlands, marshes, mangroves, flooded fields"
    },
    4: {
        "id": 4,
        "name": "Agriculture / Crops",
        "color": "#E49635",
        "description": "Croplands, paddy fields, seasonal cultivation"
    },
    5: {
        "id": 5,
        "name": "Shrub and Scrub",
        "color": "#DFC35A",
        "description": "Sparse bushes, scrubland, arid shrub communities"
    },
    6: {
        "id": 6,
        "name": "Built Area",
        "color": "#C4281B",
        "description": "Industrial zones, urban structures, manufacturing hubs"
    },
    7: {
        "id": 7,
        "name": "Bare Ground",
        "color": "#A59B8F",
        "description": "Exposed soil, sand, rocks, quarries, mining sites"
    },
    8: {
        "id": 8,
        "name": "Snow and Ice",
        "color": "#B39FE1",
        "description": "Perennial snow, glaciated terrain, alpine ice"
    }
}

CONFIDENCE_MAP = {
    "l": 30.0,
    "low": 30.0,
    "n": 60.0,
    "nominal": 60.0,
    "h": 90.0,
    "high": 90.0
}

# ==============================================================================
# 2. INDUSTRIAL FACILITIES DATABASE
# ==============================================================================

FACILITIES_DB = [
    {
        "id": "FAC-001",
        "name": "Eastern Petro Refinery (Manali)",
        "type": "Oil Refinery",
        "lat": 13.1680,
        "lng": 80.2580,
        "risk": 91,
        "events": 34,
        "persistence": "HIGH",
        "description": "Major petrochemical refining hub with heavy flaring infrastructure."
    },
    {
        "id": "FAC-002",
        "name": "Coastal Thermal Power (Ennore)",
        "type": "Thermal Power Plant",
        "lat": 13.2200,
        "lng": 80.3200,
        "risk": 78,
        "events": 22,
        "persistence": "MEDIUM",
        "description": "Coal-fired thermal generation plant with active boiler stacks."
    },
    {
        "id": "FAC-003",
        "name": "South Steel Works (Sriperumbudur)",
        "type": "Steel Plant",
        "lat": 12.9700,
        "lng": 79.9500,
        "risk": 83,
        "events": 28,
        "persistence": "HIGH",
        "description": "Integrated blast furnace and continuous smelting plant."
    },
    {
        "id": "FAC-004",
        "name": "Industrial LNG Hub (Kattupalli)",
        "type": "LNG Facility",
        "lat": 13.3100,
        "lng": 80.3300,
        "risk": 88,
        "events": 19,
        "persistence": "HIGH",
        "description": "Liquefied natural gas terminal and cryogenic storage complex."
    },
    {
        "id": "FAC-005",
        "name": "Neyveli Lignite Thermal Power",
        "type": "Thermal Power Plant",
        "lat": 11.6000,
        "lng": 79.4800,
        "risk": 82,
        "events": 25,
        "persistence": "HIGH",
        "description": "Lignite-fired thermal power complex and open-cast mining."
    },
    {
        "id": "FAC-006",
        "name": "Reliance Jamnagar Refinery",
        "type": "Oil Refinery",
        "lat": 22.3550,
        "lng": 69.8640,
        "risk": 96,
        "events": 68,
        "persistence": "HIGH",
        "description": "World's largest petroleum refinery and petrochemical complex."
    },
    {
        "id": "FAC-007",
        "name": "Gujarat Petrochemical Complex (Vadodara)",
        "type": "Petrochemical Plant",
        "lat": 22.3072,
        "lng": 73.1812,
        "risk": 89,
        "events": 41,
        "persistence": "HIGH",
        "description": "High-capacity polymer synthesis and cracking refinery."
    },
    {
        "id": "FAC-008",
        "name": "Mundra Ultra Mega Power Plant",
        "type": "Thermal Power Plant",
        "lat": 22.8200,
        "lng": 69.5200,
        "risk": 90,
        "events": 45,
        "persistence": "HIGH",
        "description": "4,000 MW coastal super-critical thermal power plant."
    },
    {
        "id": "FAC-009",
        "name": "Hazira Steel & Petrochemical Hub",
        "type": "Steel & Petrochem",
        "lat": 21.1100,
        "lng": 72.6300,
        "risk": 92,
        "events": 52,
        "persistence": "HIGH",
        "description": "Integrated steel plant, LNG terminal, and heavy manufacturing."
    },
    {
        "id": "FAC-010",
        "name": "Tata Steel Plant (Jamshedpur)",
        "type": "Steel Plant",
        "lat": 22.7925,
        "lng": 86.2029,
        "risk": 94,
        "events": 61,
        "persistence": "HIGH",
        "description": "Historic integrated steel manufacturing facility and blast furnaces."
    },
    {
        "id": "FAC-011",
        "name": "Bokaro Steel Plant (SAIL)",
        "type": "Steel Plant",
        "lat": 23.6693,
        "lng": 86.1511,
        "risk": 87,
        "events": 39,
        "persistence": "HIGH",
        "description": "Integrated steel works producing flat steel products."
    },
    {
        "id": "FAC-012",
        "name": "Bhilai Steel Authority (SAIL)",
        "type": "Steel Plant",
        "lat": 21.1938,
        "lng": 81.3509,
        "risk": 86,
        "events": 37,
        "persistence": "HIGH",
        "description": "Heavy industrial steel works and rail manufacturing plant."
    },
    {
        "id": "FAC-013",
        "name": "Korba Super Thermal Power",
        "type": "Thermal Power Plant",
        "lat": 22.3595,
        "lng": 82.6841,
        "risk": 75,
        "events": 29,
        "persistence": "MEDIUM",
        "description": "2,600 MW coal thermal station with extensive cooling towers."
    },
    {
        "id": "FAC-014",
        "name": "Raigarh Jindal Steel & Power",
        "type": "Steel & Power",
        "lat": 21.8974,
        "lng": 83.3950,
        "risk": 88,
        "events": 43,
        "persistence": "HIGH",
        "description": "Coal-to-steel manufacturing complex and captive power."
    },
    {
        "id": "FAC-015",
        "name": "Rourkela Steel Plant (SAIL)",
        "type": "Steel Plant",
        "lat": 22.2530,
        "lng": 84.8596,
        "risk": 89,
        "events": 44,
        "persistence": "HIGH",
        "description": "Integrated steel plant with heavy blast furnace infrastructure."
    },
    {
        "id": "FAC-016",
        "name": "Angul Jindal Steel & Power",
        "type": "Steel & Power",
        "lat": 20.8400,
        "lng": 85.1500,
        "risk": 91,
        "events": 48,
        "persistence": "HIGH",
        "description": "6 MTPA steel plant and coal gasification complex."
    },
    {
        "id": "FAC-017",
        "name": "Paradip Oil Refinery (IOCL)",
        "type": "Oil Refinery",
        "lat": 20.2700,
        "lng": 86.6700,
        "risk": 93,
        "events": 55,
        "persistence": "HIGH",
        "description": "15 MMTPA modern coastal petroleum refinery."
    },
    {
        "id": "FAC-018",
        "name": "Visakhapatnam Steel Plant (RINL)",
        "type": "Steel Plant",
        "lat": 17.6300,
        "lng": 83.1800,
        "risk": 88,
        "events": 38,
        "persistence": "HIGH",
        "description": "Shore-based integrated steel plant."
    },
    {
        "id": "FAC-019",
        "name": "Visakhapatnam HPCL Refinery",
        "type": "Oil Refinery",
        "lat": 17.6900,
        "lng": 83.2500,
        "risk": 90,
        "events": 42,
        "persistence": "HIGH",
        "description": "Major East Coast petroleum refinery and storage terminal."
    },
    {
        "id": "FAC-020",
        "name": "Simhadri Thermal Power (NTPC)",
        "type": "Thermal Power Plant",
        "lat": 17.6000,
        "lng": 83.0800,
        "risk": 79,
        "events": 23,
        "persistence": "MEDIUM",
        "description": "2,000 MW coal-fired thermal power generation station."
    },
    {
        "id": "FAC-021",
        "name": "Singrauli Thermal Power (NTPC)",
        "type": "Thermal Power Plant",
        "lat": 24.1000,
        "lng": 82.6700,
        "risk": 91,
        "events": 50,
        "persistence": "HIGH",
        "description": "2,000 MW pithead thermal power station."
    },
    {
        "id": "FAC-022",
        "name": "Vindhyachal Super Thermal Power",
        "type": "Thermal Power Plant",
        "lat": 24.1100,
        "lng": 82.6100,
        "risk": 95,
        "events": 64,
        "persistence": "HIGH",
        "description": "4,760 MW India's largest thermal power plant complex."
    },
    {
        "id": "FAC-023",
        "name": "Durgapur Steel Plant (SAIL)",
        "type": "Steel Plant",
        "lat": 23.5500,
        "lng": 87.2800,
        "risk": 84,
        "events": 31,
        "persistence": "HIGH",
        "description": "Integrated steel plant manufacturing wheel and axle products."
    },
    {
        "id": "FAC-024",
        "name": "Haldia Petrochemicals & Refinery",
        "type": "Petrochemical Plant",
        "lat": 22.0600,
        "lng": 88.0800,
        "risk": 87,
        "events": 36,
        "persistence": "HIGH",
        "description": "Naphtha cracker and downstream polymer processing hub."
    },
    {
        "id": "FAC-025",
        "name": "JSW Steel Vijayanagar (Bellary)",
        "type": "Steel Plant",
        "lat": 15.1800,
        "lng": 76.6500,
        "risk": 93,
        "events": 57,
        "persistence": "HIGH",
        "description": "12 MTPA single-location integrated steel plant."
    },
    {
        "id": "FAC-026",
        "name": "Mangalore Refinery (MRPL)",
        "type": "Oil Refinery",
        "lat": 12.9800,
        "lng": 74.8300,
        "risk": 89,
        "events": 39,
        "persistence": "HIGH",
        "description": "15 MMTPA coastal crude oil refinery."
    },
    {
        "id": "FAC-027",
        "name": "BPCL Kochi Refinery",
        "type": "Oil Refinery",
        "lat": 9.9800,
        "lng": 76.3600,
        "risk": 88,
        "events": 33,
        "persistence": "HIGH",
        "description": "15.5 MMTPA integrated petroleum refinery and propylene complex."
    },
    {
        "id": "FAC-028",
        "name": "Panipat Refinery & Petrochem (IOCL)",
        "type": "Oil Refinery",
        "lat": 29.4700,
        "lng": 76.8800,
        "risk": 94,
        "events": 62,
        "persistence": "HIGH",
        "description": "15 MMTPA mega refinery and aromatic petrochemical plant."
    },
    {
        "id": "FAC-029",
        "name": "Bathinda Refinery (HMEL)",
        "type": "Oil Refinery",
        "lat": 30.0300,
        "lng": 74.9300,
        "risk": 90,
        "events": 41,
        "persistence": "HIGH",
        "description": "11.3 MMTPA public-private joint venture petroleum refinery."
    },
    {
        "id": "FAC-030",
        "name": "Ramagundam Super Thermal (NTPC)",
        "type": "Thermal Power Plant",
        "lat": 18.7600,
        "lng": 79.4500,
        "risk": 86,
        "events": 35,
        "persistence": "HIGH",
        "description": "2,600 MW thermal station and floating solar power plant."
    }
]

# ==============================================================================
# 3. HAVERSINE DISTANCE HELPER
# ==============================================================================

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance between two coordinates in kilometers."""
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def find_nearest_facility(lat: float, lon: float) -> Dict[str, Any]:
    """Find the nearest industrial facility and list all facilities within 100 km."""
    best_facility = FACILITIES_DB[0]
    min_dist = float("inf")
    nearby_list = []

    for f in FACILITIES_DB:
        d = calculate_haversine_distance(lat, lon, f["lat"], f["lng"])
        if d < min_dist:
            min_dist = d
            best_facility = f
        if d <= 100.0:
            nearby_list.append({
                "name": f["name"],
                "type": f["type"],
                "distanceKm": round(d, 1),
                "risk": f.get("risk", 75)
            })

    # Sort nearby by distance
    nearby_list.sort(key=lambda x: x["distanceKm"])

    return {
        "facility": best_facility["name"],
        "facilityId": best_facility["id"],
        "facilityType": best_facility["type"],
        "facilityLat": best_facility["lat"],
        "facilityLng": best_facility["lng"],
        "distanceKm": round(min_dist, 2),
        "distanceStr": f"{min_dist:.1f} km",
        "nearbyFacilities": nearby_list
    }

# ==============================================================================
# 4. THERMAL RISK SCORING MODEL (FROM NOTEBOOK)
# ==============================================================================

class ThermalRiskModel:
    """
    Implements the risk assessment scoring algorithm established in
    the 'landcover integration.ipynb' notebook:
      Risk_Score = 0.50 * FRP_Score + 0.30 * Brightness_Score + 0.20 * Confidence_Score
    """

    # Reference bounds for normalization/percentiles based on NASA FIRMS SV-C2 archive
    FRP_MIN = 5.0
    FRP_MAX = 350.0

    BRIGHTNESS_MIN = 295.0
    BRIGHTNESS_MAX = 480.0

    @classmethod
    def calculate_percentile_frp(cls, frp: float) -> float:
        val = max(cls.FRP_MIN, min(float(frp), cls.FRP_MAX))
        # Log-transformed percentile distribution matching FIRMS
        pct = (math.log1p(val - cls.FRP_MIN) / math.log1p(cls.FRP_MAX - cls.FRP_MIN)) * 100.0
        return round(max(0.0, min(pct, 100.0)), 1)

    @classmethod
    def calculate_percentile_brightness(cls, brightness: float) -> float:
        val = max(cls.BRIGHTNESS_MIN, min(float(brightness), cls.BRIGHTNESS_MAX))
        pct = ((val - cls.BRIGHTNESS_MIN) / (cls.BRIGHTNESS_MAX - cls.BRIGHTNESS_MIN)) * 100.0
        return round(max(0.0, min(pct, 100.0)), 1)

    @classmethod
    def parse_confidence(cls, confidence: Any) -> float:
        if isinstance(confidence, (int, float)):
            return float(confidence)
        s = str(confidence).strip().lower()
        if s in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[s]
        try:
            return float(s)
        except ValueError:
            return 60.0

    @classmethod
    def compute_risk(cls, frp: float, brightness: float, confidence: Any) -> Dict[str, Any]:
        frp_pct = cls.calculate_percentile_frp(frp)
        bright_pct = cls.calculate_percentile_brightness(brightness)
        conf_val = cls.parse_confidence(confidence)
        conf_pct = max(0.0, min(conf_val, 100.0))

        # Composite score matching notebook: 0.50 * FRP + 0.30 * Brightness + 0.20 * Confidence
        composite_score = (0.50 * frp_pct) + (0.30 * bright_pct) + (0.20 * conf_pct)
        composite_score = round(max(0.0, min(composite_score, 100.0)), 1)

        # Risk level classification thresholds from notebook:
        # >=80: CRITICAL, >=60: HIGH, >=35: MODERATE, <35: LOW
        if composite_score >= 80.0:
            level = "CRITICAL"
            color = "#ef4444"
            badge = "CRITICAL RISK"
        elif composite_score >= 60.0:
            level = "HIGH"
            color = "#f97316"
            badge = "HIGH RISK"
        elif composite_score >= 35.0:
            level = "MODERATE"
            color = "#facc15"
            badge = "MODERATE RISK"
        else:
            level = "LOW"
            color = "#22c55e"
            badge = "LOW RISK"

        return {
            "riskScore": composite_score,
            "riskLevel": level,
            "riskColor": color,
            "riskBadge": badge,
            "breakdown": {
                "frpScore": frp_pct,
                "brightnessScore": bright_pct,
                "confidenceScore": conf_pct,
                "weights": {
                    "frpWeight": 0.50,
                    "brightnessWeight": 0.30,
                    "confidenceWeight": 0.20
                }
            }
        }

# ==============================================================================
# 5. DYNAMIC WORLD LAND COVER INFERENCE ENGINE
# ==============================================================================

class LandCoverEngine:
    """
    Predicts Dynamic World land cover class and multiclass probabilities
    for geographic coordinates and satellite observations.
    """

    @classmethod
    def estimate_landcover(cls, lat: float, lon: float, facility_dist_km: float) -> Dict[str, Any]:
        """
        Calculates the Dynamic World land-cover profile.
        If near an industrial facility (< 3 km), class 6 (Built Area) dominates.
        Otherwise evaluates geographic profile (cropland, forest, water, etc.).
        """
        # Dynamic World class probabilities array [0..8]
        probs = [0.0] * 9

        if facility_dist_km <= 1.5:
            # Industrial Built Area dominant
            probs[6] = 0.72  # Built Area
            probs[7] = 0.12  # Bare Ground
            probs[4] = 0.08  # Crops
            probs[2] = 0.05  # Grass
            probs[1] = 0.03  # Trees
        elif facility_dist_km <= 4.0:
            probs[6] = 0.45  # Built Area
            probs[4] = 0.25  # Crops
            probs[7] = 0.15  # Bare Ground
            probs[2] = 0.10  # Grass
            probs[1] = 0.05  # Trees
        else:
            # Regional India geographic heuristics
            # Coastal / estuarine areas
            if (lat < 14.0 and lon > 80.0) or (lat < 23.0 and lon > 88.0):
                probs[4] = 0.45  # Agriculture/Crops
                probs[2] = 0.20  # Grass
                probs[1] = 0.15  # Trees
                probs[6] = 0.10  # Built
                probs[0] = 0.10  # Water
            elif lat > 28.0 and lon < 76.0:  # Arid / northwest
                probs[7] = 0.40  # Bare Ground
                probs[5] = 0.30  # Shrub
                probs[4] = 0.20  # Agriculture
                probs[6] = 0.10  # Built
            else:
                probs[4] = 0.50  # Agriculture
                probs[1] = 0.25  # Trees / Forest
                probs[2] = 0.15  # Grass
                probs[6] = 0.05  # Built
                probs[5] = 0.05  # Shrub

        # Normalize probabilities to sum to 100%
        total_p = sum(probs)
        normalized = [round((p / total_p) * 100.0, 1) for p in probs]
        primary_class_id = int(np.argmax(normalized))
        primary_meta = DYNAMIC_WORLD_CLASSES[primary_class_id]

        distribution = [
            {
                "classId": cid,
                "name": DYNAMIC_WORLD_CLASSES[cid]["name"],
                "color": DYNAMIC_WORLD_CLASSES[cid]["color"],
                "percentage": normalized[cid]
            }
            for cid in range(9)
        ]
        # Sort by percentage descending
        distribution.sort(key=lambda x: x["percentage"], reverse=True)

        return {
            "primaryClassId": primary_class_id,
            "primaryClassName": primary_meta["name"],
            "primaryColor": primary_meta["color"],
            "primaryDescription": primary_meta["description"],
            "distribution": distribution
        }

# ==============================================================================
# 6. MULTI-CLASS AI CLASSIFICATION & EXPLAINABILITY ENGINE
# ==============================================================================

class EventClassifier:
    """
    Computes AI classification, probability breakdown, and explainability factors.
    Classes: Industrial Fire, Persistent Thermal Source, Natural Fire, Agricultural Burning, Unknown
    """

    @classmethod
    def classify(cls, frp: float, brightness: float, confidence: float,
                 dist_km: float, landcover_id: int, detections_count: int) -> Dict[str, Any]:

        # Base scores for each classification
        ind_score = 10.0
        pers_score = 10.0
        nat_score = 10.0
        agri_score = 10.0
        unk_score = 5.0

        # Proximity to industrial facility
        if dist_km <= 1.0:
            ind_score += 55.0
            pers_score += 25.0
        elif dist_km <= 3.0:
            ind_score += 35.0
            pers_score += 20.0
        elif dist_km <= 6.0:
            ind_score += 15.0
            agri_score += 20.0

        # Land cover effect (Class 6: Built Area, Class 4: Crops, Class 1: Trees)
        if landcover_id == 6:  # Built
            ind_score += 30.0
            pers_score += 20.0
        elif landcover_id == 4:  # Crops
            agri_score += 50.0
            nat_score += 10.0
        elif landcover_id in [1, 2, 5]:  # Trees, Grass, Shrub
            nat_score += 45.0
            agri_score += 15.0
        elif landcover_id == 0:  # Water / offshore flaring
            pers_score += 40.0
            ind_score += 30.0

        # Thermal intensity (FRP & Brightness)
        if frp > 120.0 or brightness > 370.0:
            ind_score += 25.0
            pers_score += 15.0
        elif frp > 50.0:
            ind_score += 10.0
            nat_score += 15.0
            agri_score += 10.0
        else:
            agri_score += 20.0
            unk_score += 10.0

        # Persistence detection history
        if detections_count >= 20:
            pers_score += 50.0
            ind_score += 20.0
        elif detections_count >= 8:
            pers_score += 25.0
            ind_score += 10.0

        raw_scores = [
            ("Industrial Fire", ind_score),
            ("Persistent Thermal Source", pers_score),
            ("Natural Fire", nat_score),
            ("Agricultural Burning", agri_score),
            ("Unknown", unk_score)
        ]

        # Softmax normalization for probabilities
        exp_vals = [math.exp(s / 20.0) for _, s in raw_scores]
        sum_exp = sum(exp_vals)
        probs = [(name, round((e / sum_exp) * 100.0, 1)) for (name, _), e in zip(raw_scores, exp_vals)]
        probs.sort(key=lambda x: x[1], reverse=True)

        primary_class = probs[0][0]
        confidence_val = probs[0][1]

        # Explainability analysis
        reasons = []
        if dist_km <= 2.5:
            reasons.append(f"Located within {dist_km:.1f} km of known industrial infrastructure")
        if landcover_id == 6:
            reasons.append("Dynamic World 10m land cover identified as Built / Industrial Area")
        elif landcover_id == 4:
            reasons.append("Dynamic World 10m land cover identified as Agricultural Cropland")
        elif landcover_id in [1, 2, 5]:
            reasons.append("Dynamic World 10m land cover identified as Natural Forest / Vegetation canopy")

        if frp > 100.0:
            reasons.append(f"Very high thermal output ({frp:.1f} MW FRP) characteristic of industrial combustion")
        if detections_count >= 10:
            reasons.append(f"High temporal recurrence ({detections_count} detections across 30 days)")

        if not reasons:
            reasons.append("Thermal signature matches standard baseline patterns")

        explanation_text = ". ".join(reasons) + "."

        factors = {
            "thermalIntensity": "VERY HIGH" if frp > 150 else ("HIGH" if frp > 70 else "MODERATE"),
            "persistence": "HIGH" if detections_count >= 20 else ("MEDIUM" if detections_count >= 8 else "LOW"),
            "industrialProximity": "VERY HIGH" if dist_km <= 1.0 else ("HIGH" if dist_km <= 3.0 else ("MEDIUM" if dist_km <= 7.0 else "LOW")),
            "confidence": "HIGH" if confidence >= 80 else ("MEDIUM" if confidence >= 50 else "LOW"),
            "landCover": DYNAMIC_WORLD_CLASSES[landcover_id]["name"].upper()
        }

        risk_breakdown_bars = [
            {"name": "Thermal Intensity", "value": f"{min(30, int(frp / 6))} / 30", "percent": min(100, int((frp / 180) * 100))},
            {"name": "Persistence", "value": f"{min(25, int(detections_count * 0.7))} / 25", "percent": min(100, int((detections_count / 30) * 100))},
            {"name": "Industrial Proximity", "value": f"{max(0, min(20, int((10 - dist_km) * 2)))} / 20", "percent": max(0, min(100, int(((10 - dist_km) / 10) * 100)))},
            {"name": "Detection Confidence", "value": f"{int((confidence / 100) * 15)} / 15", "percent": int(confidence)},
            {"name": "Environmental Context", "value": "8 / 10" if landcover_id == 6 else "5 / 10", "percent": 80 if landcover_id == 6 else 50}
        ]

        return {
            "classification": primary_class,
            "confidence": confidence_val,
            "probabilities": probs,
            "explanation": explanation_text,
            "factors": factors,
            "riskBreakdownBars": risk_breakdown_bars
        }

# ==============================================================================
# 7. SEED DATASET GENERATOR (MATCHING FRONTEND + NOTEBOOK DATA)
# ==============================================================================

def generate_initial_events(count: int = 120) -> List[Dict[str, Any]]:
    """Generates the initial rich event dataset with complete ML annotations."""
    events = []
    base_date = datetime.now()

    for i in range(count):
        facility = FACILITIES_DB[i % len(FACILITIES_DB)]

        # Offsets around the facility
        lat_offset = ((i % 10) - 5) * 0.012
        lng_offset = ((i % 11) - 5) * 0.012

        lat = round(facility["lat"] + lat_offset, 5)
        lng = round(facility["lng"] + lng_offset, 5)

        frp = float(50 + (i * 7) % 240)
        temperature = float(300 + (i * 3) % 90)
        confidence = float(70 + (i % 28))
        satellite = "VIIRS" if (i % 2 == 0) else "MODIS"
        sensor = "Suomi NPP" if (i % 2 == 0) else "Terra"

        # Persistence metrics
        detections24h = int(2 + (i % 8))
        detections7d = int(8 + (i % 20))
        detections30d = int(15 + (i % 45))

        # Spatial lookup
        geo_context = find_nearest_facility(lat, lng)

        # Land cover estimation using dynamic GEE/heuristic logic
        from .landcover_gee import get_landcover_gee
        gee_lc = get_landcover_gee(lat, lng)

        # Risk scoring
        risk_data = ThermalRiskModel.compute_risk(frp, temperature, confidence)

        # AI classification
        ai_res = EventClassifier.classify(
            frp=frp,
            brightness=temperature,
            confidence=confidence,
            dist_km=geo_context["distanceKm"],
            landcover_id=gee_lc["primaryClassId"],
            detections_count=detections30d
        )

        persistence_level = "HIGH" if detections30d >= 35 else ("MEDIUM" if detections30d >= 18 else "LOW")

        event_id = f"TF-{str(10293 + i).zfill(5)}"

        events.append({
            "id": event_id,
            "lat": lat,
            "lng": lng,
            "classification": ai_res["classification"],
            "risk": risk_data["riskScore"],
            "riskLevel": risk_data["riskLevel"],
            "riskColor": risk_data["riskColor"],
            "confidence": ai_res["confidence"],
            "frp": frp,
            "temperature": temperature,
            "satellite": satellite,
            "sensor": sensor,
            "facility": geo_context["facility"],
            "facilityType": geo_context["facilityType"],
            "facilityDistance": geo_context["distanceStr"],
            "facilityDistanceKm": geo_context["distanceKm"],
            "distance": geo_context["distanceStr"],
            "landCover": gee_lc["primaryClassName"],
            "landCoverId": gee_lc["primaryClassId"],
            "landCoverColor": gee_lc["primaryColor"],
            "landCoverDistribution": gee_lc["distribution"],
            "persistence": persistence_level,
            "detections24h": detections24h,
            "detections7d": detections7d,
            "detections30d": detections30d,
            "probabilities": ai_res["probabilities"],
            "explanation": ai_res["explanation"],
            "factors": ai_res["factors"],
            "riskBreakdownBars": ai_res["riskBreakdownBars"],
            "timestamp": base_date.strftime("%Y-%m-%d %H:%M:%S UTC")
        })

    return events
