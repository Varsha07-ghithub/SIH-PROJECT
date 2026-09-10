# ============================================================
# FIRE INTELLIGENCE DASHBOARD - MAIN BACKEND
# Flask + Random Forest + NASA FIRMS
# ============================================================

import os
import glob
import json
import math
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

# ------------------------------------------------------------
# OPTIONAL ML LIBRARIES
# ------------------------------------------------------------
try:
    import joblib
except Exception:
    joblib = None

try:
    import pickle
except Exception:
    pickle = None


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIRS = [
    BASE_DIR,
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "model"),
    "/content",
    "/content/models",
    "/content/model"
]

DATA_DIRS = [
    BASE_DIR,
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "datasets"),
    "/content",
    "/content/datasets"
]

MODEL = None
MODEL_PATH = None

EVENTS = []

FEATURES = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "confidence",
    "bright_t31",
    "frp",
    "daynight"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def clean_json(data):
    """
    Convert numpy values into normal Python values.
    """

    if isinstance(data, dict):
        return {
            str(k): clean_json(v)
            for k, v in data.items()
        }

    if isinstance(data, list):
        return [
            clean_json(v)
            for v in data
        ]

    if isinstance(data, tuple):
        return [
            clean_json(v)
            for v in data
        ]

    if isinstance(data, np.integer):
        return int(data)

    if isinstance(data, np.floating):
        return float(data)

    if isinstance(data, np.ndarray):
        return data.tolist()

    return data


def find_model():
    """
    Automatically searches common model locations.
    """

    patterns = [
        "*.pkl",
        "*.joblib",
        "*.pickle"
    ]

    candidates = []

    for directory in MODEL_DIRS:

        if not os.path.exists(directory):
            continue

        for pattern in patterns:

            candidates.extend(
                glob.glob(
                    os.path.join(directory, pattern)
                )
            )

    # Prefer filenames containing fire/rf/random/model
    priority = []

    for path in candidates:

        name = os.path.basename(path).lower()

        score = 0

        if "fire" in name:
            score += 5

        if "random" in name:
            score += 5

        if "forest" in name:
            score += 5

        if "model" in name:
            score += 3

        if "rf" in name:
            score += 4

        priority.append((score, path))

    priority.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if priority:
        return priority[0][1]

    return None


def load_model():

    global MODEL
    global MODEL_PATH

    if joblib is None:
        print("WARNING: joblib is not installed.")
        return None

    path = find_model()

    if path is None:

        print("WARNING: No ML model found.")
        print("Put your .pkl or .joblib model beside main.py")
        print("or inside a models folder.")

        return None

    try:

        MODEL = joblib.load(path)
        MODEL_PATH = path

        print()
        print("=" * 60)
        print("ML MODEL LOADED")
        print("=" * 60)
        print("Model:", path)
        print("Type :", type(MODEL))
        print("=" * 60)
        print()

        return MODEL

    except Exception as e:

        print("ERROR loading model:")
        print(e)

        MODEL = None
        MODEL_PATH = None

        return None


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_features(data):

    row = {
        "latitude": safe_float(
            data.get("latitude")
        ),

        "longitude": safe_float(
            data.get("longitude")
        ),

        "brightness": safe_float(
            data.get("brightness", 310)
        ),

        "scan": safe_float(
            data.get("scan", 1)
        ),

        "track": safe_float(
            data.get("track", 1)
        ),

        "confidence": safe_float(
            data.get("confidence", 70)
        ),

        "bright_t31": safe_float(
            data.get("bright_t31", 290)
        ),

        "frp": safe_float(
            data.get("frp", 10)
        ),

        "daynight": safe_int(
            data.get("daynight", 1)
        )
    }

    df = pd.DataFrame(
        [row],
        columns=FEATURES
    )

    return df


# ============================================================
# FALLBACK FIRE CLASSIFICATION
# ============================================================

def fallback_prediction(data):

    brightness = safe_float(
        data.get("brightness", 310)
    )

    frp = safe_float(
        data.get("frp", 10)
    )

    confidence = safe_float(
        data.get("confidence", 70)
    )

    bright_t31 = safe_float(
        data.get("bright_t31", 290)
    )

    temperature_difference = (
        brightness - bright_t31
    )

    score = 0

    if brightness >= 330:
        score += 35
    elif brightness >= 315:
        score += 20
    elif brightness >= 305:
        score += 10

    if frp >= 50:
        score += 35
    elif frp >= 20:
        score += 20
    elif frp >= 5:
        score += 10

    if confidence >= 80:
        score += 20
    elif confidence >= 50:
        score += 10

    if temperature_difference >= 25:
        score += 10
    elif temperature_difference >= 15:
        score += 5

    score = min(100, score)

    if score >= 75:
        prediction = 3
    elif score >= 45:
        prediction = 2
    elif score >= 20:
        prediction = 1
    else:
        prediction = 0

    probabilities = [
        ["No Fire", max(0, 100 - score)],
        ["Low Fire", min(100, score * 0.25)],
        ["Moderate Fire", min(100, score * 0.50)],
        ["High Fire", min(100, score * 0.25)]
    ]

    total = sum(x[1] for x in probabilities)

    if total > 0:

        probabilities = [
            [
                name,
                round(value * 100 / total, 2)
            ]

            for name, value in probabilities
        ]

    return prediction, probabilities


# ============================================================
# MODEL PREDICTION
# ============================================================

def run_prediction(data):

    features = prepare_features(data)

    # --------------------------------------------------------
    # REAL MODEL
    # --------------------------------------------------------

    if MODEL is not None:

        try:

            prediction = MODEL.predict(
                features
            )[0]

            prediction = safe_int(
                prediction
            )

            probabilities = None

            # Random Forest supports predict_proba
            if hasattr(MODEL, "predict_proba"):

                try:

                    proba = MODEL.predict_proba(
                        features
                    )[0]

                    probabilities = [
                        [
                            f"Class {i}",
                            round(
                                float(p) * 100,
                                2
                            )
                        ]

                        for i, p in enumerate(proba)
                    ]

                except Exception:
                    probabilities = None

            if probabilities is None:

                probabilities = [
                    [
                        f"Class {prediction}",
                        100
                    ]
                ]

            return prediction, probabilities, True

        except Exception as e:

            print()
            print("MODEL PREDICTION ERROR")
            print(e)
            print(traceback.format_exc())
            print()

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    prediction, probabilities = fallback_prediction(
        data
    )

    return prediction, probabilities, False


# ============================================================
# CLASSIFICATION NAME
# ============================================================

def classification_name(class_id):

    mapping = {
        0: "No Fire",
        1: "Low Fire",
        2: "Moderate Fire",
        3: "High Fire"
    }

    return mapping.get(
        safe_int(class_id),
        "Unknown"
    )


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(class_id):

    class_id = safe_int(class_id)

    if class_id == 3:
        return "CRITICAL"

    if class_id == 2:
        return "HIGH"

    if class_id == 1:
        return "MODERATE"

    return "LOW"


# ============================================================
# LAND COVER FALLBACK
# ============================================================

def estimate_landcover(lat, lon):

    """
    Safe fallback.

    This DOES NOT pretend to have real satellite
    land-cover information.

    Replace this function later with Google Earth Engine /
    Dynamic World API.
    """

    lat = safe_float(lat)
    lon = safe_float(lon)

    return {
        "land_cover": "Unknown",
        "land_cover_source": "Not connected",
        "confidence": None,
        "message": (
            "Connect Google Earth Engine / Dynamic World "
            "for real land-cover classification."
        )
    }


# ============================================================
# EVENT CREATOR
# ============================================================

def create_event(data, prediction, probabilities, model_used):

    lat = safe_float(
        data.get("latitude")
    )

    lon = safe_float(
        data.get("longitude")
    )

    classification = classification_name(
        prediction
    )

    risk = risk_level(
        prediction
    )

    event = {

        "id": (
            "FIRE-"
            + datetime.now().strftime(
                "%Y%m%d%H%M%S%f"
            )
        ),

        "latitude": lat,

        "longitude": lon,

        "brightness": safe_float(
            data.get("brightness", 0)
        ),

        "frp": safe_float(
            data.get("frp", 0)
        ),

        "confidence": safe_float(
            data.get("confidence", 0)
        ),

        "bright_t31": safe_float(
            data.get("bright_t31", 0)
        ),

        "scan": safe_float(
            data.get("scan", 0)
        ),

        "track": safe_float(
            data.get("track", 0)
        ),

        "daynight": safe_int(
            data.get("daynight", 1)
        ),

        "prediction": safe_int(
            prediction
        ),

        "type": safe_int(
            prediction
        ),

        "classification": classification,

        "risk": risk,

        "risk_level": risk,

        "probabilities": probabilities,

        "model_used": model_used,

        "timestamp": datetime.now().isoformat(),

        "land_cover": "Unknown",

        "state": "Unknown",

        "district": "Unknown"
    }

    return event


# ============================================================
# ROUTE: HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "name": "Fire Intelligence Backend",
        "version": "1.0",
        "message": "Backend is running successfully.",
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH
    })


# ============================================================
# ROUTE: HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "backend": "online",
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "events": len(EVENTS),
        "time": datetime.now().isoformat()
    })


# ============================================================
# ROUTE: PREDICT
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:
            data = {}

        # ----------------------------------------------------
        # Validate coordinates
        # ----------------------------------------------------

        if "latitude" not in data:
            return jsonify({
                "success": False,
                "error": "latitude is required"
            }), 400

        if "longitude" not in data:
            return jsonify({
                "success": False,
                "error": "longitude is required"
            }), 400

        lat = safe_float(
            data.get("latitude")
        )

        lon = safe_float(
            data.get("longitude")
        )

        if not -90 <= lat <= 90:

            return jsonify({
                "success": False,
                "error": "Invalid latitude"
            }), 400

        if not -180 <= lon <= 180:

            return jsonify({
                "success": False,
                "error": "Invalid longitude"
            }), 400

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction, probabilities, model_used = (
            run_prediction(data)
        )

        # ----------------------------------------------------
        # Event
        # ----------------------------------------------------

        event = create_event(
            data,
            prediction,
            probabilities,
            model_used
        )

        # ----------------------------------------------------
        # Land cover
        # ----------------------------------------------------

        landcover = estimate_landcover(
            lat,
            lon
        )

        event.update(
            landcover
        )

        # ----------------------------------------------------
        # Store event
        # ----------------------------------------------------

        EVENTS.insert(
            0,
            event
        )

        # Keep only latest 500 events
        del EVENTS[500:]

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = {

            "success": True,

            "prediction": prediction,

            "type": prediction,

            "classification": classification_name(
                prediction
            ),

            "risk": risk_level(
                prediction
            ),

            "risk_level": risk_level(
                prediction
            ),

            "probabilities": probabilities,

            "model_used": model_used,

            "model_loaded": MODEL is not None,

            "event": event,

            "coordinates": {
                "latitude": lat,
                "longitude": lon
            },

            "land_cover": event.get(
                "land_cover",
                "Unknown"
            ),

            "state": event.get(
                "state",
                "Unknown"
            ),

            "district": event.get(
                "district",
                "Unknown"
            )
        }

        return jsonify(
            clean_json(response)
        )

    except Exception as e:

        print()
        print("PREDICTION ERROR")
        print(e)
        print(traceback.format_exc())
        print()

        return jsonify({

            "success": False,

            "error": str(e),

            "message": (
                "Prediction failed. "
                "Check backend terminal."
            )

        }), 500


# ============================================================
# ROUTE: EVENTS
# ============================================================

@app.route("/api/events", methods=["GET"])
def get_events():

    return jsonify(
        clean_json(EVENTS)
    )


# ============================================================
# ROUTE: STATS
# ============================================================

@app.route("/api/stats", methods=["GET"])
def get_stats():

    total = len(EVENTS)

    fire_events = [
        e for e in EVENTS
        if safe_int(
            e.get("prediction", 0)
        ) in [1, 2, 3]
    ]

    no_fire_events = [
        e for e in EVENTS
        if safe_int(
            e.get("prediction", 0)
        ) == 0
    ]

    high_risk = [
        e for e in EVENTS
        if safe_int(
            e.get("prediction", 0)
        ) == 3
    ]

    moderate_risk = [
        e for e in EVENTS
        if safe_int(
            e.get("prediction", 0)
        ) == 2
    ]

    low_risk = [
        e for e in EVENTS
        if safe_int(
            e.get("prediction", 0)
        ) == 1
    ]

    frp_values = [
        safe_float(e.get("frp", 0))
        for e in fire_events
    ]

    avg_frp = (
        sum(frp_values) / len(frp_values)
        if frp_values
        else 0
    )

    max_frp = (
        max(frp_values)
        if frp_values
        else 0
    )

    return jsonify({

        "success": True,

        "total_events": total,

        "total": total,

        "fire_events": len(fire_events),

        "fire_count": len(fire_events),

        "no_fire_events": len(no_fire_events),

        "high_risk": len(high_risk),

        "moderate_risk": len(moderate_risk),

        "low_risk": len(low_risk),

        "average_frp": round(
            avg_frp,
            2
        ),

        "max_frp": round(
            max_frp,
            2
        ),

        "model_loaded": MODEL is not None

    })


# ============================================================
# ROUTE: NASA FIRMS POINTS
# ============================================================

@app.route(
    "/api/firms/points",
    methods=["GET"]
)
def firms_points():

    try:

        lat_min = safe_float(
            request.args.get(
                "lat_min",
                6
            )
        )

        lat_max = safe_float(
            request.args.get(
                "lat_max",
                38
            )
        )

        lon_min = safe_float(
            request.args.get(
                "lon_min",
                66
            )
        )

        lon_max = safe_float(
            request.args.get(
                "lon_max",
                100
            )
        )

        limit = safe_int(
            request.args.get(
                "limit",
                1000
            ),
            1000
        )

        limit = min(
            max(limit, 1),
            10000
        )

        # ----------------------------------------------------
        # Search CSV files
        # ----------------------------------------------------

        csv_files = []

        for directory in DATA_DIRS:

            if os.path.exists(directory):

                csv_files.extend(
                    glob.glob(
                        os.path.join(
                            directory,
                            "**",
                            "*.csv"
                        ),
                        recursive=True
                    )
                )

        # Remove duplicates
        csv_files = list(
            dict.fromkeys(csv_files)
        )

        selected_file = None

        # Prefer FIRMS files
        for path in csv_files:

            name = os.path.basename(
                path
            ).lower()

            if (
                "fire" in name
                or "firms" in name
            ):

                selected_file = path
                break

        if selected_file is None:

            if csv_files:
                selected_file = csv_files[0]

        if selected_file is None:

            return jsonify({

                "success": True,

                "points": [],

                "count": 0,

                "message": (
                    "No FIRMS CSV file found."
                )

            })

        # ----------------------------------------------------
        # Read only required columns
        # ----------------------------------------------------

        df = pd.read_csv(
            selected_file,
            low_memory=False
        )

        if (
            "latitude" not in df.columns
            or
            "longitude" not in df.columns
        ):

            return jsonify({

                "success": False,

                "error": (
                    "CSV does not contain "
                    "latitude/longitude columns."
                )

            }), 400

        df = df[
            (df["latitude"] >= lat_min)
            &
            (df["latitude"] <= lat_max)
            &
            (df["longitude"] >= lon_min)
            &
            (df["longitude"] <= lon_max)
        ]

        # ----------------------------------------------------
        # Limit data
        # ----------------------------------------------------

        df = df.head(
            limit
        )

        points = []

        for _, row in df.iterrows():

            point = {

                "latitude": safe_float(
                    row.get(
                        "latitude",
                        0
                    )
                ),

                "longitude": safe_float(
                    row.get(
                        "longitude",
                        0
                    )
                ),

                "brightness": safe_float(
                    row.get(
                        "brightness",
                        0
                    )
                ),

                "frp": safe_float(
                    row.get(
                        "frp",
                        0
                    )
                ),

                "confidence": safe_float(
                    row.get(
                        "confidence",
                        0
                    )
                )
            }

            if "type" in df.columns:

                point["type"] = safe_int(
                    row.get(
                        "type",
                        0
                    )
                )

                point["classification"] = (
                    classification_name(
                        point["type"]
                    )
                )

            points.append(
                point
            )

        return jsonify({

            "success": True,

            "points": clean_json(
                points
            ),

            "count": len(points),

            "source": selected_file

        })

    except Exception as e:

        print(
            "FIRMS ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# ROUTE: LAND COVER
# ============================================================

@app.route(
    "/api/landcover/point",
    methods=["GET"]
)
def landcover_point():

    try:

        lat = safe_float(
            request.args.get(
                "lat"
            )
        )

        lon = safe_float(
            request.args.get(
                "lon"
            )
        )

        if not (-90 <= lat <= 90):

            return jsonify({
                "success": False,
                "error": "Invalid latitude"
            }), 400

        if not (-180 <= lon <= 180):

            return jsonify({
                "success": False,
                "error": "Invalid longitude"
            }), 400

        result = estimate_landcover(
            lat,
            lon
        )

        return jsonify({

            "success": True,

            "latitude": lat,

            "longitude": lon,

            **result

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# ROUTE: CLEAR EVENTS
# ============================================================

@app.route(
    "/api/events/clear",
    methods=["POST"]
)
def clear_events():

    EVENTS.clear()

    return jsonify({

        "success": True,

        "message": "Events cleared."

    })


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "error": "API route not found",

        "path": request.path

    }), 404


@app.errorhandler(500)
def internal_error(error):

    return jsonify({

        "success": False,

        "error": "Internal server error"

    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("🔥 FIRE INTELLIGENCE DASHBOARD BACKEND")
    print("=" * 70)

    print()
    print("Loading ML model...")
    load_model()

    print()
    print("=" * 70)
    print("SERVER STARTING")
    print("=" * 70)

    print("Frontend API:")
    print("http://127.0.0.1:8000")

    print()
    print("Health:")
    print("http://127.0.0.1:8000/api/health")

    print()
    print("API Routes:")
    print("GET  /")
    print("GET  /api/health")
    print("POST /api/predict")
    print("GET  /api/events")
    print("GET  /api/stats")
    print("GET  /api/firms/points")
    print("GET  /api/landcover/point")
    print("POST /api/events/clear")

    print()
    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )
