import os
import glob
import math
import traceback
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Fire Intelligence API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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

MODEL = None
MODEL_PATH = None
MODEL_ERROR = None

EVENTS = []


# ============================================================
# SAFE CONVERSION
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        value = float(value)

        if math.isnan(value):
            return default

        if math.isinf(value):
            return default

        return value

    except Exception:

        return default


def safe_int(value, default=0):

    try:
        return int(float(value))

    except Exception:

        return default


# ============================================================
# FIND MODEL
# ============================================================

def find_model():

    possible_names = [
        "model.pkl",
        "fire_model.pkl",
        "random_forest.pkl",
        "random_forest_model.pkl",
        "rf_model.pkl",
        "fire_rf_model.pkl",
        "model.joblib",
        "fire_model.joblib",
        "random_forest.joblib"
    ]

    # First check exact names
    for name in possible_names:

        path = os.path.join(
            BASE_DIR,
            name
        )

        if os.path.isfile(path):

            return path


    # Then search recursively
    patterns = [
        "*.pkl",
        "*.joblib",
        "*.pickle"
    ]

    candidates = []

    for pattern in patterns:

        candidates.extend(
            glob.glob(
                os.path.join(
                    BASE_DIR,
                    "**",
                    pattern
                ),
                recursive=True
            )
        )


    if not candidates:

        return None


    # Prefer files containing model/fire/rf
    preferred = []

    for path in candidates:

        filename = os.path.basename(
            path
        ).lower()

        score = 0

        if "model" in filename:
            score += 5

        if "fire" in filename:
            score += 5

        if "random" in filename:
            score += 5

        if "forest" in filename:
            score += 5

        if filename.startswith("rf"):
            score += 5

        preferred.append(
            (score, path)
        )


    preferred.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return preferred[0][1]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global MODEL
    global MODEL_PATH
    global MODEL_ERROR

    try:

        path = find_model()

        if path is None:

            MODEL = None
            MODEL_PATH = None

            MODEL_ERROR = (
                "No model file found. "
                "Upload your .pkl or .joblib model."
            )

            print(MODEL_ERROR)

            return


        print(
            "Loading model:",
            path
        )


        MODEL = joblib.load(
            path
        )

        MODEL_PATH = path

        MODEL_ERROR = None

        print(
            "MODEL LOADED SUCCESSFULLY"
        )

        print(
            "Model type:",
            type(MODEL)
        )


    except Exception as e:

        MODEL = None
        MODEL_PATH = None

        MODEL_ERROR = str(e)

        print(
            "MODEL LOAD ERROR:"
        )

        print(
            traceback.format_exc()
        )


# ============================================================
# LOAD MODEL
# ============================================================

load_model()


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    data: Dict[str, Any]
):

    row = {

        "latitude":
            safe_float(
                data.get(
                    "latitude"
                )
            ),

        "longitude":
            safe_float(
                data.get(
                    "longitude"
                )
            ),

        "brightness":
            safe_float(
                data.get(
                    "brightness",
                    310
                )
            ),

        "scan":
            safe_float(
                data.get(
                    "scan",
                    1
                )
            ),

        "track":
            safe_float(
                data.get(
                    "track",
                    1
                )
            ),

        "confidence":
            safe_float(
                data.get(
                    "confidence",
                    70
                )
            ),

        "bright_t31":
            safe_float(
                data.get(
                    "bright_t31",
                    290
                )
            ),

        "frp":
            safe_float(
                data.get(
                    "frp",
                    10
                )
            ),

        "daynight":
            safe_int(
                data.get(
                    "daynight",
                    1
                )
            )
    }


    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


# ============================================================
# FALLBACK PREDICTION
# ============================================================

def fallback_prediction(data):

    brightness = safe_float(
        data.get(
            "brightness",
            310
        )
    )

    frp = safe_float(
        data.get(
            "frp",
            10
        )
    )

    confidence = safe_float(
        data.get(
            "confidence",
            70
        )
    )

    bright_t31 = safe_float(
        data.get(
            "bright_t31",
            290
        )
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


    if (
        brightness -
        bright_t31
    ) >= 25:

        score += 10

    elif (
        brightness -
        bright_t31
    ) >= 15:

        score += 5


    score = min(
        score,
        100
    )


    if score >= 75:

        prediction = 3

    elif score >= 45:

        prediction = 2

    elif score >= 20:

        prediction = 1

    else:

        prediction = 0


    # Simple probabilities
    probabilities = {

        "No Fire": 0.0,

        "Low Fire": 0.0,

        "Moderate Fire": 0.0,

        "High Fire": 0.0
    }


    if prediction == 0:

        probabilities["No Fire"] = 100

    elif prediction == 1:

        probabilities["Low Fire"] = 100

    elif prediction == 2:

        probabilities["Moderate Fire"] = 100

    else:

        probabilities["High Fire"] = 100


    return prediction, probabilities


# ============================================================
# PREDICTION
# ============================================================

def run_prediction(data):

    features = prepare_features(
        data
    )


    # Use real ML model
    if MODEL is not None:

        try:

            prediction = MODEL.predict(
                features
            )[0]

            prediction = safe_int(
                prediction
            )


            probabilities = {}


            if hasattr(
                MODEL,
                "predict_proba"
            ):

                try:

                    proba = (
                        MODEL
                        .predict_proba(
                            features
                        )[0]
                    )


                    classes = getattr(
                        MODEL,
                        "classes_",
                        range(len(proba))
                    )


                    for cls, probability in zip(
                        classes,
                        proba
                    ):

                        probabilities[
                            classification_name(
                                cls
                            )
                        ] = round(
                            float(
                                probability
                            ) * 100,
                            2
                        )


                except Exception:

                    probabilities = {}


            if not probabilities:

                probabilities = {

                    classification_name(
                        prediction
                    ): 100

                }


            return (
                prediction,
                probabilities,
                True
            )


        except Exception as e:

            print(
                "Prediction failed:"
            )

            print(
                traceback.format_exc()
            )


    # Fallback
    prediction, probabilities = (
        fallback_prediction(
            data
        )
    )


    return (
        prediction,
        probabilities,
        False
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classification_name(
    class_id
):

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
# RISK
# ============================================================

def risk_level(
    class_id
):

    class_id = safe_int(
        class_id
    )


    if class_id == 3:
        return "CRITICAL"

    if class_id == 2:
        return "HIGH"

    if class_id == 1:
        return "MODERATE"

    return "LOW"


# ============================================================
# LAND COVER
# ============================================================

def estimate_landcover(
    lat,
    lon
):

    # Earth Engine / Dynamic World
    # can be connected here later.

    return {

        "land_cover":
            "Unknown",

        "land_cover_source":
            "Not connected",

        "land_cover_confidence":
            None,

        "state":
            "Unknown",

        "district":
            "Unknown"
    }


# ============================================================
# CREATE EVENT
# ============================================================

def create_event(
    data,
    prediction,
    probabilities,
    model_used
):

    lat = safe_float(
        data.get(
            "latitude"
        )
    )

    lon = safe_float(
        data.get(
            "longitude"
        )
    )


    event = {

        "id":
            "FIRE-" +
            datetime.now().strftime(
                "%Y%m%d%H%M%S%f"
            ),

        "latitude":
            lat,

        "longitude":
            lon,

        "brightness":
            safe_float(
                data.get(
                    "brightness",
                    0
                )
            ),

        "scan":
            safe_float(
                data.get(
                    "scan",
                    0
                )
            ),

        "track":
            safe_float(
                data.get(
                    "track",
                    0
                )
            ),

        "confidence":
            safe_float(
                data.get(
                    "confidence",
                    0
                )
            ),

        "bright_t31":
            safe_float(
                data.get(
                    "bright_t31",
                    0
                )
            ),

        "frp":
            safe_float(
                data.get(
                    "frp",
                    0
                )
            ),

        "daynight":
            safe_int(
                data.get(
                    "daynight",
                    1
                )
            ),

        "prediction":
            prediction,

        "type":
            prediction,

        "classification":
            classification_name(
                prediction
            ),

        "risk":
            risk_level(
                prediction
            ),

        "risk_level":
            risk_level(
                prediction
            ),

        "probabilities":
            probabilities,

        "model_used":
            model_used,

        "timestamp":
            datetime.now().isoformat()
    }


    event.update(
        estimate_landcover(
            lat,
            lon
        )
    )


    return event


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "status": "online",

        "message":
            "Fire Intelligence API is running",

        "framework":
            "FastAPI",

        "model_loaded":
            MODEL is not None
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "status":
            "healthy",

        "backend":
            "online",

        "framework":
            "FastAPI",

        "model_loaded":
            MODEL is not None,

        "model_file":
            os.path.basename(
                MODEL_PATH
            )
            if MODEL_PATH
            else None,

        "model_error":
            MODEL_ERROR,

        "events":
            len(EVENTS)
    }


# ============================================================
# DEBUG
# ============================================================

@app.get("/api/debug")
def debug():

    try:

        files = os.listdir(
            BASE_DIR
        )

    except Exception:

        files = []


    return {

        "base_directory":
            BASE_DIR,

        "files":
            files,

        "model_loaded":
            MODEL is not None,

        "model_path":
            MODEL_PATH,

        "model_error":
            MODEL_ERROR
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/api/predict")
async def predict(
    request: Request
):

    try:

        data = await request.json()


        if not isinstance(
            data,
            dict
        ):

            return JSONResponse(

                status_code=400,

                content={

                    "success":
                        False,

                    "error":
                        "Request body must be JSON."
                }
            )


        if "latitude" not in data:

            return JSONResponse(

                status_code=400,

                content={

                    "success":
                        False,

                    "error":
                        "latitude is required"
                }
            )


        if "longitude" not in data:

            return JSONResponse(

                status_code=400,

                content={

                    "success":
                        False,

                    "error":
                        "longitude is required"
                }
            )


        lat = safe_float(
            data["latitude"]
        )

        lon = safe_float(
            data["longitude"]
        )


        if not -90 <= lat <= 90:

            return JSONResponse(

                status_code=400,

                content={

                    "success":
                        False,

                    "error":
                        "Invalid latitude"
                }
            )


        if not -180 <= lon <= 180:

            return JSONResponse(

                status_code=400,

                content={

                    "success":
                        False,

                    "error":
                        "Invalid longitude"
                }
            )


        prediction, probabilities, model_used = (
            run_prediction(
                data
            )
        )


        event = create_event(

            data,

            prediction,

            probabilities,

            model_used
        )


        EVENTS.insert(
            0,
            event
        )


        # Keep only latest 500 events
        del EVENTS[500:]


        return {

            "success":
                True,

            "prediction":
                prediction,

            "type":
                prediction,

            "classification":
                classification_name(
                    prediction
                ),

            "risk":
                risk_level(
                    prediction
                ),

            "risk_level":
                risk_level(
                    prediction
                ),

            "probabilities":
                probabilities,

            "model_used":
                model_used,

            "model_loaded":
                MODEL is not None,

            "coordinates": {

                "latitude":
                    lat,

                "longitude":
                    lon
            },

            "event":
                event
        }


    except Exception as e:

        print(
            traceback.format_exc()
        )


        return JSONResponse(

            status_code=500,

            content={

                "success":
                    False,

                "error":
                    str(e)
            }
        )


# ============================================================
# EVENTS
# ============================================================

@app.get("/api/events")
def get_events():

    return {

        "success":
            True,

        "events":
            EVENTS,

        "count":
            len(EVENTS)
    }


# ============================================================
# STATS
# ============================================================

@app.get("/api/stats")
def get_stats():

    total = len(
        EVENTS
    )


    no_fire = 0
    low = 0
    moderate = 0
    high = 0


    frps = []


    for event in EVENTS:

        prediction = safe_int(
            event.get(
                "prediction",
                0
            )
        )


        if prediction == 0:

            no_fire += 1

        elif prediction == 1:

            low += 1

        elif prediction == 2:

            moderate += 1

        elif prediction == 3:

            high += 1


        if prediction > 0:

            frps.append(
                safe_float(
                    event.get(
                        "frp",
                        0
                    )
                )
            )


    fire_count = (
        low +
        moderate +
        high
    )


    average_frp = (
        sum(frps) /
        len(frps)
        if frps
        else 0
    )


    max_frp = (
        max(frps)
        if frps
        else 0
    )


    return {

        "success":
            True,

        "total":
            total,

        "total_events":
            total,

        "fire_count":
            fire_count,

        "fire_events":
            fire_count,

        "no_fire_events":
            no_fire,

        "low_risk":
            low,

        "moderate_risk":
            moderate,

        "high_risk":
            high,

        "average_frp":
            round(
                average_frp,
                2
            ),

        "max_frp":
            round(
                max_frp,
                2
            ),

        "model_loaded":
            MODEL is not None
    }


# ============================================================
# LAND COVER POINT
# ============================================================

@app.get("/api/landcover/point")
def landcover_point(

    lat: float = Query(...),

    lon: float = Query(...)

):

    if not -90 <= lat <= 90:

        return JSONResponse(

            status_code=400,

            content={

                "success":
                    False,

                "error":
                    "Invalid latitude"
            }
        )


    if not -180 <= lon <= 180:

        return JSONResponse(

            status_code=400,

            content={

                "success":
                    False,

                "error":
                    "Invalid longitude"
            }
        )


    return {

        "success":
            True,

        "latitude":
            lat,

        "longitude":
            lon,

        **estimate_landcover(
            lat,
            lon
        )
    }


# ============================================================
# CLEAR EVENTS
# ============================================================

@app.post("/api/events/clear")
def clear_events():

    EVENTS.clear()


    return {

        "success":
            True,

        "message":
            "All events cleared"
    }


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        )
    )
