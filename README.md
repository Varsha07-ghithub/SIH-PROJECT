🛰️ Real-Time Emergency Response & Thermal Risk Intelligence Assistant

> **Geothermal AI Platform**: Multi-source Geospatial AI monitoring system integrating NASA FIRMS Real-Time Satellite Thermal Telemetry, Google Earth Engine Dynamic World 10m Land Cover Classification, and Multi-Factor Machine Learning Risk Scoring.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Satellite Data](https://img.shields.io/badge/NASA%20FIRMS-VIIRS%20%2F%20MODIS-red.svg)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Dynamic%20World-4285F4.svg)

---

## 🌟 Key Capabilities

### 📡 1. Real-Time NASA FIRMS Satellite Telemetry
- **Live Stream Synchronization**: Automated 15-second background auto-polling loop connecting to NASA VIIRS/MODIS Near-Real-Time (NRT) global satellite streams.
- **Global & Historical Coverage**: Ingests **3.58 Million historical records** alongside **worldwide global 2026 daily detections** (January 2026 – Present).

### ⚡ 2. Dynamic World Land Cover & Risk Inference Model
- **10m High-Resolution Land Cover**: Integrated Google Earth Engine (GEE) Dynamic World classification (Trees/Forest, Water, Built Area/Industrial, Cropland, Grass, Flooded Vegetation).
- **Multi-Factor Explainable AI**: Combines Fire Radiative Power (MW FRP), Brightness Temperature (K), Satellite Confidence %, and Industrial Facility Proximity to generate explainable 0–100 Risk Scores.

### 🎮 3. Executive Mission Control Dashboard
- **Unified Command Toolbar**: Glassmorphic 2-row header integrating real-time KPI stat pills (`📡 Live Detections`, `🔴 Critical`, `🟠 High`, `🟡 Moderate`, `🟢 Low`).
- **Global Quick Fly Regions**: Smooth camera flight buttons to instantly focus on `🌍 World`, `🇮🇳 India`, `🇺🇸 North America`, `🇪🇺 Europe`, `🌍 Africa`, `🌏 East Asia`, and `🇦🇺 Australia`.
- **Date Selector**: Select any observation date from **Jan 1, 2026 to Today**.

### 🎬 4. Multi-Temporal Hotspot Timelapse Playback
- **Interactive Timeline**: Click `▶ PLAY 2026 TIMELAPSE` to play an automated day-by-day playback animation showing satellite thermal anomaly expansion across global regions.

### 🚨 5. Real-Time Alert Center & High-Risk Audio Siren
- **Web Audio API Emergency Siren**: Synthesizes real-time audio warning alarms when `CRITICAL` (>80 Risk Score) anomalies appear.
- **TopBar Controls**: Toggle `🔊 SIREN ALARM: ON / 🔇 MUTED` directly in the navbar.
- **Interactive Operations**:
  - `🎯 VIEW ON MAP`: Switches tab to Mission Control, flies map to coordinates, and draws facility/thermal perimeters.
  - `⚡ INVESTIGATE`: Opens the ML Predictor modal with pre-filled telemetry.
  - `✓ ACKNOWLEDGE`: Toggles operational acknowledgment tracking.

### 📄 6. Official AI Incident PDF Generator & Multi-Format Exporters
- **One-Click PDF/Printable Reports**: Generates formal incident report documents for emergency dispatchers.
- **Real File Exports**: Download instant `.csv` data tables, `.geojson` spatial features, and `.kml` Google Earth files.

---

## 🏗️ Architecture & Technology Stack

```
   ┌─────────────────────────────────────────────────────────┐
   │            NASA FIRMS NRT Satellite Feed                │
   │           (SUOMI VIIRS C2 & MODIS Global 24h)           │
   └───────────────────────────┬─────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │            FastAPI Python Backend Server                │
   │       (firms_loader.py • landcover_gee.py • model.py)   │
   └───────────────────────────┬─────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │          Single Page Application (SPA UI)               │
   │     (Leaflet Satellite Maps + React JSX + Web Audio)    │
   └───────────────────────────┬─────────────────────────────┘
```

- **Backend**: Python 3.12, FastAPI, Uvicorn, Pandas, NumPy, Google Earth Engine SDK (`earthengine-api`).
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism design tokens), React (Babel standalone), Leaflet.js (Google Satellite Tiles, CartoDB Dark, Heatmap plugin).
- **Data Stores**: Spatial index data structures, 3.58M historical India CSV points, 20,000+ 2026 global daily records.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ installed
- Modern Web Browser (Chrome, Edge, Firefox, Safari)

### Installation & Launch

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Varsha07-ghithub/Real-Time-Emergency-Response-Triage-Assistant.git
   cd Real-Time-Emergency-Response-Triage-Assistant
   ```

2. **Install Required Python Packages**:
   ```bash
   pip install fastapi uvicorn pandas numpy requests earthengine-api
   ```

3. **Start the Application Server**:
   ```bash
   python start_localhost.py
   ```
   *Or run the batch script on Windows:*
   ```bash
   start_localhost.bat
   ```

4. **Access the Dashboard**:
   Open your browser at:
   👉 **`http://localhost:8000`**

---

## 📡 API Endpoints Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | System health check and server status |
| `/api/firms/live-sync` | `GET` | Ingests live Near-Real-Time NASA FIRMS satellite feed |
| `/api/firms/points?date=YYYY-MM-DD` | `GET` | Fetches filtered thermal detections by date |
| `/api/firms/dates` | `GET` | Lists available satellite observation dates (Jan 2026 – Present) |
| `/api/landcover/point` | `GET` | Queries GEE Dynamic World 10m land cover distribution |
| `/api/predict` | `POST` | Executes ML risk scoring & land cover inference |

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

### 🌐 GitHub Repository Link
👉 
