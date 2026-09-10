"""
Integrated Localhost Runner for SIH Project:
Starts the FastAPI Backend and serves the React/Leaflet Frontend on http://localhost:8000/
"""

import sys
import os
import time
import webbrowser
import threading
import uvicorn

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def open_browser():
    time.sleep(1.5)
    print("\n" + "="*70)
    print("SERVER READY AT: http://localhost:8000")
    print("="*70)
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass

def main():
    print("="*70)
    print("GEOTHERMAL AI & INDUSTRIAL FIRE INTELLIGENCE PLATFORM")
    print("   Integrating NASA FIRMS + Dynamic World Land Cover Model")
    print("="*70)
    print("Frontend: http://localhost:8000/")
    print("REST API: http://localhost:8000/api")
    print("Swagger Docs: http://localhost:8000/docs")
    print("ReDoc Docs: http://localhost:8000/redoc")
    print("="*70)
    print("Starting server on 127.0.0.1:8000 ... (Press CTRL+C to stop)\n")

    # Start browser opener in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Uvicorn ASGI server
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
