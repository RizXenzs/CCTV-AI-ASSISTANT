"""
server.py — FastAPI application server.
Serves both the REST API and the built React frontend as static files.
The user only needs to run `python src/main.py` and open http://localhost:8000.
"""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.routes import router

logger = logging.getLogger(__name__)

# Path to the built React frontend
DASHBOARD_DIR = Path(__file__).parent.parent.parent / "dashboard" / "dist"


def create_app(cctv_app) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    api_app = FastAPI(
        title="CCTV AI Dashboard API",
        description="REST API and MJPEG streams for the CCTV AI Detection system",
        version="1.0.0"
    )

    # Add CORS middleware to allow the frontend to access the API
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach the main CCTV application state so routes can access it
    api_app.state.cctv_app = cctv_app

    # Include API routes under /api
    api_app.include_router(router, prefix="/api")

    # --- Serve the built React frontend ---
    if DASHBOARD_DIR.exists():
        # Serve static assets (JS, CSS, images) from the dist/assets folder
        assets_dir = DASHBOARD_DIR / "assets"
        if assets_dir.exists():
            api_app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Catch-all route: serve index.html for any non-API path
        # This enables React Router's client-side routing
        @api_app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            # If the file exists in dist, serve it (e.g., favicon.ico, manifest.json)
            file_path = DASHBOARD_DIR / full_path
            if full_path and file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            # Otherwise, serve index.html (SPA fallback)
            return FileResponse(str(DASHBOARD_DIR / "index.html"))

        logger.info("Dashboard frontend enabled: serving from %s", DASHBOARD_DIR)
    else:
        @api_app.get("/")
        async def root():
            return {
                "status": "ok", 
                "service": "CCTV AI Backend",
                "message": "Dashboard not built yet. Run 'cd dashboard && npm run build' first."
            }
        logger.warning("Dashboard not found at %s — run 'cd dashboard && npm run build'", DASHBOARD_DIR)

    return api_app
