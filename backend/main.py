"""
main.py: FastAPI ML Service & Orchestration Entrypoint for SatQuery AI.
SIH26167 | Indian Space Research Organisation (ISRO)
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes.inference import router as inference_router
from backend.routes.upload import router as upload_router
from backend.routes.report import router as report_router
from backend.model_manager import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("satquery.server")

# Directory setup with dynamic cross-platform paths (Mac/Linux/Windows)
UPLOADS_DIR = PROJECT_ROOT / "backend" / "uploads"
THUMBNAILS_DIR = UPLOADS_DIR / "thumbnails"
REPORTS_DIR = PROJECT_ROOT / "backend" / "reports"
TRACES_DIR = PROJECT_ROOT / "backend" / "traces"

for d in [UPLOADS_DIR, THUMBNAILS_DIR, REPORTS_DIR, TRACES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=========================================================")
    logger.info("🛰️  Starting SatQuery AI Backend (ISRO SIH26167)")
    logger.info("   Model: SatQueryUnified (RemoteCLIP ViT-L/14 Shared Backbone)")
    logger.info("   Heads: VQA, Grounding (DETR), Change (Siamese), Fusion (X-Attn)")
    logger.info("   Sensors: Sentinel-1/2, Cartosat-2S, RISAT-1/2")
    logger.info("   Environment: Dynamic Path Resolution & Apple Silicon / CUDA / CPU Engine")
    logger.info("=========================================================")
    
    # Warm load the model in memory
    try:
        ModelManager.get_instance()
    except Exception as e:
        logger.warning("Model initialization notice: %s", e)
        
    yield
    # Shutdown
    logger.info("SatQuery AI backend shutting down.")


app = FastAPI(
    title="SatQuery AI API",
    description="Agentic Vision-Language Assistant for Multimodal Remote Sensing Analysis",
    version="2.0.0",
    lifespan=lifespan
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows Next.js local frontend on port 3000 / 3001
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
app.mount("/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# Include Routers
app.include_router(inference_router)
app.include_router(upload_router)
app.include_router(report_router)


@app.get("/health")
async def health():
    mgr = None
    device_info = "cpu"
    try:
        mgr = ModelManager.get_instance()
        device_info = str(mgr.device)
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "SatQuery AI",
        "ps_number": "SIH26167",
        "organization": "ISRO",
        "device": device_info,
        "capabilities": [
            "single_image_vqa",
            "text_guided_region_grounding",
            "multitemporal_change_understanding",
            "optical_sar_cross_modal_analysis",
            "agentic_orchestration"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
