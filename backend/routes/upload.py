"""
routes/upload.py: File Upload and Preprocessing Endpoint.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from backend.preprocessing.compatibility_checker import CompatibilityChecker
from backend.preprocessing.geotiff_handler import to_thumbnail

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
THUMBNAIL_DIR = UPLOAD_DIR / "thumbnails"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

checker = CompatibilityChecker()

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accepts 1 or more satellite images (GeoTIFF, TIFF, PNG, JPEG).
    Validates sensor metadata, checks pair co-registration,
    generates quick-look thumbnails for UI display, and returns compatibility report.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    saved_paths = []
    thumbnail_urls = []
    
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in [".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Use GeoTIFF, TIFF, PNG, or JPEG.")
            
        unique_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
        dest_path = UPLOAD_DIR / unique_name
        
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
            
        saved_paths.append(str(dest_path))
        
        # Generate thumbnail
        thumb_name = f"thumb_{Path(unique_name).stem}.jpg"
        thumb_path = THUMBNAIL_DIR / thumb_name
        try:
            pil_thumb = to_thumbnail(str(dest_path), max_size=512)
            if pil_thumb is not None:
                pil_thumb.save(thumb_path, "JPEG", quality=85)
                thumbnail_urls.append(f"/thumbnails/{thumb_name}")
            else:
                thumbnail_urls.append("")
        except Exception:
            thumbnail_urls.append("")

    # Run compatibility check
    report = checker.check(saved_paths)
    
    return {
        "status": "success",
        "file_paths": saved_paths,
        "thumbnails": thumbnail_urls,
        "compatibility": {
            "valid": report.valid,
            "input_type": report.input_type,
            "sensor_types": report.sensor_types,
            "issues": report.issues,
            "warnings": report.warnings,
            "recommended_tasks": report.recommended_tasks,
            "file_infos": [
                {
                    "path": info.path,
                    "filename": Path(info.path).name,
                    "bands": info.bands,
                    "width": info.width,
                    "height": info.height,
                    "crs": info.crs,
                    "resolution_m": round(info.resolution_x, 2) if not (info.resolution_x != info.resolution_x) else 10.0,
                    "sensor_hint": info.sensor_hint,
                    "dtype": info.dtype,
                    "acquisition_date": info.acquisition_date.isoformat() if info.acquisition_date else None,
                }
                for info in report.file_infos
            ]
        }
    }
