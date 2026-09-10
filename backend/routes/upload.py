"""
routes/upload.py: File Upload and Preprocessing Endpoint.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.preprocessing.compatibility_checker import CompatibilityChecker
from backend.preprocessing.geotiff_handler import to_thumbnail

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = Path("d:/SIH/backend/uploads")
THUMBNAIL_DIR = Path("d:/SIH/backend/uploads/thumbnails")
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
        if ext not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
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
            pil_thumb.save(thumb_path, "JPEG", quality=85)
            thumbnail_urls.append(f"/thumbnails/{thumb_name}")
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
            "recommended_tasks": report.recommended_tasks
        }
    }
