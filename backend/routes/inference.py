"""
routes/inference.py: API Endpoints for SatQuery AI.
"""

import os
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from backend.controller.agent import AgenticController
from backend.model_manager import ModelManager

router = APIRouter(prefix="/api", tags=["inference"])

# Request models
class QueryRequest(BaseModel):
    query: str
    file_paths: List[str]

class VQARequest(BaseModel):
    file_path: str
    question: str

class GroundingRequest(BaseModel):
    file_path: str
    phrase: str

class ChangeRequest(BaseModel):
    image1_path: str
    image2_path: str
    question: Optional[str] = None

class FusionRequest(BaseModel):
    optical_path: str
    sar_path: str
    question: str


# Controller & Manager references
_controller: Optional[AgenticController] = None
_model_manager: Optional[ModelManager] = None

def get_controller() -> AgenticController:
    global _controller
    if _controller is None:
        _controller = AgenticController()
    return _controller

def get_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager.get_instance()
    return _model_manager


@router.post("/query")
async def agentic_query(req: QueryRequest):
    """
    Main Agentic Endpoint: Automatically analyzes the query, checks inputs,
    selects the proper specialist model, and returns evidence + execution trace.
    """
    controller = get_controller()
    res = controller.execute_query(req.query, req.file_paths)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@router.post("/vqa")
async def direct_vqa(req: VQARequest):
    """Direct single-image VQA & captioning endpoint."""
    mgr = get_manager()
    return mgr.run_vqa(req.file_path, req.question)


@router.post("/grounding")
async def direct_grounding(req: GroundingRequest):
    """Direct text-guided region grounding endpoint."""
    mgr = get_manager()
    return mgr.run_grounding(req.file_path, req.phrase)


@router.post("/change")
async def direct_change(req: ChangeRequest):
    """Direct bi-temporal change detection & description endpoint."""
    mgr = get_manager()
    return mgr.run_change_analysis(req.image1_path, req.image2_path, req.question)


@router.post("/fusion")
async def direct_fusion(req: FusionRequest):
    """Direct optical-SAR joint fusion endpoint."""
    mgr = get_manager()
    return mgr.run_fusion_analysis(req.optical_path, req.sar_path, req.question)
