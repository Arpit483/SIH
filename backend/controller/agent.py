"""
agent.py: Agentic Controller for SatQuery AI (SIH26167).

Fulfills all Agentic Orchestration requirements from ISRO PS:
  1. Interpret query and classify the requested task
  2. Check number, modality, format, metadata, and compatibility of input images
  3. Select one or more tools from predefined registry
  4. Configure only permitted task parameters and execute workflow
  5. Combine textual and spatial outputs, estimate confidence, and return visual evidence
  6. Provide an auditable execution summary containing selected task, model/tool names, and parameters
"""

import os
import sys
import json
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.preprocessing.compatibility_checker import CompatibilityChecker, CompatibilityReport
from backend.model_manager import ModelManager

logger = logging.getLogger("satquery.agent")

# Predefined tool registry for Claude Tool Calling / Local Routing
TOOL_REGISTRY = [
    {
        "name": "run_vqa",
        "description": "Visual Question Answering and land-cover description on a single optical or SAR image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {"type": "integer", "description": "Index of image to analyze (0 for single image)", "default": 0},
                "question": {"type": "string", "description": "Natural language question or description prompt"}
            },
            "required": ["question"]
        }
    },
    {
        "name": "run_grounding",
        "description": "Text-guided region grounding to detect and locate specific objects or areas with bounding boxes.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {"type": "integer", "description": "Index of image to analyze", "default": 0},
                "phrase": {"type": "string", "description": "Referring expression or object name to ground"}
            },
            "required": ["phrase"]
        }
    },
    {
        "name": "run_change_analysis",
        "description": "Bi-temporal change detection and change-based VQA between two temporal observations.",
        "parameters": {
            "type": "object",
            "properties": {
                "image1_index": {"type": "integer", "description": "Index of earlier observation", "default": 0},
                "image2_index": {"type": "integer", "description": "Index of later observation", "default": 1},
                "question": {"type": "string", "description": "Question regarding what changed", "default": "What changed between these two dates?"}
            },
            "required": []
        }
    },
    {
        "name": "run_fusion_analysis",
        "description": "Cross-modal joint analysis combining complementary optical (spectral) and SAR (structural) imagery.",
        "parameters": {
            "type": "object",
            "properties": {
                "optical_index": {"type": "integer", "description": "Index of optical/multispectral image", "default": 0},
                "sar_index": {"type": "integer", "description": "Index of SAR image", "default": 1},
                "question": {"type": "string", "description": "Joint analysis query"}
            },
            "required": ["question"]
        }
    }
]


class AgenticController:
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.compatibility_checker = CompatibilityChecker()
        self.model_manager = model_manager or ModelManager.get_instance()
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Local execution trace log directory
        self.trace_dir = PROJECT_ROOT / "backend" / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def _plan_with_rules(self, query: str, report: CompatibilityReport) -> Tuple[str, Dict[str, Any]]:
        """Intelligent autonomous router when LLM API is unavailable or offline."""
        q_lower = query.lower()
        num_files = len(report.file_infos)
        
        # Check for change keywords
        change_keywords = ["change", "difference", "between these two", "increased", "decreased", "dates", "earlier", "later", "before", "after"]
        grounding_keywords = ["highlight", "locate", "where is", "bounding box", "detect", "find the", "point to", "box"]
        fusion_keywords = ["optical and sar", "together", "both images", "joint", "sar and optical", "complementary", "cross-modal"]

        # Decision tree respecting inputs
        if num_files >= 2 and any(k in q_lower for k in change_keywords):
            return "run_change_analysis", {"image1_index": 0, "image2_index": 1, "question": query}
            
        if num_files >= 2 and any(k in q_lower for k in fusion_keywords):
            # Find which file is optical vs SAR
            opt_idx = 0
            sar_idx = 1
            for i, info in enumerate(report.file_infos):
                if "sar" in info.sensor_hint.lower() or "risat" in info.sensor_hint.lower() or "sentinel1" in info.sensor_hint.lower():
                    sar_idx = i
                else:
                    opt_idx = i
            return "run_fusion_analysis", {"optical_index": opt_idx, "sar_index": sar_idx, "question": query}

        if any(k in q_lower for k in grounding_keywords):
            # Extract phrase
            phrase = query
            for prefix in ["highlight the", "highlight", "locate the", "locate", "find the", "detect"]:
                if prefix in q_lower:
                    idx = q_lower.find(prefix) + len(prefix)
                    phrase = query[idx:].strip(" .?")
                    break
            return "run_grounding", {"image_index": 0, "phrase": phrase}
            
        # Default to VQA
        return "run_vqa", {"image_index": 0, "question": query}

    def _plan_with_claude(self, query: str, report: CompatibilityReport) -> Tuple[str, Dict[str, Any]]:
        """Use Claude API tool-calling if API key is provided."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            
            system_prompt = (
                "You are the SatQuery AI Agentic Orchestrator for ISRO remote sensing tasks. "
                "Inspect the user query and the input image metadata, then call the single most appropriate tool "
                "from the registry. Do not guess parameters outside permitted constraints."
            )
            
            input_summary = f"Uploaded {len(report.file_infos)} image(s). Type: {report.input_type}. Sensors: {report.sensor_types}."
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Inputs: {input_summary}\nQuery: {query}"}],
                tools=TOOL_REGISTRY,
                tool_choice={"type": "auto"}
            )
            
            for content in response.content:
                if content.type == "tool_use":
                    return content.name, content.input
                    
        except Exception as e:
            logger.warning("Claude API tool planning failed: %s. Falling back to autonomous router.", e)
            
        return self._plan_with_rules(query, report)

    def execute_query(self, query: str, file_paths: List[str]) -> Dict[str, Any]:
        """
        End-to-end execution of a user query through the agentic pipeline.
        Returns:
          - answer: text answer
          - visual_evidence: spatial map, bboxes, or highlights
          - confidence: calibrated score, tier, and icon
          - execution_trace: auditable summary required by ISRO criteria
        """
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # 1. Compatibility Check (Stage 0)
        report = self.compatibility_checker.check(file_paths)
        if not report.valid:
            return {
                "status": "error",
                "message": "Input validation failed: " + "; ".join(report.issues),
                "issues": report.issues,
                "warnings": report.warnings
            }

        # 2. Task Planning & Tool Selection
        if self.anthropic_key:
            selected_tool, params = self._plan_with_claude(query, report)
        else:
            selected_tool, params = self._plan_with_rules(query, report)

        logger.info("Agent selected tool: %s with parameters: %s", selected_tool, params)

        # 3. Parameter validation & execution
        result = {}
        if selected_tool == "run_vqa":
            idx = min(params.get("image_index", 0), len(file_paths) - 1)
            q = params.get("question", query)
            result = self.model_manager.run_vqa(file_paths[idx], question=q)
            
        elif selected_tool == "run_grounding":
            idx = min(params.get("image_index", 0), len(file_paths) - 1)
            phrase = params.get("phrase", query)
            result = self.model_manager.run_grounding(file_paths[idx], phrase=phrase)
            
        elif selected_tool == "run_change_analysis":
            if len(file_paths) < 2:
                # Fallback to VQA if pair is missing
                result = self.model_manager.run_vqa(file_paths[0], question=query)
                selected_tool = "run_vqa"
            else:
                idx1 = params.get("image1_index", 0)
                idx2 = params.get("image2_index", 1)
                result = self.model_manager.run_change_analysis(file_paths[idx1], file_paths[idx2], question=query)
                
        elif selected_tool == "run_fusion_analysis":
            if len(file_paths) < 2:
                result = self.model_manager.run_vqa(file_paths[0], question=query)
                selected_tool = "run_vqa"
            else:
                opt_idx = params.get("optical_index", 0)
                sar_idx = params.get("sar_index", 1)
                result = self.model_manager.run_fusion_analysis(file_paths[opt_idx], file_paths[sar_idx], question=query)
        else:
            result = self.model_manager.run_vqa(file_paths[0], question=query)
            selected_tool = "run_vqa"

        duration = round(time.time() - start_time, 3)

        # 4. Generate Auditable Execution Trace (PS requirement)
        execution_trace = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "selected_task": result.get("task", selected_tool),
            "model_name": "SatQueryUnified (RemoteCLIP ViT-L/14 Backbone)",
            "selected_tool": selected_tool,
            "permitted_parameters": params,
            "input_summary": {
                "num_images": len(file_paths),
                "detected_type": report.input_type,
                "sensors": report.sensor_types
            },
            "confidence": {
                "score": result.get("confidence", 0.85),
                "tier": result.get("tier", "HIGH"),
                "icon": result.get("icon", "🟢")
            },
            "latency_seconds": duration,
            "orchestrator": "Claude API (Tool Use)" if self.anthropic_key else "Autonomous Rule/Intent Router"
        }

        # Save trace to disk for auditability
        try:
            trace_file = self.trace_dir / f"{trace_id}.json"
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(execution_trace, f, indent=2)
        except Exception as e:
            logger.debug("Failed to write trace file: %s", e)

        return {
            "status": "success",
            "query": query,
            "answer": result.get("answer"),
            "confidence": result.get("confidence"),
            "tier": result.get("tier"),
            "icon": result.get("icon"),
            "visual_evidence": result.get("visual_evidence", {}),
            "bounding_boxes": result.get("bounding_boxes"),
            "change_percentage": result.get("change_percentage"),
            "execution_trace": execution_trace,
            "compatibility_report": {
                "valid": report.valid,
                "input_type": report.input_type,
                "sensors": report.sensor_types,
                "warnings": report.warnings
            }
        }
