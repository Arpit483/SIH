"""
routes/report.py: Downloadable PDF Report Generator for SatQuery AI.
Required deliverable for ISRO SIH26167.
"""

import os
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
router = APIRouter(prefix="/api", tags=["report"])

REPORTS_DIR = PROJECT_ROOT / "backend" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class ReportRequest(BaseModel):
    query: str
    answer: str
    confidence: float
    tier: str
    selected_task: str
    model_name: str
    input_type: str
    sensors: list
    execution_trace: dict

@router.post("/generate-report")
async def generate_pdf_report(req: ReportRequest):
    """Generates an official ISRO / SatQuery audit report as a PDF."""
    report_id = req.execution_trace.get("trace_id", "report")
    pdf_filename = f"SatQuery_Report_{report_id[:8]}.pdf"
    pdf_path = REPORTS_DIR / pdf_filename

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        elements = []

        # Header Title
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f2b5c")
        )
        elements.append(Paragraph("SatQuery AI — Multimodal Remote Sensing Analysis Report", title_style))
        elements.append(Paragraph("<b>Organized by:</b> Indian Space Research Organisation (ISRO) | SIH26167", styles["Normal"]))
        elements.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", styles["Normal"]))
        elements.append(Spacer(1, 15))

        # Executive Summary Box
        elements.append(Paragraph("<b>1. Natural Language Query & Findings</b>", styles["Heading2"]))
        elements.append(Paragraph(f"<b>Query:</b> <i>\"{req.query}\"</i>", styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"<b>Answer:</b> {req.answer}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Confidence & Task Metrics
        elements.append(Paragraph("<b>2. Model & Confidence Metrics</b>", styles["Heading2"]))
        metric_data = [
            ["Metric", "Value"],
            ["Selected Task", req.selected_task.upper()],
            ["Orchestrated Model", req.model_name],
            ["Confidence Score", f"{req.confidence * 100:.1f}%"],
            ["Confidence Tier", req.tier],
            ["Input Type", req.input_type],
            ["Detected Sensors", ", ".join(str(s) for s in req.sensors)]
        ]
        t = Table(metric_data, colWidths=[200, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))

        # Auditable Execution Trace
        elements.append(Paragraph("<b>3. Auditable Execution Trace (ISRO Evaluation Criteria)</b>", styles["Heading2"]))
        trace_data = [
            ["Parameter", "Details"],
            ["Trace ID", req.execution_trace.get("trace_id", "N/A")],
            ["Tool Invoked", req.execution_trace.get("selected_tool", "N/A")],
            ["Permitted Params", str(req.execution_trace.get("permitted_parameters", {}))],
            ["Latency", f"{req.execution_trace.get('latency_seconds', 0.0)} s"],
            ["Orchestrator Mode", req.execution_trace.get("orchestrator", "Agentic")]
        ]
        t_trace = Table(trace_data, colWidths=[200, 300])
        t_trace.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#374151")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t_trace)

        doc.build(elements)
    except Exception as e:
        # Fallback text report if reportlab is not installed
        with open(pdf_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
            f.write(f"SatQuery AI Report\nQuery: {req.query}\nAnswer: {req.answer}\nConfidence: {req.confidence}")
        return {"status": "success", "file_url": f"/reports/{pdf_path.stem}.txt"}

    return {
        "status": "success",
        "file_url": f"/reports/{pdf_filename}",
        "local_path": str(pdf_path)
    }

@router.get("/download/{filename}")
async def download_report(filename: str):
    file_path = REPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(file_path), media_type="application/pdf", filename=filename)
