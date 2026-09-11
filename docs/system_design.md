# SatQuery AI — System Design Document (SDD)
**Project Code:** SIH26167  
**Organization:** Indian Space Research Organisation (ISRO) — Department of Space  
**Theme:** Space Technology / AI for Earth Observation  
**System:** Multi-Modal Remote Sensing Vision-Language System with Agentic Tool-Use Orchestrator  

---

## 1. Executive Overview & Problem Context

The ISRO SIH26167 challenge demands a robust, production-grade AI system capable of:
1. Understanding complex natural-language queries from Earth Observation (EO) scientists.
2. Operating on heterogeneous remote sensing imagery: Single optical (multispectral), Synthetic Aperture Radar (SAR), optical–SAR coregistered pairs, and bi-temporal image pairs.
3. Overcoming severe cross-sensor domain gaps between global training satellites (Sentinel-1/2) and ISRO evaluation satellites (**Cartosat-2S** and **RISAT-1/2**).
4. Executing multiple specialized vision tasks: Visual Question Answering (VQA), Text-Guided Visual Grounding, Bi-temporal Change Detection (pixel heatmaps), and Cross-Modal Sensor Fusion.
5. Providing **calibrated confidence estimates** and an **auditable execution trace** governed by an **Agentic Controller**.

SatQuery AI solves this with a **decoupled, modular architecture**: a **Claude-powered Agentic Controller** acting as the mission strategist, driving a **locally hosted PyTorch multi-task perception engine (`SatQueryUnified`)** backed by a **5-stage sensor normalizer**.

---

## 2. Architectural Principles & Requirements Compliance

| Architectural Requirement | Design Solution | Module Path |
|---|---|---|
| **Multi-Modal Satellite Ingestion** | GeoTIFF/TIFF ingestion with CRS, band, and spatial resolution validation. PNG/JPEG benchmark fallback. | `backend/preprocessing/geotiff_handler.py` |
| **Cross-Sensor Domain Adaptation** | 5-stage pipeline: Channel mapping, Dark Object Subtraction (DOS), Refined Lee speckle filter, Fourier Domain Adaptation (FDA). | `backend/preprocessing/sensor_normalizer.py` |
| **Shared Foundation Representation** | **RemoteCLIP ViT-L/14** backbone pre-trained on 5,000,000 remote sensing image-text pairs (RS5M). | `training/models/satquery_unified.py` |
| **Specialist VQA Tool** | Linear projection + 2-layer cross-attention bridge into **Flan-T5-base** language decoder. | `training/models/heads/vqa_head.py` |
| **Specialist Grounding Tool** | 100 learned object queries + 3-layer transformer decoder + MLP bounding box prediction. | `training/models/heads/grounding_head.py` |
| **Specialist Change Tool** | Siamese difference module + 4-stage bilinear U-Net generating pixel change probability map. | `training/models/heads/change_head.py` |
| **Specialist Fusion Tool** | Bidirectional cross-attention layers allowing optical and SAR tokens to condition on each other. | `training/models/heads/fusion_head.py` |
| **Agentic Controller** | Claude 3.5 Sonnet executing against a strictly typed tool registry with JSON execution audit traces. | `backend/controller/agent.py` |
| **Verifiable Confidence** | Composite log-prob uncertainty + boundary distance metrics aggregated via weighted harmonic mean. | `backend/confidence/unified_confidence_engine.py` |
| **Geospatial Web UI** | Next.js 14 App Router + Leaflet dual-map canvas with swipe slider, bounding box overlays, and PDF export. | `frontend/app/` |

---

## 3. Detailed Component Decomposition

### 3.1 Presentation Tier (`frontend/`)
- **Technology Stack:** Next.js 14 (App Router), React 18, Tailwind CSS, Lucide Icons, Leaflet / OpenLayers.
- **Key Modules:**
  - `UploadZone.tsx`: Handles multi-band GeoTIFF file drag-and-drop; calls `/api/upload` and renders compatibility reports.
  - `MapCanvas.tsx`: Leaflet viewer rendering true-color RGB tiles, interactive bounding box overlays (GeoJSON), and transparent Jet/Thermal change detection masks.
  - `ChatInterface.tsx`: Real-time streaming conversation feed displaying user queries, assistant synthesis, and confidence badges (🟢, 🟡, 🔴).
  - `ExecutionTrace.tsx`: Expandable step-by-step accordion showing tool invocations, parameters passed, execution timing, and intermediate outputs.
  - `ReportExport.tsx`: Generates a publication-grade PDF report with satellite metadata, map thumbnails, detected regions, and mission summaries.

### 3.2 Agentic Controller Tier (`backend/controller/`)
- **Technology Stack:** Anthropic Python SDK (Claude 3.5 Sonnet), Pydantic v2, Python 3.10+.
- **Responsibilities:**
  - Parses free-form scientist prompts into operational intents.
  - Validates input availability (e.g. checks if optical+SAR pair exists before attempting fusion).
  - Selects and executes tools from the defined registry: `run_vqa`, `run_grounding`, `run_change`, `run_fusion`.
  - Enforces schema guardrails: The agent can only modify permitted parameters; raw weights or arbitrary code execution are blocked.
  - Synthesizes multi-tool outputs into a cohesive executive briefing.

### 3.3 Sensor Preprocessing & Domain Adaptation Tier (`backend/preprocessing/`)
- **Technology Stack:** Rasterio, GDAL, NumPy, SciPy.
- **5-Stage Pipeline (`SensorNormalizer`):**
  1. **Channel Harmonization:**
     - Optical: Extracts Red (B4), Green (B3), Blue (B2), and NIR (B8). Maps Cartosat-2S HRMX bands `[B, G, R, NIR]` to Sentinel-2 indices.
     - SAR: Harmonizes dual-polarization `[co_pol, cross_pol]`. For RISAT-1 Hybrid Pol (Stokes $S_0, S_1, S_2, S_3$):
       $$\text{pseudo\_VV} \approx 0.5 \cdot (S_0 + S_1)$$
       $$\text{pseudo\_VH} \approx 0.5 \cdot (S_0 - S_1) \cdot \frac{1 - m}{1 + m}$$
  2. **Radiometric Calibration:**
     - Cartosat-2S 11-bit Digital Numbers ($DN \in [0, 2047]$) converted to Surface Reflectance (BOA) via Dark Object Subtraction (DOS):
       $$\rho_\lambda = \frac{DN - DN_{\text{dark}}}{DN_{\text{max}} - DN_{\text{dark}}}$$
     - RISAT DN converted to calibrated backscatter $\sigma^0$ (dB), followed by a **$5 \times 5$ Refined Lee speckle filter** to preserve edges while suppressing multiplicative granular noise.
  3. **Spatial Scale Alignment:**
     - Anti-aliasing Gaussian downsampling ($\sigma = 1.5$) brings sub-meter Cartosat (0.65m PAN / 2.0m HRMX) and RISAT (1–3m) to the standard 10m Ground Sampling Distance (GSD) of the vision backbone.
  4. **Statistical & Fourier Adaptation:**
     - Cumulative Distribution Function (CDF) histogram matching against the Sentinel reference baseline.
     - Optional Fourier Domain Adaptation (FDA, $\beta = 0.05$) to swap the low-frequency spectral amplitude without altering spatial phase structures.
  5. **Standardization:**
     - Robust percentile clipping $[P_1, P_{99}]$ followed by normalization into $[0.0, 1.0]$ float32 tensors resized to $(3, 224, 224)$.

### 3.4 Deep Learning Inference Engine (`training/models/`)
- **Technology Stack:** PyTorch 2.2+, OpenCLIP, HuggingFace Transformers, Einops.
- **Model Topology (`SatQueryUnified`):**
  - **Shared Vision Backbone:** RemoteCLIP ViT-L/14 (304M params). Extracts 196 patch tokens + 1 CLS token ($197 \times 1024$).
  - **VQA Head:** Projects 1024-dim visual tokens to 768-dim T5 space via LayerNorm + Linear. Two cross-attention layers allow text tokens to attend to visual patches. Flan-T5-base autoregressively generates free-form text.
  - **Grounding Head:** 100 learned query embeddings cross-attend to visual tokens and text query embeddings. A 3-layer transformer decoder feeds a 3-layer MLP predicting bounding boxes $(c_x, c_y, w, h) \in [0, 1]$ and a classification logit.
  - **Change Head:** Takes bi-temporal token pairs $(f_1, f_2)$, applies a learned difference block ($\text{Conv2d}(2048 \rightarrow 512 \rightarrow 256)$), and upsamples through a 4-stage bilinear U-Net to produce a continuous spatial change probability map $[H, W] \in [0.0, 1.0]$.
  - **Fusion Head:** 2 bidirectional cross-attention layers with residual connections, allowing optical patch tokens to attend to SAR tokens and vice-versa, outputting fused multi-sensor tokens.

### 3.5 Unified Confidence Engine (`backend/confidence/`)
- **Formulations:**
  - **VQA Confidence:** Composite score combining geometric mean token probability and minimum token probability, scaled by learned temperature $T$:
    $$C_{\text{vqa}} = 0.7 \cdot \exp\left(\frac{1}{N} \sum_{i=1}^N \log p_i\right) + 0.3 \cdot \min_{i} p_i$$
  - **Grounding Confidence:** Sigmoid of the maximum text-aligned detection logit combined with spatial coordinate entropy:
    $$C_{\text{grd}} = 0.6 \cdot \sigma(\text{logit}_{\text{max}}) + 0.4 \cdot (1 - \text{box\_variance})$$
  - **Change Detection Confidence:** Distance from decision boundary ($0.5$) combined with ambiguity ratio (pixels falling in the uncertain range $[0.35, 0.65]$):
    $$C_{\text{chg}} = 0.6 \cdot (1 - \text{Ratio}_{\text{ambiguous}}) + 0.4 \cdot \text{MeanConfidence}_{\text{fg}}$$
  - **Multi-Task Aggregation:** Weighted harmonic mean heavily penalizing the least-reliable bottleneck component:
    $$C_{\text{unified}} = \frac{\sum w_i}{\sum \frac{w_i}{C_i + \epsilon}}$$
  - **Tiers:** HIGH (🟢 $\ge 0.80$), MEDIUM (🟡 $0.55 \le C < 0.80$), LOW (🔴 $< 0.55$).

---

## 4. REST API Endpoint Specifications

### 4.1 `POST /api/upload`
Uploads raw GeoTIFF or image files for inspection and caching.
- **Request:** `multipart/form-data` with `files: List[UploadFile]`
- **Response:**
```json
{
  "upload_id": "upl_982341",
  "files": [
    {
      "filename": "S2A_20241015.tif",
      "sensor_hint": "sentinel2",
      "bands": 12,
      "crs": "EPSG:32643",
      "resolution_m": 10.0,
      "width": 1024,
      "height": 1024,
      "thumbnail_url": "/static/thumbnails/upl_982341_0.png"
    }
  ],
  "compatibility": {
    "valid": true,
    "input_type": "single_optical",
    "recommended_tools": ["run_vqa", "run_grounding"]
  }
}
```

### 4.2 `POST /api/query` (Agentic Chat Endpoint)
Primary entry point for conversational agent interactions.
- **Request:**
```json
{
  "query": "Identify any newly constructed aircraft hangars near the runway between these two dates and assess the change.",
  "file_ids": ["upl_982341_0.tif", "upl_982341_1.tif"],
  "temperature": 0.2
}
```
- **Response:**
```json
{
  "response": "Analysis complete. A new 120m structure consistent with an aircraft hangar was constructed adjacent to Runway 09. Ground surface disturbance is confirmed.",
  "execution_trace": [
    {
      "step": 1,
      "tool": "run_change",
      "params": {"image1": "upl_982341_0.tif", "image2": "upl_982341_1.tif"},
      "duration_ms": 340,
      "status": "success",
      "output": {"change_detected": true, "change_ratio": 0.042}
    },
    {
      "step": 2,
      "tool": "run_grounding",
      "params": {"image": "upl_982341_1.tif", "query": "aircraft hangar"},
      "duration_ms": 210,
      "status": "success",
      "output": {"boxes": [[0.45, 0.62, 0.12, 0.08]], "score": 0.88}
    }
  ],
  "visual_artifacts": {
    "change_map_url": "/static/masks/change_upl_982341.png",
    "grounding_boxes": [{"cx": 0.45, "cy": 0.62, "w": 0.12, "h": 0.08, "label": "aircraft hangar"}]
  },
  "confidence": {
    "score": 0.86,
    "tier": "HIGH",
    "icon": "🟢",
    "bottleneck_task": "change_detection"
  }
}
```

### 4.3 `GET /api/report/{upload_id}`
Generates and downloads the official ISRO Mission Analysis PDF.
- **Response:** Binary stream (`application/pdf`) with formal header, metadata table, thumbnail, detection overlays, and signature block.

---

## 5. Hardware Sizing & Runtime Constraints

| Metric | Kaggle 2xT4 (Training) | Local RTX 4060 (Serving) |
|---|---|---|
| **Usable VRAM** | $2 \times 15.6 \text{ GB} = 31.2 \text{ GB}$ | $8.0 \text{ GB}$ |
| **Model Footprint** | ~6.3 GB per GPU (Stage 1) / ~11 GB (Stage 2) | **4.2 GB (All 4 heads loaded)** |
| **Inference Precision** | FP32 (Training stability) | FP16 / BF16 (Optimized TensorRT/PyTorch) |
| **Batch Size** | 8–16 (Distributed) | 1 (Interactive Real-time) |
| **Per-Query Latency** | N/A | **VQA: ~180ms \| Grounding: ~95ms \| Change: ~210ms** |
| **Safety Headroom** | ~4.5 GB free per GPU | **~3.8 GB free VRAM headroom** |
