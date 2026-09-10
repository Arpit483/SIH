# SatQuery AI — Technical Implementation & Deployment Plan
**Project Code:** SIH26167  
**System:** Multi-Modal Remote Sensing VLM with Agentic Orchestrator  
**Document:** End-to-End Execution, Training, Deployment & Demonstration Roadmap  

---

## 1. Execution Roadmap Overview

The implementation roadmap is structured into **six progressive phases**, ensuring zero downtime between cloud training, local weight deployment, and final Grand Finale demonstration.

```mermaid
gantt
    title SatQuery AI Implementation & Deployment Schedule
    dateFormat  YYYY-MM-DD
    section Phase 1: Cloud Training
    Stage 1: Head Warmup (Frozen Backbone)       :done,    p1_1, 2026-09-09, 2026-09-10
    Stage 2: Joint Multi-Task (Backbone Unfrozen):active,  p1_2, 2026-09-10, 2026-09-11
    Stage 3: Task Polish & Benchmark Evaluation  :         p1_3, 2026-09-11, 2026-09-11
    section Phase 2: Local Model Deployment
    Weight Export & Checkpoint Verification       :         p2_1, 2026-09-11, 2026-09-12
    RTX 4060 Local Inference Benchmark (<500ms)  :         p2_2, 2026-09-12, 2026-09-12
    section Phase 3: Backend & Preprocessing
    FastAPI Endpoints & ModelManager Integration  :         p3_1, 2026-09-12, 2026-09-13
    SensorNormalizer & GeoTIFF Handler Linking    :         p3_2, 2026-09-13, 2026-09-13
    Confidence Engine & Calibration Tuning        :         p3_3, 2026-09-13, 2026-09-14
    section Phase 4: Agentic Controller
    Claude 3.5 Sonnet Tool Invocations Binding    :         p4_1, 2026-09-14, 2026-09-14
    Auditable Execution Trace & Stream Logging    :         p4_2, 2026-09-14, 2026-09-15
    section Phase 5: Frontend Dashboard
    Next.js 14 Map Canvas & Bounding Box Overlays :         p5_1, 2026-09-15, 2026-09-15
    Swipe Slider for Bi-Temporal Change Detection :         p5_2, 2026-09-15, 2026-09-16
    PDF Mission Report Generation                 :         p5_3, 2026-09-16, 2026-09-16
    section Phase 6: Verification & Demo
    Benchmark Harness (RSVQA, VRSBench, CDVQA)   :         p6_1, 2026-09-16, 2026-09-17
    ISRO Cartosat-2S & RISAT Evaluation Rehearsal:         p6_2, 2026-09-17, 2026-09-17
```

---

## 2. Phase 1: Model Training Lifecycle (Kaggle 2xT4)

### Stage 1: Head Initialization (Backbone Frozen) — COMPLETED ✅
- **Objective:** Train all four task heads from random initialization without corrupting the 5M remote-sensing feature space of RemoteCLIP.
- **Data Pool:** 45,173 samples (AdaptLLM-RS instructions + RSICD aerial captions).
- **Hyperparameters:** `EPOCHS = 3`, `LR = 1.5e-4`, `GRAD_ACCUM = 4`, `batch_size = 16`, FP32 precision.
- **Outcome:** Loss stabilized from **13.33 down to 0.92**, outputting `satquery_stage1_best.pt`.

### Stage 2: Joint Multi-Task Alignment (Backbone Unfrozen) — IN PROGRESS 🔄
- **Objective:** Jointly fine-tune the vision encoder and task heads on multi-spectral satellite imagery to achieve domain adaptation.
- **Data Pool:** 55,173 samples including **10,000 real BigEarthNet Sentinel-2 GeoTIFF patches** with official land-cover metadata.
- **Differential Learning Rates:**
  - RemoteCLIP ViT-L/14 Backbone: `1e-5` (preserves generalized RS representations).
  - Four Task Heads: `5e-5` (accelerates task specialization).
- **Architecture Safeguards:**
  - Single-GPU execution on `cuda:0` with **Gradient Checkpointing** on all 24 ViT transformer blocks to eliminate `ReduceAddCoalesced` OOM crashes.
  - VRAM budget: **6.11 GB baseline / 10.8 GB peak** (well inside the 14.5 GB T4 ceiling).
- **Outcome:** Generates `satquery_stage2_best.pt`.

### Stage 3: Per-Task Polish & Benchmark Evaluation
- **Objective:** Final calibration on target benchmark distributions (RSVQA-HR, RSVQA-LR, VRSBench, CDVQA) and confidence temperature scaling.
- **Hyperparameters:** `EPOCHS = 3`, `LR = 2e-5`, `GRAD_ACCUM = 4`.
- **Outcome:** Exports `satquery_final.pt` and `eval_report.json`.

---

## 3. Phase 2: Checkpoint Export & Local Deployment

### 3.1 Local RTX 4060 Verification
- **Target Machine:** Windows 11 Workstation, NVIDIA GeForce RTX 4060 (8 GB GDDR6 VRAM), CUDA 12.x.
- **Model Size:** 747M parameters (~2.4 GB uncompressed PyTorch state dict).
- **Memory Footprint:** **~4.2 GB VRAM** in FP16 inference mode with all four heads concurrently active, leaving ~3.8 GB headroom for raster rendering.
- **Deployment Script (`scripts/verify_local_gpu.py`):**
  1. Instantiates `SatQueryUnified(pretrained=None)`.
  2. Loads `satquery_final.pt` with `map_location='cuda:0'`.
  3. Executes dry-run forward passes across all four heads with dummy $(3, 224, 224)$ tensors.
  4. Verifies latency SLA:
     - `run_vqa`: $< 250\text{ ms}$
     - `run_grounding`: $< 120\text{ ms}$
     - `run_change`: $< 280\text{ ms}$
     - `run_fusion`: $< 310\text{ ms}$

---

## 4. Phase 3: Backend Serving Architecture (`backend/`)

### 4.1 Model Manager (`backend/model_manager.py`)
- Acts as a thread-safe singleton managing model loading, GPU warmup, inference batching, and VRAM garbage collection.
- Implements hot-reload capability so checkpoints can be updated without restarting the FastAPI server.

### 4.2 Preprocessing Integration (`backend/preprocessing/`)
- Intercepts all incoming files from `/api/upload` and `/api/query`.
- Runs `CompatibilityChecker.check(file_paths)`:
  - If a single optical image is detected $\rightarrow$ routes to `SensorNormalizer` for DOS radiometric calibration and Gaussian downsampling.
  - If a single SAR raster is detected $\rightarrow$ applies sigma-naught decibel conversion and Refined Lee $5 \times 5$ speckle filter.
  - If two images are uploaded with different acquisition dates $\rightarrow$ validates CRS and spatial overlap for the Siamese Change Head.
  - If co-registered optical and SAR images are uploaded $\rightarrow$ confirms coordinate alignment for the Fusion Head.

### 4.3 Confidence Scoring Integration (`backend/confidence/`)
- Calls `UnifiedConfidenceEngine`:
  - Collects logits and token probabilities from each active head.
  - Computes task uncertainty metrics and temperature calibration.
  - Aggregates the multi-task composite score via weighted harmonic mean.
  - Returns numeric score, quality tier (`HIGH`, `MEDIUM`, `LOW`), and bottleneck diagnostic.

---

## 5. Phase 4: Agentic Controller Integration (`backend/controller/`)

### 5.1 System Prompt Engineering
The agent operates under a specialized remote sensing system prompt instructing it to:
1. Act as an expert ISRO Earth Observation intelligence analyst.
2. Formulate explicit hypotheses before calling tools.
3. Select only tools supported by the uploaded imagery's modality and temporal characteristics.
4. Interpret confidence scores and alert the user if atmospheric conditions, clouds, or sensor gaps degrade certainty.

### 5.2 Tool Registry Definition
```python
TOOL_REGISTRY = [
    {
        "name": "run_vqa",
        "description": "Answer visual questions regarding land cover, structures, or geography in an optical or SAR satellite image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "question": {"type": "string"}
            },
            "required": ["image_id", "question"]
        }
    },
    {
        "name": "run_grounding",
        "description": "Detect and return normalized bounding boxes for specified target objects in a satellite scene.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "target_phrase": {"type": "string"}
            },
            "required": ["image_id", "target_phrase"]
        }
    },
    {
        "name": "run_change",
        "description": "Compute pixel-level change detection heatmaps and generate textual change summaries for bi-temporal image pairs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pre_image_id": {"type": "string"},
                "post_image_id": {"type": "string"},
                "question": {"type": "string"}
            },
            "required": ["pre_image_id", "post_image_id"]
        }
    },
    {
        "name": "run_fusion",
        "description": "Perform joint optical-SAR cross-modal analysis on co-registered multi-sensor satellite pairs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "optical_image_id": {"type": "string"},
                "sar_image_id": {"type": "string"},
                "question": {"type": "string"}
            },
            "required": ["optical_image_id", "sar_image_id", "question"]
        }
    }
]
```

---

## 6. Phase 5: Geospatial Web Dashboard (`frontend/`)

### 6.1 User Interface Workflow
1. **Upload & Auto-Diagnosis:**
   - User drags and drops GeoTIFF files into `UploadZone`.
   - The UI displays detected sensor types (e.g. `Sentinel-2 L2A`, `Cartosat-2S HRMX`), CRS (`EPSG:32643`), resolution ($10\text{ m}$), and acquisition timestamps.
2. **Interactive Dual-Map Canvas (`components/MapCanvas.tsx`):**
   - Renders imagery using Leaflet / React-Leaflet.
   - For grounding: Renders vector SVG bounding boxes with class tags and individual confidence percentages.
   - For change detection: Provides an interactive before/after split swipe slider and an adjustable alpha-blended change heatmap overlay.
3. **Conversational Feed & Trace Accordion (`components/ChatInterface.tsx`):**
   - Natural language interaction with streaming markdown.
   - Embedded audit accordion reveals every tool execution step, parameter, latency, and status.
4. **Mission Report Generation (`components/ReportExport.tsx`):**
   - One-click export compiling findings into an official PDF document containing satellite metadata, detected objects, change statistics, and verification signatures.

---

## 7. Phase 6: Benchmark Evaluation & Demonstration Protocol

### 7.1 Automated Evaluation Harness (`training/evaluate.py`)
The system is evaluated against four benchmark suites to quantify performance against baseline models:
- **RSVQA-HR & RSVQA-LR:** Evaluates VQA accuracy and BLEU/CIDEr scores on high/low-resolution imagery.
- **VRSBench:** Evaluates visual grounding (IoU@0.5 and mean Average Precision) and captioning quality.
- **CDVQA:** Evaluates bi-temporal change question answering accuracy and spatial change map F1 / mIoU.

### 7.2 Grand Finale Live Demonstration Scenarios
1. **Scenario 1: Single Optical Scene VQA & Grounding**
   - Upload: Cartosat-2S scene of an industrial complex.
   - Query: *"Locate all fuel storage tanks and verify if any construction has begun nearby."*
   - Execution: Agent runs `run_grounding("fuel storage tank")` $\rightarrow$ displays bounding boxes on map $\rightarrow$ runs `run_vqa` to confirm ground conditions.
2. **Scenario 2: Bi-Temporal Disaster & Flood Impact Assessment**
   - Upload: Pre-flood (T1) and post-flood (T2) Sentinel-2 / Cartosat pairs.
   - Query: *"Map all submerged agricultural land and quantify the spatial extent of the flood."*
   - Execution: Agent executes `run_change` $\rightarrow$ renders jet-colormap flood mask $\rightarrow$ computes flooded area percentage.
3. **Scenario 3: Optical–SAR Cross-Modal All-Weather Intelligence**
   - Upload: Cloud-covered optical image + Sentinel-1 / RISAT-1 C-band SAR patch.
   - Query: *"Confirm whether naval vessels are docked at the pier despite cloud obstruction."*
   - Execution: Agent executes `run_fusion` $\rightarrow$ cross-attention head penetrates cloud cover using SAR radar backscatter $\rightarrow$ returns confirmed dockings with high confidence.
