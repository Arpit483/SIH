# SatQuery AI — Unified Agentic Remote-Sensing Intelligence Platform

> **ISRO Earth Observation Scientist Suite (Smart India Hackathon SIH26167)**  
> Autonomous Vision-Language Reasoning, Multimodal Grounding, Bi-Temporal Change Quantification, and Cross-Modal SAR-Optical Fusion.

---

## 🛰️ Unified Architecture

SatQuery AI merges an agentic Next.js frontend with an accelerated FastAPI/PyTorch ML backend pipeline into a single, cohesive full-stack application:

```
SatQuery/
├── backend/                  # FastAPI REST Service & ML Orchestration
│   ├── controller/agent.py   # Autonomous Tool Calling & Planning Engine
│   ├── model_manager.py      # Multi-task Dispatcher (CUDA / Apple Silicon MPS / CPU)
│   ├── preprocessing/        # GeoTIFF loader, CRS coregistration, sensor normalizer
│   ├── confidence/           # Calibrated uncertainty & task confidence engine
│   ├── routes/               # /api/upload, /api/query, /api/generate-report
│   └── requirements.txt      # Python dependencies
├── training/                 # Model Architecture & Multi-task Heads
│   ├── models/satquery_unified.py # Shared ViT Backbone + 4 Specialist Heads
│   └── configs/              # Unified config & hyperparameters
├── src/                      # Next.js 14 Dashboard (Mission-Control UI)
│   ├── app/                  # App Router pages & layout
│   ├── components/           # UploadZone, Leaflet MapCanvas, ChatPanel, Reports
│   ├── constants/            # Mock scenarios & shared theme tokens
│   └── lib/api.ts            # Client API bridge & live/offline fallback
├── scripts/
│   └── run_mac_m4.sh         # One-click startup script for Mac M4 (Apple Silicon)
├── Dockerfile                # Multi-stage container build
├── docker-compose.yml        # Multi-service stack definition
└── package.json              # Unified project lifecycle scripts
```

---

## ⚡ Deployment on Mac M4 (Apple Silicon)

The project includes native hardware acceleration for Apple Silicon (M1/M2/M3/M4) via PyTorch's Metal Performance Shaders (`mps` device):

### Option 1: One-Click Startup Script (Recommended)
```bash
# Run the automated setup and startup script
./scripts/run_mac_m4.sh
```
This script:
1. Creates/activates a Python virtual environment (`.venv`)
2. Installs backend dependencies with Apple Silicon MPS support
3. Installs frontend npm packages
4. Launches the FastAPI backend on `http://localhost:8000`
5. Launches the Next.js Dashboard on `http://localhost:3001`

### Option 2: Running Services Concurrently via NPM
```bash
# Start frontend only (with intelligent offline mock engine)
npm run dev

# Start backend only
npm run dev:backend

# Start unified deployment script
npm run dev:all
```

### Option 3: Docker Deployment
```bash
docker compose up --build
```

---

## 📡 REST API Specification

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Service status, active compute device (`mps`/`cuda`/`cpu`), capabilities |
| `/api/upload` | `POST` | Ingests GeoTIFF/TIFF/PNG rasters, validates CRS, returns thumbnails & report |
| `/api/query` | `POST` | Agentic endpoint: classifies intent, executes tool pipeline, returns trace & bboxes |
| `/api/generate-report` | `POST` | Generates official ISRO-formatted audit report PDF |
| `/api/vqa` | `POST` | Direct single-image VQA & captioning |
| `/api/grounding` | `POST` | Text-guided region grounding with bounding boxes |
| `/api/change` | `POST` | Bi-temporal differential change detection & description |
| `/api/fusion` | `POST` | Cross-modal Optical + SAR all-weather joint analysis |

---

## 🎨 Frontend Features (Odysseus Mission-Control Aesthetic)
- **Deep Charcoal Surface Layering**: `#0d0f12` – `#15181f` with subtle 1px borders and coral (`#e07856`) accents.
- **Dynamic Leaflet Canvas**: Zero-shot bounding boxes with confidence tooltips, draggable swipe comparator slider, and toggleable flood/SAR heatmaps.
- **Instrument Navigation**: Left sidebar with live telemetry states and 1-click test datasets.
- **Odysseus Composer Bar**: Multi-tool toggle, model selector pill, and raw parameter inspection.
- **Live / Offline Bridge**: Connects to the FastAPI backend when live, and falls back to simulated reasoning if the backend is initializing.
