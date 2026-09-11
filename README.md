# SatQuery AI — Agentic Remote-Sensing Intelligence Dashboard

> **ISRO Earth Observation Scientist Suite (Smart India Hackathon SIH26167)**  
> Autonomous Vision-Language Reasoning, Multimodal Grounding, and Bi-Temporal Change Quantification for Earth Observation.

---

## 🛰️ Overview
**SatQuery AI** is an agentic remote-sensing analysis tool engineered with a dark, mission-control aesthetic tailored for satellite operations centers. It provides a three-panel interactive interface for satellite raster ingestion, interactive geospatial canvas manipulation, zero-shot entity grounding, bi-temporal differential change detection, and SAR-optical cross-modal fusion under dense cloud cover.

---

## ⚡ Tech Stack
- **Framework**: Next.js 14 (App Router) with TypeScript
- **Styling**: Tailwind CSS with custom mission-control design system (Electric Cyan `#00f0ff`, Amber `#f59e0b`, Deep Space `#070b14`)
- **Geospatial Canvas**: Leaflet & React-Leaflet with Dark Matter tile rendering & vector overlay engine
- **Icons**: Lucide React
- **Data Visualizations**: Recharts for spatial distribution & damage area breakdown
- **Telemetry & Latency**: Simulated token streaming and async tool execution trace (300ms–800ms)

---

## 🚀 Key Features & 3-Panel Architecture

### 1. Left Ingestion Panel (`UploadZone`)
- **Drag & Drop Multi-band Raster Ingestion**: Accepts GeoTIFF, TIFF, PNG, and JPEG imagery.
- **Auto-Detected Metadata Header**: Extracts sensor hint, bands count, CRS (`EPSG:32643` / `EPSG:32644` / `EPSG:32646`), GSD resolution, and raster dimensions.
- **Dynamic Configuration Badging**: Automatically identifies input topology:
  - `Single Optical Scene` (Monoscopic VNIR)
  - `Single SAR Scene` (C-Band Radar)
  - `Cross-Modal Pair` (Optical + SAR All-Weather)
  - `Bi-Temporal Pair` (T1 Pre / T2 Post Change)
- **1-Click Test Dataset Loaders**: Instant switching between port surveillance, flood inundation, and SAR cloud penetration.

### 2. Center Geospatial Canvas (`MapCanvas`)
- **Dark Basemap Leaflet Engine**: High-contrast, instrument-like cartographic viewport.
- **Zero-Shot Grounding Bounding Boxes**: Rendered vector bounding boxes with interactive tooltips, confidence scores, and physical attributes (heading, length, draft, thermal delta, radar echo).
- **Interactive Swipe Split Comparator**: Draggable before/after divider for direct pixel-by-pixel change inspection between T1 baseline and T2 event scenes.
- **Toggleable Inundation & SAR Heatmap Overlays**: Real-time visualization of flood boundaries and radar penetration hotspots.
- **Nadir Sensor Reticle HUD**: Live latitude, longitude, zoom factor, and Ground Sampling Distance (GSD) tracker.

### 3. Right Conversational Reasoning Feed (`ChatInterface` & `ReportExport`)
- **Simulated Token Streaming**: Dynamic word-by-word streaming text reveal for assistant reasoning.
- **Tiered Confidence Badges**: `CONFIDENCE: HIGH` (Cyan), `CONFIDENCE: MED` (Amber), `CONFIDENCE: LOW` (Rose) driven by shared constants.
- **Agentic Execution Trace Accordion**: Expandable step-by-step pipeline trace displaying tool names, input JSON payloads, execution durations, and returned telemetry.
- **Preset Mission Query Chips**: Fast 1-click execution of complex EO scientific queries.
- **Mission Briefing Report View**: Generates an ISRO-formatted mission verification brief with metadata tables, high-resolution thumbnail previews, spatial distribution charts, grounded entity registries, and a mocked PDF download affordance.

---

## 🔬 Three Core Demo Scenarios
1. **Single Optical Grounding (Visakhapatnam Port)**:
   - *Query*: *"Detect and isolate all maritime vessels, cargo berths, and fuel storage tanks in the port area."*
   - *Outputs*: 8 grounded maritime & infrastructure entities, 96.4% confidence, tool execution trace.
2. **Bi-Temporal Flood Inundation (Kaziranga / Brahmaputra)**:
   - *Query*: *"Calculate total flood inundation area, compare pre/post flood boundaries, and quantify agricultural submergence."*
   - *Outputs*: 142.84 km² inundation area, 59.2% crop submergence, swipe comparator, flood heatmap layer.
3. **Cross-Modal Optical + SAR All-Weather Fusion (Mumbai Offshore)**:
   - *Query*: *"Penetrate monsoon cloud cover using SAR cross-modal fusion to detect hidden maritime vessels and slick anomalies."*
   - *Outputs*: 96.4% cloud penetration, 6 vessels identified through cloud cover via radar double-bounce echoes, low-backscatter slick anomaly.

---

## 🛠️ Getting Started

```bash
# 1. Install dependencies
npm install

# 2. Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to explore the dashboard.
