# SatQuery AI — Architecture Blueprints & Diagrams
**SIH26167 | ISRO Space Technology Problem Statement**
**System: Multimodal Remote Sensing VLM with Agentic Orchestrator & Sensor Adaptation**

---

## 1. High-Level System Architecture

The end-to-end system separates the **User Presentation Layer**, the **Agentic Orchestrator (Claude 3.5 Sonnet)**, the **FastAPI Inference Gateway**, and the **Core Deep Learning Vision-Language Model (`SatQueryUnified`)**.

```mermaid
graph TB
    subgraph Client ["Client Layer (Web Application)"]
        UI["Next.js 14 Dashboard"]
        MAP["Leaflet / OpenLayers Geospatial Canvas"]
        CHAT["Conversational & Trace Feed"]
        PDF_BTN["ISRO Report Generator (.pdf)"]
    end

    subgraph Orchestration ["Agentic Controller Layer (Autonomous Reasoning)"]
        AGENT["Claude 3.5 Sonnet Agentic Controller"]
        MEM["Conversation Context & Execution Trace"]
        TOOL_REG["Validated Tool Registry<br/>(run_vqa, run_grounding, run_change, run_fusion)"]
    end

    subgraph Gateway ["API & Processing Gateway (FastAPI)"]
        ROUTER["FastAPI Router & Schema Validator"]
        PREPROC["Sensor Normalizer & GeoTIFF Handler"]
        CONF_ENG["Unified Confidence Engine"]
        AUDIT["Auditable Execution Logger"]
    end

    subgraph DeepLearning ["Core Perception Engine (PyTorch / CUDA)"]
        BACKBONE["RemoteCLIP ViT-L/14 Backbone<br/>(Pretrained on 5M Remote Sensing Pairs)"]
        HEAD_VQA["VQA Head<br/>(Flan-T5-base Decoder + Cross-Attention)"]
        HEAD_GRD["Grounding Head<br/>(100 Learned DETR Object Queries)"]
        HEAD_CHG["Change Head<br/>(Siamese Diff + U-Net Spatial Decoder)"]
        HEAD_FUS["Fusion Head<br/>(Optical ↔ SAR Bidirectional Attention)"]
    end

    subgraph DataStorage ["Data & File Storage"]
        TIFF_STORE["Raw Multi-Spectral & SAR GeoTIFF Store"]
        CKPT_STORE["Model Checkpoints (/checkpoints)"]
        REPORT_STORE["Exported Mission Reports (.pdf)"]
    end

    UI <--> |REST / SSE Stream| ROUTER
    MAP <--> |GeoJSON / Heatmap Overlays| ROUTER
    CHAT <--> |Natural Language Queries| AGENT
    AGENT <--> |Tool Invocations & Parameters| ROUTER
    AGENT --- MEM
    AGENT --- TOOL_REG

    ROUTER --> PREPROC
    PREPROC --> |Harmonized [3, 224, 224] Tensors| BACKBONE
    BACKBONE --> |1024-d Visual Tokens [B, 197, 1024]| HEAD_VQA
    BACKBONE --> |Patch Tokens + Text Embeddings| HEAD_GRD
    BACKBONE --> |Bi-temporal Token Pairs (f1, f2)| HEAD_CHG
    BACKBONE --> |Optical + SAR Token Pairs| HEAD_FUS

    HEAD_VQA --> CONF_ENG
    HEAD_GRD --> CONF_ENG
    HEAD_CHG --> CONF_ENG
    HEAD_FUS --> CONF_ENG

    CONF_ENG --> ROUTER
    ROUTER --> AUDIT
    AUDIT --> CHAT
    PREPROC --- TIFF_STORE
    BACKBONE --- CKPT_STORE
    ROUTER --> PDF_BTN
    PDF_BTN --- REPORT_STORE
```

---

## 2. SatQueryUnified Model Architecture

The deep learning model `SatQueryUnified` processes heterogeneous satellite modalities through a shared, high-capacity vision foundation model and routes specialized task features to four dedicated heads.

```mermaid
graph TD
    subgraph Inputs ["Input Satellite Imagery"]
        OPT["Optical Image (Sentinel-2 / Cartosat-2S)"]
        SAR["SAR Image (Sentinel-1 / RISAT-1/2)"]
        T1["Bi-temporal T1 Image (Pre-event)"]
        T2["Bi-temporal T2 Image (Post-event)"]
        TXT["User Prompt / Question / Grounding Phrase"]
    end

    subgraph Foundation ["Shared Remote Sensing Foundation Backbone"]
        VIT["RemoteCLIP ViT-L/14 Vision Transformer<br/>(1024-dim, 24 Transformer Blocks, 16 Heads)"]
        TEXT_ENC["RemoteCLIP Text Encoder<br/>(77 Tokens Max, 1024-dim Projection)"]
    end

    subgraph PatchFeatures ["Shared Visual Token Space"]
        TOKENS["Visual Patch Tokens: [B, 197, 1024]<br/>(196 Spatial Patches + 1 CLS Token)"]
        TOK_T1["T1 Patch Tokens: [B, 197, 1024]"]
        TOK_T2["T2 Patch Tokens: [B, 197, 1024]"]
        TOK_OPT["Optical Patch Tokens: [B, 197, 1024]"]
        TOK_SAR["SAR Patch Tokens: [B, 197, 1024]"]
        EMB_TXT["Normalized Text Embedding: [B, 1024]"]
    end

    subgraph Heads ["Specialized Multi-Task Heads"]
        subgraph VQA_Module ["1. VQA Head"]
            V_PROJ["Linear Projection (1024 → 768)"]
            X_ATTN["2x Visual-Text Cross-Attention Layers"]
            T5["Flan-T5-base Language Decoder (768-dim)"]
            ANS["Generated Text Answer + Token Log-Probs"]
        end

        subgraph Grounding_Module ["2. Grounding Head"]
            G_PROJ["Patch & Text Projection (1024 → 256)"]
            Q_OBJ["100 Learned Object Queries"]
            DETR_DEC["3-Layer Transformer Decoder"]
            MLP_BOX["BBox Head (cx, cy, w, h) ∈ [0, 1]"]
            MLP_CLS["Confidence / Objectness Logit"]
        end

        subgraph Change_Module ["3. Siamese Change Head"]
            DIFF["Learned Difference Conv Block<br/>Conv2d(2048 → 512 → 256)"]
            UNET["4-Stage Bilinear U-Net Decoder"]
            CHG_MAP["Spatial Change Probability Map [H, W] ∈ [0, 1]"]
            CHG_FEAT["Change Features (1024-dim) → VQA Head"]
        end

        subgraph Fusion_Module ["4. Optical-SAR Fusion Head"]
            OPT_SAR["Bidirectional Cross-Attention (2 Layers, 8 Heads)<br/>Optical attends to SAR & SAR attends to Optical"]
            FUSED["Unified Multi-Sensor Representation [B, 197, 1024]"]
        end
    end

    OPT --> VIT
    SAR --> VIT
    T1 --> VIT
    T2 --> VIT
    TXT --> TEXT_ENC

    VIT --> TOKENS
    VIT --> TOK_T1
    VIT --> TOK_T2
    VIT --> TOK_OPT
    VIT --> TOK_SAR
    TEXT_ENC --> EMB_TXT

    TOKENS --> V_PROJ --> X_ATTN --> T5 --> ANS
    TOKENS & EMB_TXT --> G_PROJ --> DETR_DEC
    Q_OBJ --> DETR_DEC
    DETR_DEC --> MLP_BOX
    DETR_DEC --> MLP_CLS

    TOK_T1 & TOK_T2 --> DIFF --> UNET --> CHG_MAP
    DIFF --> CHG_FEAT --> V_PROJ

    TOK_OPT & TOK_SAR --> OPT_SAR --> FUSED --> V_PROJ
```

---

## 3. 5-Stage Cross-Sensor Domain Normalization Pipeline

This pipeline bridges the domain gap between Sentinel training distributions and evaluation satellites (Cartosat-2S and RISAT-1/2).

```mermaid
flowchart LR
    RAW["Raw Input Raster<br/>(GeoTIFF, TIFF, PNG, JPEG)"]

    subgraph S1 ["Stage 1: Channel Harmonization"]
        CH_OPT["Optical: Extract / Map to [B, G, R, NIR]<br/>Cartosat-2S HRMX (4-Band) ↔ Sentinel-2 (B2, B3, B4, B8)"]
        CH_SAR["SAR: Map to [Co-Pol, Cross-Pol]<br/>RISAT Hybrid Pol (Stokes S0..S3) → Pseudo VV/VH"]
    end

    subgraph S2 ["Stage 2: Radiometric Calibration"]
        RC_OPT["Dark Object Subtraction (DOS)<br/>11-bit DN → Surface Reflectance (BOA) ∈ [0, 1]"]
        RC_SAR["Radiometric Calibration to Sigma0 (dB)<br/>Apply Refined Lee 5x5 Adaptive Speckle Filter"]
    end

    subgraph S3 ["Stage 3: Spatial Scale Alignment"]
        DOWN["Anti-Aliasing Gaussian Downsampling (σ=1.5)<br/>Align Sub-Meter (0.65m-2m) to Baseline 10m GSD"]
    end

    subgraph S4 ["Stage 4: Statistical & Fourier Adaptation"]
        HIST["CDF Histogram Matching against Sentinel Reference"]
        FDA["Fourier Domain Adaptation (FDA, β=0.05)<br/>Swap Low-Frequency Amplitude Spectrum"]
        CLIP["Robust Percentile Clipping [P1, P99]"]
    end

    subgraph S5 ["Stage 5: Test-Time Augmentation (TTA)"]
        TTA_ROT["Multi-Scale / Flips / 90° Rotations"]
        NORM["Standardize to [3, 224, 224] Float32 Tensor"]
    end

    RAW --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> MODEL["Fed to SatQueryUnified Backbone"]
```

---

## 4. Agentic Decision & Tool-Routing Workflow

The Agentic Controller runs in a closed loop, inspecting input metadata, query intent, and tool outcomes.

```mermaid
sequenceDiagram
    autonumber
    actor Scientist as ISRO Scientist / User
    participant UI as Next.js Dashboard
    participant Agent as Claude Agent Controller
    participant Check as CompatibilityChecker
    participant Model as SatQueryUnified (FastAPI)
    participant Conf as ConfidenceEngine

    Scientist->>UI: Uploads GeoTIFFs + submits NL Query
    UI->>Check: Validate headers, CRS, acquisition dates
    Check-->>UI: CompatibilityReport (Modality, Resolution, Alignment)
    UI->>Agent: Prompt + File Context + Permitted Tool Schemas

    Note over Agent: Intent Classification & Multi-Step Reasoning
    rect rgb(240, 248, 255)
        alt Bi-Temporal Change Detection Required
            Agent->>Model: run_change(image1_id, image2_id, question)
            Model-->>Agent: {change_map_url, change_features, text_summary}
        else Optical-SAR Fusion Analysis Required
            Agent->>Model: run_fusion(optical_id, sar_id, question)
            Model-->>Agent: {fused_answer, fused_tokens}
        else Object Grounding / Localization Required
            Agent->>Model: run_grounding(image_id, target_query)
            Model-->>Agent: {boxes: [[cx,cy,w,h]], detection_scores}
        else General Satellite VQA
            Agent->>Model: run_vqa(image_id, question)
            Model-->>Agent: {answer, token_logits}
        end
    end

    Model->>Conf: Compute Task Uncertainty & Bottlenecks
    Conf-->>Model: {confidence: 0.89, tier: 'HIGH', icon: '🟢'}
    Model-->>Agent: Combined Specialist Output + Confidence

    Note over Agent: Synthesizes Final Briefing & Auditable Trace
    Agent->>UI: Final Response + Bounding Boxes + Heatmap URL + Execution Trace
    UI->>Scientist: Visual Map Render + Confidence Tier + Downloadable Report
```

---

## 5. Unified Confidence Estimation Topology

Confidence is calibrated across all four modalities before being aggregated into a single verifiable metric.

```mermaid
graph TD
    subgraph VQA_Uncertainty ["VQA Confidence"]
        V_IN["T5 Transition Token Log-Probs"]
        V_GEOM["Geometric Mean Prob = exp(mean(log_p))"]
        V_MIN["Min Token Prob = min(exp(log_p))"]
        V_CAL["Temperature Scaling (T=1.2)"]
        V_SCORE["C_vqa = 0.7*Geom + 0.3*Min"]
        V_IN --> V_GEOM & V_MIN --> V_CAL --> V_SCORE
    end

    subgraph Grounding_Uncertainty ["Grounding Confidence"]
        G_LOGIT["Sigmoid of Text-Box Query Alignment Logits"]
        G_IOU["Query Entropy & Bounding Box Variance"]
        G_SCORE["C_grd = 0.6*MaxLogit + 0.4*BoxStability"]
        G_LOGIT & G_IOU --> G_SCORE
    end

    subgraph Change_Uncertainty ["Change Confidence"]
        C_PIX["Pixel Confidence: 2 * |P(change) - 0.5|"]
        C_AMB["Ambiguity Ratio: Fraction of Pixels in [0.35, 0.65]"]
        C_FG["Foreground Confidence over Changed Pixels"]
        C_SCORE["C_chg = 0.6*(1 - Ambiguity) + 0.4*FG_Conf"]
        C_PIX & C_AMB & C_FG --> C_SCORE
    end

    subgraph Aggregation ["Multi-Task Aggregation & Quality Tiering"]
        HARMONIC["Weighted Harmonic Mean<br/>C_final = Σw / Σ(w / (C_i + ε))<br/>(Penalizes lowest-performing task bottleneck)"]
        TIERS{"Confidence Tier Thresholds"}
        HIGH["🟢 HIGH CONFIDENCE (Score ≥ 0.80)<br/>Autonomous Mission Decision Approved"]
        MED["🟡 MEDIUM CONFIDENCE (0.55 ≤ Score < 0.80)<br/>Scientist Verification Recommended"]
        LOW["🔴 LOW CONFIDENCE (Score < 0.55)<br/>Sensor Gap or Cloud Obstruction Detected"]
    end

    V_SCORE --> HARMONIC
    G_SCORE --> HARMONIC
    C_SCORE --> HARMONIC

    HARMONIC --> TIERS
    TIERS -->|Score ≥ 0.80| HIGH
    TIERS -->|0.55 ≤ Score < 0.80| MED
    TIERS -->|Score < 0.55| LOW
```
