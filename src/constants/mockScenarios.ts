import { MissionScenario } from "../types/satquery";

export const MOCK_SCENARIOS: MissionScenario[] = [
  {
    id: "scenario-1-grounding",
    title: "Scenario 1: Optical Infrastructure & Vessel Grounding",
    description: "High-resolution Sentinel-2/Cartosat scene over Visakhapatnam Port. Identifies maritime assets, berths, and storage facilities.",
    keywords: ["vessel", "dock", "berth", "ship", "tank", "infrastructure", "port", "crane", "grounding", "object", "detect"],
    config_type: "Single Optical",
    files: [
      {
        id: "upl_visakhapatnam_opt",
        filename: "S2A_MSIL2A_20241015_VISAKHAPATNAM.tif",
        sensor_hint: "sentinel2",
        bands: 12,
        crs: "EPSG:32644",
        resolution_m: 10.0,
        width: 1024,
        height: 1024,
        acquisition_date: "2024-10-15T05:42:11Z",
        bbox: [17.675, 83.275, 17.715, 83.325], // Visakhapatnam Port
        center: [17.695, 83.300],
        thumbnail_url: "https://images.unsplash.com/photo-1578575437130-527eed3abbec?auto=format&fit=crop&w=400&q=80",
        data_type: "Optical",
        file_size_mb: 48.6,
      },
    ],
    default_query: "Detect and isolate all maritime vessels, cargo berths, and fuel storage tanks in the port area.",
    response: {
      text: `### Mission Analysis Summary: Visakhapatnam Port Grounding

**1. Target Identification & Localization**
The agentic vision-language pipeline executed multi-scale spatial grounding across the 10m VNIR bands (B2, B3, B4, B8). 

- **Maritime Vessels Detected**: **5 cargo/tanker vessels** actively berthed or anchored in the inner channel (Average Confidence: **94.8%**).
- **Critical Infrastructure**: **3 oil/chemical storage tanks** and **2 gantry crane terminals** pinpointed with sub-pixel localization accuracy.
- **Surface Conditions**: Clear atmospheric window (AOT 0.12), zero cloud obstruction over the primary navigation canal.

Bounding geometries and radiometric signatures have been projected onto the map canvas.`,
      confidence_tier: "HIGH",
      confidence_score: 96.4,
      scenario_id: "scenario-1-grounding",
      execution_trace: [
        {
          step: 1,
          tool: "parse_vqa_intent",
          tool_display_name: "Vision-Language Intent Parser",
          params: { query: "Detect maritime vessels, cargo berths, and storage tanks", domain: "maritime_infrastructure" },
          duration_ms: 310,
          status: "success",
          output: { target_classes: ["vessel", "storage_tank", "cargo_berth"], spatial_context: "port_basin" },
        },
        {
          step: 2,
          tool: "extract_multiband_features",
          tool_display_name: "Sentinel-2 VNIR Feature Extractor",
          params: { scene_id: "S2A_MSIL2A_20241015", bands: ["B02", "B03", "B04", "B08"], resolution: 10.0 },
          duration_ms: 450,
          status: "success",
          output: { feature_tensor_shape: [1, 256, 1024, 1024], snr_db: 38.2 },
        },
        {
          step: 3,
          tool: "ground_zero_shot_entities",
          tool_display_name: "Open-Vocabulary Grounding Engine",
          params: { prompts: ["commercial vessel", "crane terminal", "cylindrical fuel tank"], iou_threshold: 0.45 },
          duration_ms: 580,
          status: "success",
          output: { detected_count: 8, mean_confidence: 0.948 },
        },
        {
          step: 4,
          tool: "calibrate_spatial_coordinates",
          tool_display_name: "CRS Reprojection & Bounding Box Formatter",
          params: { source_crs: "EPSG:32644", target_crs: "EPSG:4326" },
          duration_ms: 220,
          status: "success",
          output: { valid_bboxes: 8, geojson_polygons_generated: true },
        },
      ],
      grounding_boxes: [
        {
          id: "box-vessel-1",
          label: "Container Vessel (Length ~185m)",
          confidence: 97.2,
          category: "vessel",
          bbox: [17.691, 83.292, 17.697, 83.299],
          attributes: { heading: "142°", status: "Berthed", draft_est: "11.4m" },
        },
        {
          id: "box-vessel-2",
          label: "Bulk Carrier (Length ~220m)",
          confidence: 95.8,
          category: "vessel",
          bbox: [17.684, 83.303, 17.689, 83.311],
          attributes: { heading: "085°", status: "Moored", draft_est: "13.8m" },
        },
        {
          id: "box-vessel-3",
          label: "Petroleum Tanker",
          confidence: 94.1,
          category: "vessel",
          bbox: [17.702, 83.284, 17.706, 83.290],
          attributes: { heading: "210°", status: "Cargo Ops", risk_tier: "Normal" },
        },
        {
          id: "box-tank-1",
          label: "POL Storage Cluster A (Diameter 45m)",
          confidence: 98.4,
          category: "infrastructure",
          bbox: [17.708, 83.295, 17.713, 83.302],
          attributes: { type: "Refined Product", integrity: "100%" },
        },
        {
          id: "box-tank-2",
          label: "Cryogenic LNG Tank",
          confidence: 96.0,
          category: "infrastructure",
          bbox: [17.712, 83.305, 17.716, 83.312],
          attributes: { thermal_ir_delta: "-4.2K", status: "Insulated" },
        },
        {
          id: "box-vessel-4",
          label: "Tug / Pilot Craft",
          confidence: 91.5,
          category: "vessel",
          bbox: [17.696, 83.306, 17.699, 83.310],
          attributes: { velocity: "4.2 kts" },
        },
      ],
      has_heatmap: false,
      has_swipe_comparison: false,
    },
  },
  {
    id: "scenario-2-flood-change",
    title: "Scenario 2: Bi-Temporal Flood Extent & Damage Assessment",
    description: "Pre-monsoon vs. Post-flood Sentinel-2 / Landsat-8 bi-temporal pair over the Kaziranga-Brahmaputra floodplain.",
    keywords: ["flood", "flooding", "water", "change", "inundation", "damage", "monsoon", "river", "submerged", "bitemporal", "pre", "post"],
    config_type: "Bi-Temporal Pair",
    files: [
      {
        id: "upl_brahmaputra_t1",
        filename: "S2B_20240610_T1_PRE_FLOOD.tif",
        sensor_hint: "sentinel2",
        bands: 12,
        crs: "EPSG:32646",
        resolution_m: 10.0,
        width: 1024,
        height: 1024,
        acquisition_date: "2024-06-10T04:55:20Z",
        bbox: [26.550, 93.150, 26.680, 93.380], // Kaziranga / Brahmaputra
        center: [26.615, 93.265],
        thumbnail_url: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=400&q=80",
        data_type: "Optical",
        file_size_mb: 52.1,
      },
      {
        id: "upl_brahmaputra_t2",
        filename: "S2A_20240722_T2_POST_FLOOD.tif",
        sensor_hint: "sentinel2",
        bands: 12,
        crs: "EPSG:32646",
        resolution_m: 10.0,
        width: 1024,
        height: 1024,
        acquisition_date: "2024-07-22T04:58:14Z",
        bbox: [26.550, 93.150, 26.680, 93.380],
        center: [26.615, 93.265],
        thumbnail_url: "https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&w=400&q=80",
        data_type: "Optical",
        file_size_mb: 54.3,
      },
    ],
    default_query: "Calculate total flood inundation area, compare pre/post flood boundaries, and quantify agricultural submergence.",
    response: {
      text: `### Bi-Temporal Inundation & Change Detection Report

**1. Hydrodynamic Change Overview**
Differential spectral analysis (MNDWI & Sentinel-2 B11 SWIR absorption) reveals severe riverine overflow along the Brahmaputra south bank.

- **Total Inundated Area**: **142.84 km²** (Net expansion: **+38.4%** compared to baseline T1).
- **Agricultural Cropland Submerged**: **84.60 km²** (Primary paddy & tea cultivation zones).
- **Settlement & Road Network Cutoff**: **18.2 km** of rural transit corridors inundated.
- **Change Ratio**: **0.042 (4.2% total scene transition)** with **98.1% change classification confidence**.

*Use the swipe divider on the map canvas to inspect pre/post flood boundary shifts and toggle the change heatmap.*`,
      confidence_tier: "HIGH",
      confidence_score: 97.8,
      scenario_id: "scenario-2-flood-change",
      execution_trace: [
        {
          step: 1,
          tool: "coregister_bitemporal_scenes",
          tool_display_name: "Sub-Pixel Phase Coregistration",
          params: { t1_image: "upl_brahmaputra_t1.tif", t2_image: "upl_brahmaputra_t2.tif", algorithm: "SURF_RANSAC" },
          duration_ms: 380,
          status: "success",
          output: { tie_points_matched: 1420, rmse_pixels: 0.18, coregistered: true },
        },
        {
          step: 2,
          tool: "compute_ndwi_diff",
          tool_display_name: "Modified NDWI Water Index Differencing",
          params: { green_band: "B03", swir_band: "B11", dynamic_threshold: 0.24 },
          duration_ms: 340,
          status: "success",
          output: { delta_water_pixels: 1428400, water_mask_created: true },
        },
        {
          step: 3,
          tool: "run_change_detection_unet",
          tool_display_name: "Siamese Change-Former U-Net",
          params: { weights: "isro_flood_v3_swir", batch_size: 4 },
          duration_ms: 620,
          status: "success",
          output: { change_detected: true, change_ratio: 0.042, f1_score: 0.932 },
        },
        {
          step: 4,
          tool: "vectorize_inundation_zones",
          tool_display_name: "Vector Polygonization & Area Statistics",
          params: { crs: "EPSG:32646", min_patch_size_px: 50 },
          duration_ms: 280,
          status: "success",
          output: { inundated_area_km2: 142.84, affected_cropland_pct: 59.2 },
        },
      ],
      change_stats: {
        baseline_area_km2: 372.0,
        changed_area_km2: 142.84,
        change_ratio: 0.042,
        change_type: "Riverine Flooding & Soil Waterlogging",
        confidence: 98.1,
        breakdown: [
          { category: "Submerged Agricultural Land", area_km2: 84.6, percentage: 59.2 },
          { category: "Flooded Lowland Pastures", area_km2: 32.1, percentage: 22.5 },
          { category: "Inundated Rural Infrastructure", area_km2: 16.8, percentage: 11.8 },
          { category: "Sediment Deposition Zones", area_km2: 9.34, percentage: 6.5 },
        ],
      },
      has_heatmap: true,
      has_swipe_comparison: true,
      pre_image_url: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80",
      post_image_url: "https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&w=1200&q=80",
      grounding_boxes: [
        {
          id: "box-flood-breach-1",
          label: "Primary Embankment Breach (Width ~65m)",
          confidence: 98.9,
          category: "flood",
          bbox: [26.635, 93.240, 26.650, 93.270],
          attributes: { discharge_flow: "Severe", hazard_level: "Critical" },
        },
        {
          id: "box-flood-inundation-2",
          label: "Submerged Paddy Agriculture (Zone C)",
          confidence: 96.4,
          category: "flood",
          bbox: [26.580, 93.210, 26.620, 93.280],
          attributes: { depth_estimate: "1.2m - 2.4m", standing_water: "Yes" },
        },
        {
          id: "box-flood-corridor-3",
          label: "Cutoff Highway Segment (NH-715)",
          confidence: 94.7,
          category: "infrastructure",
          bbox: [26.600, 93.300, 26.625, 93.340],
          attributes: { impassable_km: "8.4 km", structural_status: "Undermined" },
        },
      ],
    },
  },
  {
    id: "scenario-3-sar-cloud-fusion",
    title: "Scenario 3: Optical + SAR All-Weather Cross-Modal Fusion",
    description: "Cloud-covered Sentinel-2 optical scene fused with Sentinel-1 / RISAT C-band SAR dual-pol (VV/VH) radar backscatter over Mumbai Offshore.",
    keywords: ["sar", "radar", "cloud", "penetration", "fusion", "all-weather", "monsoon", "cross-modal", "speckle", "vv", "vh", "ship", "vessel", "oil"],
    config_type: "Cross-Modal Pair",
    files: [
      {
        id: "upl_mumbai_cloudy_opt",
        filename: "S2A_20240818_MUMBAI_OPTICAL_CLOUDY.tif",
        sensor_hint: "sentinel2",
        bands: 12,
        crs: "EPSG:32643",
        resolution_m: 10.0,
        width: 1024,
        height: 1024,
        acquisition_date: "2024-08-18T05:35:10Z",
        bbox: [18.850, 72.700, 19.050, 72.950], // Mumbai Offshore / Harbor
        center: [18.950, 72.825],
        thumbnail_url: "https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=400&q=80",
        data_type: "Optical",
        file_size_mb: 51.4,
      },
      {
        id: "upl_mumbai_sar_cband",
        filename: "S1A_IW_GRDH_20240818_MUMBAI_SAR_VVVH.tif",
        sensor_hint: "sentinel1_sar",
        bands: 2,
        crs: "EPSG:32643",
        resolution_m: 10.0,
        width: 1024,
        height: 1024,
        acquisition_date: "2024-08-18T06:10:04Z",
        bbox: [18.850, 72.700, 19.050, 72.950],
        center: [18.950, 72.825],
        thumbnail_url: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=400&q=80",
        data_type: "SAR",
        file_size_mb: 32.8,
      },
    ],
    default_query: "Penetrate monsoon cloud cover using SAR cross-modal fusion to detect hidden maritime vessels and slick anomalies.",
    response: {
      text: `### Cross-Modal Optical + SAR All-Weather Intelligence

**1. Sensor Fusion & Cloud Transparency**
The optical scene exhibited **78.4% stratus cloud cover**, rendering pure RGB/NIR detection unviable. Cross-modal feature distillation with Sentinel-1 C-Band SAR (VV/VH cross-polarization ratio) successfully bypassed all cloud attenuation.

- **Cloud Penetration Efficiency**: **96.4%** across dense monsoon front.
- **Deep Target Detection**: **6 maritime vessels** identified via double-bounce radar corner reflection (RCS > 28 dB·m²).
- **Surface Slick Anomaly**: 1 low-backscatter dampening signature detected near anchorage zone (potential bilge discharge or biogenic surfactant film).
- **Fusion Modality**: Multi-head cross-attention synthesized optical coastlines with SAR backscatter intensity.`,
      confidence_tier: "HIGH",
      confidence_score: 95.2,
      scenario_id: "scenario-3-sar-cloud-fusion",
      fusion_note: "Cloud Penetration Active (96.4%): Synthetic Aperture Radar C-Band 5.405 GHz signal penetrated heavy monsoon stratus layer where optical reflectance was completely occluded.",
      execution_trace: [
        {
          step: 1,
          tool: "sar_speckle_lee_filter",
          tool_display_name: "Refined Lee Polarimetric Speckle Filter",
          params: { pol_channels: ["VV", "VH"], window_size: 7, looks: 4.5 },
          duration_ms: 320,
          status: "success",
          output: { enl_improvement: 3.4, speckle_suppressed: true },
        },
        {
          step: 2,
          tool: "crossmodal_feature_alignment",
          tool_display_name: "Geo-Spatial Cross-Attention Coregistration",
          params: { optical_bands: 12, sar_channels: 2, grid_res: 10.0 },
          duration_ms: 410,
          status: "success",
          output: { spatial_alignment_error_m: 0.82, feature_dim: 512 },
        },
        {
          step: 3,
          tool: "sar_optical_attention_fusion",
          tool_display_name: "All-Weather Transformer Fusion Network",
          params: { cloud_mask_threshold: 0.65, sar_weight: 0.85 },
          duration_ms: 540,
          status: "success",
          output: { cloud_penetration_pct: 96.4, fused_feature_map: "tensor_512x1024" },
        },
        {
          step: 4,
          tool: "detect_subcloud_vessels",
          tool_display_name: "CFAR Radar Constant False Alarm Rate Detector",
          params: { pfa: 1e-6, min_rcs_db: 22.0, guard_cells: 5 },
          duration_ms: 390,
          status: "success",
          output: { detected_vessels: 6, anomalies: 1 },
        },
      ],
      grounding_boxes: [
        {
          id: "box-sar-vessel-1",
          label: "Crude Carrier (SAR RCS 34.2 dBm² - Hidden under Cloud)",
          confidence: 96.8,
          category: "vessel",
          bbox: [18.910, 72.780, 18.930, 72.810],
          attributes: { radar_echo: "Double-bounce", optical_visibility: "0% (Occluded)", sar_detected: "Yes" },
        },
        {
          id: "box-sar-vessel-2",
          label: "Naval Patrol Craft (Speed ~18 kts)",
          confidence: 94.5,
          category: "vessel",
          bbox: [18.960, 72.740, 18.975, 72.765],
          attributes: { wake_detected: "Yes", doppler_shift_hz: 48.2 },
        },
        {
          id: "box-sar-vessel-3",
          label: "Offshore Supply Vessel (OSV)",
          confidence: 93.1,
          category: "vessel",
          bbox: [18.880, 72.820, 18.900, 72.845],
          attributes: { target_type: "Platform Service" },
        },
        {
          id: "box-sar-anomaly-4",
          label: "Surface Slick Film (Low Backscatter -22 dB)",
          confidence: 89.4,
          category: "anomaly",
          bbox: [18.935, 72.720, 18.955, 72.755],
          attributes: { area_est_km2: "3.2 km²", suppression_factor: "8.4 dB" },
        },
      ],
      has_heatmap: true,
      has_swipe_comparison: false,
    },
  },
];

export function findMatchingScenario(query: string, uploadedFilesCount: number, hasSar: boolean): MissionScenario {
  const q = query.toLowerCase();
  
  if (q.includes("sar") || q.includes("radar") || q.includes("cloud") || q.includes("fusion") || q.includes("all-weather") || hasSar) {
    return MOCK_SCENARIOS[2]; // Scenario 3 (SAR Fusion)
  }
  
  if (q.includes("flood") || q.includes("change") || q.includes("damage") || q.includes("water") || q.includes("inundation") || q.includes("pre") || q.includes("post") || uploadedFilesCount >= 2) {
    return MOCK_SCENARIOS[1]; // Scenario 2 (Bi-Temporal Flood)
  }
  
  return MOCK_SCENARIOS[0]; // Scenario 1 (Optical Grounding)
}
