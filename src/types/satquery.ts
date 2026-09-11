export type SensorType = "sentinel2" | "landsat8" | "sentinel1_sar" | "cartosat3" | "risat1_sar" | "planetscope";

export type InputConfiguration = 
  | "Single Optical" 
  | "Single SAR" 
  | "Cross-Modal Pair" 
  | "Bi-Temporal Pair";

export interface GeoTIFFMetadata {
  id: string;
  filename: string;
  sensor_hint: SensorType;
  bands: number;
  crs: string;
  resolution_m: number;
  width: number;
  height: number;
  acquisition_date: string;
  bbox: [number, number, number, number]; // [minLat, minLng, maxLat, maxLng]
  center: [number, number]; // [lat, lng]
  thumbnail_url?: string;
  data_type: "Optical" | "SAR";
  file_size_mb: number;
}

export type ConfidenceTier = "HIGH" | "MEDIUM" | "LOW";

export interface GroundingBox {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [minLat, minLng, maxLat, maxLng]
  category: "vessel" | "aircraft" | "infrastructure" | "flood" | "vegetation" | "urban" | "anomaly";
  attributes?: Record<string, string | number>;
}

export interface ChangeStats {
  baseline_area_km2: number;
  changed_area_km2: number;
  change_ratio: number;
  change_type: string;
  confidence: number;
  breakdown: {
    category: string;
    area_km2: number;
    percentage: number;
  }[];
}

export interface ExecutionTraceStep {
  step: number;
  tool: string;
  tool_display_name: string;
  params: Record<string, any>;
  duration_ms: number;
  status: "running" | "success" | "error";
  output?: Record<string, any>;
}

export interface AssistantResponse {
  id: string;
  sender: "assistant";
  timestamp: string;
  text: string;
  confidence_tier: ConfidenceTier;
  confidence_score: number;
  scenario_id: string;
  execution_trace: ExecutionTraceStep[];
  grounding_boxes?: GroundingBox[];
  change_stats?: ChangeStats;
  fusion_note?: string;
  has_heatmap?: boolean;
  has_swipe_comparison?: boolean;
  pre_image_url?: string;
  post_image_url?: string;
}

export interface UserMessage {
  id: string;
  sender: "user";
  timestamp: string;
  text: string;
}

export type ChatMessage = UserMessage | AssistantResponse;

export interface MissionScenario {
  id: string;
  title: string;
  description: string;
  keywords: string[];
  config_type: InputConfiguration;
  files: GeoTIFFMetadata[];
  default_query: string;
  response: Omit<AssistantResponse, "id" | "timestamp" | "sender">;
}
