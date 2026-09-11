/**
 * api.ts: Client API utilities for SatQuery AI.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface UploadResponse {
  status: string;
  file_paths: string[];
  thumbnails: string[];
  compatibility: {
    valid: boolean;
    input_type: string;
    sensor_types: string[];
    issues: string[];
    warnings: string[];
    recommended_tasks: string[];
  };
}

export interface QueryResponse {
  status: string;
  query: string;
  answer: string;
  confidence: number;
  tier: "HIGH" | "MEDIUM" | "LOW";
  icon: string;
  visual_evidence: any;
  bounding_boxes?: number[][];
  change_percentage?: number;
  execution_trace: {
    trace_id: string;
    timestamp: string;
    query: string;
    selected_task: string;
    model_name: string;
    selected_tool: string;
    permitted_parameters: any;
    input_summary: any;
    confidence: any;
    latency_seconds: number;
    orchestrator: string;
  };
  compatibility_report: any;
}

export async function uploadSatelliteImages(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const res = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Upload failed");
  }

  return res.json();
}

export async function executeAgentQuery(query: string, filePaths: string[]): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE_URL}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, file_paths: filePaths }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Query execution failed");
  }

  return res.json();
}

export async function generatePdfReport(data: {
  query: string;
  answer: string;
  confidence: number;
  tier: string;
  selected_task: string;
  model_name: string;
  input_type: string;
  sensors: string[];
  execution_trace: any;
}): Promise<{ status: string; file_url: string; local_path?: string }> {
  const res = await fetch(`${API_BASE_URL}/api/generate-report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    throw new Error("Failed to generate PDF report");
  }

  return res.json();
}
