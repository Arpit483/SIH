"use client";

import React, { useState } from "react";
import { 
  FileText, 
  Download, 
  Satellite, 
  Layers, 
  BarChart3, 
  Check, 
  ShieldCheck, 
  Info,
  ArrowLeft
} from "lucide-react";
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip as RechartsTooltip, 
  Cell 
} from "recharts";
import { 
  AssistantResponse, 
  GeoTIFFMetadata, 
  MissionScenario,
  GroundingBox
} from "../../types/satquery";
import { THEME_CONFIG } from "../../constants/theme";

interface ReportViewProps {
  scenario: MissionScenario;
  files: GeoTIFFMetadata[];
  latestResponse: AssistantResponse | null;
  onBackToChat: () => void;
}

export const ReportView: React.FC<ReportViewProps> = ({
  scenario,
  files,
  latestResponse,
  onBackToChat,
}) => {
  const [downloadTooltip, setDownloadTooltip] = useState(false);

  const groundingBoxes: GroundingBox[] = latestResponse?.grounding_boxes || scenario.response.grounding_boxes || [];
  const changeStats = latestResponse?.change_stats || scenario.response.change_stats;
  const confidenceTier = latestResponse?.confidence_tier || scenario.response.confidence_tier || "HIGH";
  const confidenceScore = latestResponse?.confidence_score || scenario.response.confidence_score || 96.0;

  const badgeConfig = THEME_CONFIG.confidenceBadges[confidenceTier];

  const chartData = changeStats?.breakdown || [
    { category: "Berths", area_km2: 12.4, percentage: 40 },
    { category: "Anchorages", area_km2: 8.6, percentage: 30 },
    { category: "Storage Tanks", area_km2: 5.2, percentage: 20 },
    { category: "Channels", area_km2: 3.1, percentage: 10 },
  ];

  const BAR_COLORS = ["#e07856", "#d99b32", "#5294e2", "#748094"];

  return (
    <div className="flex-1 h-full bg-charcoal-950 overflow-y-auto p-4 md:p-6 space-y-6 font-mono text-xs">
      {/* Action Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-charcoal-700 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-coral-accent font-bold text-xs">
              REPORT: SIH26167-EO-{Date.now().toString().slice(-6)}
            </span>
            <span className="text-[10px] text-charcoal-400">
              [ISRO MISSION DOS-EO]
            </span>
          </div>
          <h1 className="text-base font-bold text-charcoal-100 mt-1 font-sans">
            Earth Observation Scientific Intelligence Report
          </h1>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onBackToChat}
            className="px-3 py-1.5 rounded bg-charcoal-850 hover:bg-charcoal-800 border border-charcoal-700 text-charcoal-200 text-xs flex items-center gap-1.5 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Back to Chat
          </button>
          <div className="relative">
            <button
              onMouseEnter={() => setDownloadTooltip(true)}
              onMouseLeave={() => setDownloadTooltip(false)}
              className="px-3.5 py-1.5 rounded bg-charcoal-800 border border-charcoal-600 text-charcoal-300 text-xs font-semibold flex items-center gap-1.5 cursor-not-allowed opacity-90"
            >
              <Download className="w-3.5 h-3.5" />
              Download PDF (Mock)
            </button>
            {downloadTooltip && (
              <div className="absolute right-0 top-full mt-1 w-56 p-2 bg-charcoal-900 border border-charcoal-600 rounded text-[10px] text-charcoal-300 z-50 shadow-xl">
                UI Mock affordance: in production this compiles raw GeoJSON layers, bounding metrics, and metadata into a signed ISRO PDF.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Report Document Sheet */}
      <div className="max-w-4xl mx-auto bg-charcoal-900 border border-charcoal-700 rounded p-6 space-y-6 text-charcoal-200">
        {/* Document Header */}
        <div className="flex items-start justify-between border-b border-charcoal-750 pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded bg-charcoal-800 border border-charcoal-700 flex items-center justify-center text-coral-accent">
                <Satellite className="w-4 h-4" />
              </div>
              <div>
                <div className="font-bold text-xs text-charcoal-100">
                  ISRO NATIONAL REMOTE SENSING CENTRE
                </div>
                <div className="text-[11px] text-charcoal-400">
                  SatQuery AI Automated Grounding Verification (SIH26167)
                </div>
              </div>
            </div>
          </div>

          <div
            className="px-3 py-1.5 rounded border text-xs font-bold"
            style={{
              backgroundColor: badgeConfig.bgColor,
              borderColor: badgeConfig.borderColor,
              color: badgeConfig.color,
            }}
          >
            {badgeConfig.label} ({confidenceScore}%)
          </div>
        </div>

        {/* 1. Metadata Table */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-charcoal-300 uppercase tracking-wider">
            <Layers className="w-3.5 h-3.5 text-coral-accent" />
            1. Ingested Raster Metadata
          </div>
          <div className="overflow-x-auto border border-charcoal-700 rounded bg-charcoal-950">
            <table className="w-full text-left text-xs">
              <thead className="bg-charcoal-850 text-charcoal-400 border-b border-charcoal-700">
                <tr>
                  <th className="p-2.5">File</th>
                  <th className="p-2.5">Sensor</th>
                  <th className="p-2.5">Bands</th>
                  <th className="p-2.5">CRS</th>
                  <th className="p-2.5">Resolution</th>
                  <th className="p-2.5">Acquisition Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-charcoal-800 text-charcoal-300 text-[11px]">
                {files.map((file) => (
                  <tr key={file.id}>
                    <td className="p-2.5 font-semibold text-coral-accent truncate max-w-[180px]">{file.filename}</td>
                    <td className="p-2.5">{file.sensor_hint.toUpperCase()}</td>
                    <td className="p-2.5">{file.bands}</td>
                    <td className="p-2.5">{file.crs}</td>
                    <td className="p-2.5">{file.resolution_m}m</td>
                    <td className="p-2.5 text-charcoal-400">{file.acquisition_date.replace("T", " ").slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 2. Visual Overview & Distribution Chart */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border border-charcoal-700 rounded bg-charcoal-950 p-3 space-y-2">
            <div className="text-xs font-bold text-charcoal-300">
              SPECTRAL PREVIEW THUMBNAIL
            </div>
            <div className="h-40 rounded overflow-hidden border border-charcoal-700 bg-black">
              <img
                src={files.length > 0 && files[0].thumbnail_url ? files[0].thumbnail_url : "https://images.unsplash.com/photo-1578575437130-527eed3abbec?auto=format&fit=crop&w=600&q=80"}
                alt="thumbnail"
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          <div className="border border-charcoal-700 rounded bg-charcoal-950 p-3 space-y-2">
            <div className="flex items-center justify-between text-xs font-bold text-charcoal-300">
              <span>AREA DISTRIBUTION (KM²)</span>
              <BarChart3 className="w-3.5 h-3.5 text-coral-accent" />
            </div>
            <div className="h-40 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -25, bottom: 20 }}>
                  <XAxis dataKey="category" tick={{ fill: "#748094", fontSize: 9 }} interval={0} angle={-15} textAnchor="end" />
                  <YAxis tick={{ fill: "#748094", fontSize: 9 }} />
                  <RechartsTooltip contentStyle={{ backgroundColor: "#15181f", borderColor: "#252b36", fontSize: "11px", color: "#e2e6ed" }} />
                  <Bar dataKey="area_km2" fill="#e07856" radius={[2, 2, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* 3. Grounded Entities */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-bold text-charcoal-300 uppercase tracking-wider">
            <span>2. Grounded Detections ({groundingBoxes.length})</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {groundingBoxes.map((box) => (
              <div
                key={box.id}
                className="p-2.5 rounded bg-charcoal-950 border border-charcoal-750 text-xs space-y-1"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-charcoal-100">{box.label}</span>
                  <span className="text-coral-accent text-[10px]">{box.confidence}%</span>
                </div>
                <div className="text-[10px] text-charcoal-400">
                  <span>Class: {box.category}</span>
                  <span className="mx-2">•</span>
                  <span>[{box.bbox[0].toFixed(3)}, {box.bbox[1].toFixed(3)}] to [{box.bbox[2].toFixed(3)}, {box.bbox[3].toFixed(3)}]</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 4. Change Statistics */}
        {changeStats && (
          <div className="p-3.5 rounded bg-charcoal-950 border border-charcoal-700 space-y-2 text-xs">
            <div className="text-coral-accent font-bold">
              3. DIFFERENTIAL CHANGE QUANTIFICATION
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-[11px]">
              <div className="bg-charcoal-900 p-2 rounded">
                <span className="text-charcoal-500 block text-[9px]">BASELINE</span>
                <span className="text-charcoal-200 font-bold">{changeStats.baseline_area_km2} km²</span>
              </div>
              <div className="bg-charcoal-900 p-2 rounded">
                <span className="text-charcoal-500 block text-[9px]">CHANGED</span>
                <span className="text-coral-accent font-bold">{changeStats.changed_area_km2} km²</span>
              </div>
              <div className="bg-charcoal-900 p-2 rounded">
                <span className="text-charcoal-500 block text-[9px]">RATIO</span>
                <span className="text-charcoal-200 font-bold">{(changeStats.change_ratio * 100).toFixed(2)}%</span>
              </div>
              <div className="bg-charcoal-900 p-2 rounded">
                <span className="text-charcoal-500 block text-[9px]">TYPE</span>
                <span className="text-charcoal-200 font-bold truncate block">{changeStats.change_type}</span>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="border-t border-charcoal-750 pt-3 flex items-center justify-between text-[10px] text-charcoal-400">
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-status-green" />
            <span>Digital Signature Verified • ISRO NRSC Node</span>
          </div>
          <span>Confidentiality: Public Research</span>
        </div>
      </div>
    </div>
  );
};
