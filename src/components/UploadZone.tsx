"use client";

import React, { useState, useRef } from "react";
import { 
  UploadCloud, 
  FileCode, 
  Layers, 
  Trash2, 
  Info, 
  FileCheck,
  ChevronRight,
  Sparkles,
  Paperclip
} from "lucide-react";
import { GeoTIFFMetadata, InputConfiguration, SensorType } from "../types/satquery";
import { THEME_CONFIG } from "../constants/theme";

interface UploadZoneProps {
  files: GeoTIFFMetadata[];
  onFilesChange: (files: GeoTIFFMetadata[]) => void;
  inputConfig: InputConfiguration;
  onSelectPreset: (scenarioIndex: number) => void;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  files,
  onFilesChange,
  inputConfig,
  onSelectPreset,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFileForModal, setSelectedFileForModal] = useState<GeoTIFFMetadata | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const parseMockMetadataFromFile = (file: File, index: number): GeoTIFFMetadata => {
    const isSar = file.name.toLowerCase().includes("sar") || file.name.toLowerCase().includes("s1") || file.name.toLowerCase().includes("risat");
    const isLandsat = file.name.toLowerCase().includes("landsat") || file.name.toLowerCase().includes("l8");
    const isCartosat = file.name.toLowerCase().includes("carto") || file.name.toLowerCase().includes("c3");
    
    let sensor_hint: SensorType = "sentinel2";
    if (isSar) sensor_hint = "sentinel1_sar";
    else if (isLandsat) sensor_hint = "landsat8";
    else if (isCartosat) sensor_hint = "cartosat3";

    const data_type: "Optical" | "SAR" = isSar ? "SAR" : "Optical";
    const bands = isSar ? 2 : sensor_hint === "cartosat3" ? 4 : 12;
    const resolution_m = isSar ? 10.0 : sensor_hint === "cartosat3" ? 1.12 : 10.0;
    const crs = "EPSG:32643";

    return {
      id: `custom_upl_${Date.now()}_${index}`,
      filename: file.name,
      sensor_hint,
      bands,
      crs,
      resolution_m,
      width: 1024,
      height: 1024,
      acquisition_date: new Date().toISOString(),
      bbox: [18.910, 72.780, 19.010, 72.910],
      center: [18.960, 72.845],
      thumbnail_url: URL.createObjectURL(file),
      data_type,
      file_size_mb: Number((file.size / (1024 * 1024) || 24.5).toFixed(1)),
    };
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const newFileList = Array.from(e.dataTransfer.files).slice(0, 2);
      const parsed = newFileList.map((f, i) => parseMockMetadataFromFile(f, i));
      onFilesChange(parsed);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFileList = Array.from(e.target.files).slice(0, 2);
      const parsed = newFileList.map((f, i) => parseMockMetadataFromFile(f, i));
      onFilesChange(parsed);
    }
  };

  const handleRemoveFile = (id: string) => {
    const updated = files.filter((f) => f.id !== id);
    onFilesChange(updated);
  };

  const configInfo = THEME_CONFIG.configBadges[inputConfig] || THEME_CONFIG.configBadges["Single Optical"];

  return (
    <div className="flex flex-col h-full bg-charcoal-900 border-r border-charcoal-700 select-none overflow-y-auto font-mono text-xs">
      {/* Panel Top Header */}
      <div className="p-3 border-b border-charcoal-700 bg-charcoal-850 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-coral-accent" />
          <span className="font-bold text-charcoal-100 tracking-wide">
            Raster Ingestion
          </span>
        </div>
        <span className="text-[10px] text-charcoal-400 px-1.5 py-0.5 rounded bg-charcoal-800 border border-charcoal-700">
          {files.length}/2 SLOTS
        </span>
      </div>

      <div className="p-3 space-y-4 flex-1">
        {/* Input Configuration Indicator */}
        <div className="space-y-1.5">
          <div className="text-[10px] text-charcoal-400 uppercase tracking-wider">
            Input Configuration
          </div>
          <div className="p-2.5 rounded bg-charcoal-850 border border-charcoal-700 space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-bold text-charcoal-100 flex items-center gap-1.5 text-xs">
                <FileCheck className="w-3.5 h-3.5 text-coral-accent" />
                {configInfo.label}
              </span>
              <span className="text-[9px] text-charcoal-400 bg-charcoal-800 px-1.5 py-0.2 rounded border border-charcoal-700">
                AUTO
              </span>
            </div>
            <p className="text-[10px] text-charcoal-400 font-sans">
              {configInfo.description}
            </p>
          </div>
        </div>

        {/* Drag & Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border border-dashed rounded p-4 text-center cursor-pointer transition-all ${
            isDragging
              ? "border-coral-accent bg-charcoal-800"
              : "border-charcoal-700 bg-charcoal-850 hover:border-charcoal-500 hover:bg-charcoal-800"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".tif,.tiff,.png,.jpg,.jpeg,.json,.geojson"
            className="hidden"
            onChange={handleFileInputChange}
          />
          <div className="flex flex-col items-center justify-center space-y-2 py-1">
            <div className="w-8 h-8 rounded bg-charcoal-800 border border-charcoal-600 flex items-center justify-center text-coral-accent">
              <UploadCloud className="w-4 h-4" />
            </div>
            <div className="text-xs font-semibold text-charcoal-200">
              Drag & Drop GeoTIFF / TIFF
            </div>
            <p className="text-[10px] text-charcoal-400 max-w-[200px]">
              Multi-band Sentinel-2, Landsat-8, Sentinel-1 SAR, Cartosat-3
            </p>
          </div>
        </div>

        {/* 1-Click Test Scenarios */}
        <div className="space-y-1.5 pt-1">
          <div className="text-[10px] text-charcoal-400 uppercase tracking-wider flex items-center gap-1">
            <Sparkles className="w-2.5 h-2.5 text-coral-accent" />
            Test Scene Presets
          </div>
          <div className="space-y-1">
            <button
              onClick={() => onSelectPreset(0)}
              className="w-full px-2.5 py-1.5 text-left rounded bg-charcoal-850 hover:bg-charcoal-800 border border-charcoal-700 hover:border-charcoal-600 text-[11px] text-charcoal-300 hover:text-charcoal-100 flex items-center justify-between transition-colors group"
            >
              <span className="truncate">1. Optical: Visakhapatnam Port</span>
              <ChevronRight className="w-3 h-3 text-charcoal-500 group-hover:text-coral-accent transition-colors" />
            </button>
            <button
              onClick={() => onSelectPreset(1)}
              className="w-full px-2.5 py-1.5 text-left rounded bg-charcoal-850 hover:bg-charcoal-800 border border-charcoal-700 hover:border-charcoal-600 text-[11px] text-charcoal-300 hover:text-charcoal-100 flex items-center justify-between transition-colors group"
            >
              <span className="truncate">2. Bi-Temporal: Brahmaputra Flood</span>
              <ChevronRight className="w-3 h-3 text-charcoal-500 group-hover:text-coral-accent transition-colors" />
            </button>
            <button
              onClick={() => onSelectPreset(2)}
              className="w-full px-2.5 py-1.5 text-left rounded bg-charcoal-850 hover:bg-charcoal-800 border border-charcoal-700 hover:border-charcoal-600 text-[11px] text-charcoal-300 hover:text-charcoal-100 flex items-center justify-between transition-colors group"
            >
              <span className="truncate">3. SAR Fusion: Mumbai All-Weather</span>
              <ChevronRight className="w-3 h-3 text-charcoal-500 group-hover:text-coral-accent transition-colors" />
            </button>
          </div>
        </div>

        {/* Loaded Rasters List */}
        <div className="space-y-2 pt-2 border-t border-charcoal-700">
          <div className="flex items-center justify-between text-[10px] text-charcoal-400 uppercase tracking-wider">
            <span>Loaded Tiles ({files.length})</span>
            {files.length > 0 && (
              <button
                onClick={() => onFilesChange([])}
                className="text-status-red hover:underline capitalize"
              >
                Clear all
              </button>
            )}
          </div>

          {files.length === 0 ? (
            <div className="p-3 rounded bg-charcoal-850 border border-dashed border-charcoal-700 text-center text-charcoal-400 text-[11px]">
              No raster tiles currently loaded
            </div>
          ) : (
            <div className="space-y-2">
              {files.map((file) => (
                <div
                  key={file.id}
                  className="p-2.5 rounded bg-charcoal-850 border border-charcoal-700 hover:border-charcoal-600 transition-colors space-y-2"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      {file.thumbnail_url ? (
                        <img
                          src={file.thumbnail_url}
                          alt="thumb"
                          className="w-9 h-9 rounded object-cover border border-charcoal-700 flex-shrink-0"
                        />
                      ) : (
                        <div className="w-9 h-9 rounded bg-charcoal-800 border border-charcoal-700 flex items-center justify-center text-coral-accent text-[9px] font-bold">
                          {file.data_type === "SAR" ? "SAR" : "MSI"}
                        </div>
                      )}
                      <div className="min-w-0">
                        <div className="text-[11px] font-semibold text-charcoal-100 truncate" title={file.filename}>
                          {file.filename}
                        </div>
                        <div className="flex items-center gap-1.5 mt-0.5 text-[9px] text-charcoal-400">
                          <span>{file.sensor_hint.toUpperCase()}</span>
                          <span>•</span>
                          <span>{file.file_size_mb}MB</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setSelectedFileForModal(file)}
                        className="p-1 rounded text-charcoal-400 hover:text-charcoal-100 hover:bg-charcoal-700"
                        title="View GeoTIFF Schema"
                      >
                        <Info className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleRemoveFile(file.id)}
                        className="p-1 rounded text-charcoal-400 hover:text-status-red hover:bg-charcoal-700"
                        title="Remove"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Metadata Grid */}
                  <div className="grid grid-cols-3 gap-1 pt-1.5 border-t border-charcoal-750 text-[9px] text-charcoal-400">
                    <div className="bg-charcoal-900 p-1 rounded">
                      <span className="text-charcoal-500 block">CRS</span>
                      <span className="text-charcoal-200">{file.crs}</span>
                    </div>
                    <div className="bg-charcoal-900 p-1 rounded">
                      <span className="text-charcoal-500 block">GSD</span>
                      <span className="text-coral-accent">{file.resolution_m}m</span>
                    </div>
                    <div className="bg-charcoal-900 p-1 rounded">
                      <span className="text-charcoal-500 block">DIMS</span>
                      <span className="text-charcoal-200">{file.width}×{file.height}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Compatibility Report JSON Inspector Modal */}
      {selectedFileForModal && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
          <div className="bg-charcoal-850 border border-charcoal-600 rounded max-w-md w-full p-4 shadow-2xl space-y-3 font-mono">
            <div className="flex items-center justify-between border-b border-charcoal-700 pb-2">
              <div className="flex items-center gap-2 text-coral-accent text-xs font-bold">
                <FileCode className="w-4 h-4" />
                GeoTIFF Compatibility Report
              </div>
              <button
                onClick={() => setSelectedFileForModal(null)}
                className="text-charcoal-400 hover:text-charcoal-100 text-xs px-2 py-0.5 rounded bg-charcoal-750"
              >
                ✕
              </button>
            </div>

            <pre className="p-3 bg-charcoal-950 border border-charcoal-750 rounded text-charcoal-300 text-[11px] overflow-x-auto leading-relaxed">
{JSON.stringify(
  {
    filename: selectedFileForModal.filename,
    sensor_hint: selectedFileForModal.sensor_hint,
    bands: selectedFileForModal.bands,
    crs: selectedFileForModal.crs,
    resolution_m: selectedFileForModal.resolution_m,
    width: selectedFileForModal.width,
    height: selectedFileForModal.height,
  },
  null,
  2
)}
            </pre>

            <div className="flex justify-end pt-1">
              <button
                onClick={() => setSelectedFileForModal(null)}
                className="px-3 py-1 bg-charcoal-750 text-charcoal-200 hover:bg-charcoal-700 border border-charcoal-600 rounded text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
