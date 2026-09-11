"use client";

import React, { useEffect, useState } from "react";
import { 
  MapContainer, 
  TileLayer, 
  Rectangle, 
  Tooltip, 
  Popup, 
  useMap, 
  Polygon,
  CircleMarker
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { GroundingBox, GeoTIFFMetadata } from "../../types/satquery";
import { 
  Crosshair, 
  Layers, 
  Flame, 
  SplitSquareVertical, 
  Eye,
  EyeOff,
  Grid
} from "lucide-react";

interface LeafletMapProps {
  center: [number, number];
  zoom: number;
  files: GeoTIFFMetadata[];
  groundingBoxes: GroundingBox[];
  showHeatmap: boolean;
  onToggleHeatmap: () => void;
  hasSwipeComparison: boolean;
  showSwipe: boolean;
  onToggleSwipe: () => void;
  activeScenarioId?: string;
}

const MapController: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom, { animate: true, duration: 1.0 });
  }, [center, zoom, map]);
  return null;
};

const CoordsTracker: React.FC<{ resolution: number }> = ({ resolution }) => {
  const map = useMap();
  const [coords, setCoords] = useState<{ lat: number; lng: number; zoom: number }>({
    lat: map.getCenter().lat,
    lng: map.getCenter().lng,
    zoom: map.getZoom(),
  });

  useEffect(() => {
    const onMove = () => {
      const c = map.getCenter();
      setCoords({
        lat: Number(c.lat.toFixed(5)),
        lng: Number(c.lng.toFixed(5)),
        zoom: map.getZoom(),
      });
    };
    map.on("move", onMove);
    map.on("zoom", onMove);
    return () => {
      map.off("move", onMove);
      map.off("zoom", onMove);
    };
  }, [map]);

  return (
    <div className="absolute bottom-3 left-3 z-[400] bg-charcoal-900/90 border border-charcoal-700 px-3 py-2 rounded font-mono text-[11px] text-charcoal-300 space-y-1 select-none pointer-events-auto">
      <div className="flex items-center gap-1.5 text-coral-accent font-semibold text-[10px] uppercase">
        <Crosshair className="w-3 h-3" />
        <span>Nadir Sensor Reticle</span>
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
        <div>
          <span className="text-charcoal-500">LAT: </span>
          <span className="text-charcoal-100">{coords.lat}°N</span>
        </div>
        <div>
          <span className="text-charcoal-500">LNG: </span>
          <span className="text-charcoal-100">{coords.lng}°E</span>
        </div>
        <div>
          <span className="text-charcoal-500">ZOOM: </span>
          <span className="text-coral-accent">{coords.zoom}x</span>
        </div>
        <div>
          <span className="text-charcoal-500">GSD: </span>
          <span className="text-charcoal-200">{resolution}m</span>
        </div>
      </div>
    </div>
  );
};

export const LeafletMap: React.FC<LeafletMapProps> = ({
  center,
  zoom,
  files,
  groundingBoxes,
  showHeatmap,
  onToggleHeatmap,
  hasSwipeComparison,
  showSwipe,
  onToggleSwipe,
  activeScenarioId,
}) => {
  const [showBoxes, setShowBoxes] = useState(true);
  const [activeTileType, setActiveTileType] = useState<"dark" | "satellite">("dark");

  const resolution = files.length > 0 ? files[0].resolution_m : 10.0;

  // Single coral accent for high confidence, distinct restrained tones for others
  const getBoxStyle = (category: string, confidence: number) => {
    if (confidence >= 95) {
      return { color: "#e07856", fillColor: "#e07856", fillOpacity: 0.18, weight: 1.5 };
    }
    switch (category) {
      case "vessel":
        return { color: "#e07856", fillColor: "#e07856", fillOpacity: 0.15, weight: 1.5 };
      case "flood":
        return { color: "#5294e2", fillColor: "#5294e2", fillOpacity: 0.3, weight: 1.5 };
      default:
        return { color: "#d99b32", fillColor: "#d99b32", fillOpacity: 0.18, weight: 1.5 };
    }
  };

  const renderHeatmapPolygons = () => {
    if (!showHeatmap) return null;

    if (activeScenarioId === "scenario-2-flood-change") {
      return (
        <>
          <Polygon
            positions={[
              [26.58, 93.18],
              [26.62, 93.22],
              [26.66, 93.30],
              [26.63, 93.36],
              [26.59, 93.32],
              [26.56, 93.24],
            ]}
            pathOptions={{
              color: "#5294e2",
              fillColor: "#5294e2",
              fillOpacity: 0.35,
              weight: 1,
              dashArray: "3, 3",
            }}
          >
            <Tooltip permanent direction="center" className="!bg-charcoal-900 !text-charcoal-100 !border-charcoal-700 !font-mono text-[10px]">
              INUNDATION ZONE: 142.84 km²
            </Tooltip>
          </Polygon>
          <CircleMarker
            center={[26.64, 93.25]}
            radius={22}
            pathOptions={{
              color: "#e07856",
              fillColor: "#e07856",
              fillOpacity: 0.4,
              weight: 1.5,
            }}
          >
            <Tooltip permanent direction="top" className="!bg-charcoal-900 !text-coral-accent !border-charcoal-700 !font-mono text-[9px]">
              PRIMARY BREACH
            </Tooltip>
          </CircleMarker>
        </>
      );
    }

    if (activeScenarioId === "scenario-3-sar-cloud-fusion") {
      return (
        <Polygon
          positions={[
            [18.89, 72.72],
            [18.98, 72.71],
            [19.02, 72.84],
            [18.94, 72.90],
            [18.87, 72.80],
          ]}
          pathOptions={{
            color: "#e07856",
            fillColor: "#e07856",
            fillOpacity: 0.25,
            weight: 1,
          }}
        >
          <Tooltip permanent direction="center" className="!bg-charcoal-900 !text-coral-accent !border-charcoal-700 !font-mono text-[10px]">
            SAR PENETRATION 96.4%
          </Tooltip>
        </Polygon>
      );
    }

    return null;
  };

  return (
    <div className="relative w-full h-full bg-charcoal-950 overflow-hidden font-mono text-xs select-none">
      {/* Top HUD Controls Bar */}
      <div className="absolute top-3 left-3 right-3 z-[400] flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* Left Status */}
        <div className="flex items-center gap-2 bg-charcoal-900/90 border border-charcoal-700 px-3 py-1.5 rounded pointer-events-auto text-charcoal-300">
          <span className="w-1.5 h-1.5 rounded-full bg-coral-accent" />
          <span className="font-semibold text-charcoal-100">Canvas:</span>
          <span>{files.length > 0 ? files[0].sensor_hint.toUpperCase() : "STANDBY"}</span>
          <span className="text-charcoal-600">|</span>
          <span className="text-charcoal-400">{groundingBoxes.length} grounded</span>
        </div>

        {/* Right Toggle Controls */}
        <div className="flex items-center gap-1.5 bg-charcoal-900/90 border border-charcoal-700 p-1 rounded pointer-events-auto text-xs">
          <button
            onClick={() => setActiveTileType(activeTileType === "dark" ? "satellite" : "dark")}
            className={`px-2 py-1 rounded text-[11px] flex items-center gap-1 transition-colors ${
              activeTileType === "satellite"
                ? "bg-charcoal-750 text-coral-accent font-semibold"
                : "text-charcoal-400 hover:text-charcoal-200"
            }`}
          >
            <Layers className="w-3 h-3" />
            {activeTileType === "dark" ? "Dark Vector" : "Satellite"}
          </button>

          <button
            onClick={onToggleHeatmap}
            className={`px-2 py-1 rounded text-[11px] flex items-center gap-1 transition-colors ${
              showHeatmap
                ? "bg-charcoal-750 text-coral-accent font-semibold"
                : "text-charcoal-400 hover:text-charcoal-200"
            }`}
          >
            <Flame className="w-3 h-3" />
            Heatmap {showHeatmap ? "On" : "Off"}
          </button>

          {hasSwipeComparison && (
            <button
              onClick={onToggleSwipe}
              className={`px-2 py-1 rounded text-[11px] flex items-center gap-1 transition-colors ${
                showSwipe
                  ? "bg-charcoal-750 text-coral-accent font-semibold"
                  : "text-charcoal-400 hover:text-charcoal-200"
              }`}
            >
              <SplitSquareVertical className="w-3 h-3 text-coral-accent" />
              Swipe
            </button>
          )}

          <button
            onClick={() => setShowBoxes(!showBoxes)}
            className={`px-2 py-1 rounded text-[11px] flex items-center gap-1 transition-colors ${
              showBoxes ? "text-coral-accent" : "text-charcoal-500"
            }`}
          >
            {showBoxes ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            BBoxes
          </button>
        </div>
      </div>

      {/* Target Legend (Bottom-Right) */}
      <div className="absolute bottom-3 right-3 z-[400] bg-charcoal-900/90 border border-charcoal-700 p-2.5 rounded text-[10px] text-charcoal-300 space-y-1 select-none pointer-events-auto max-w-[190px]">
        <div className="text-charcoal-400 font-semibold border-b border-charcoal-750 pb-1 flex justify-between">
          <span>GROUNDED REGIONS</span>
          <span className="text-coral-accent">EPSG:4326</span>
        </div>
        <div className="space-y-1 pt-0.5">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-sm border border-coral-accent bg-coral-accent/20" />
            <span>High Conf Target / Vessel</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-sm border border-status-amber bg-status-amber/20" />
            <span>Infrastructure / Tank</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-sm border border-status-blue bg-status-blue/20" />
            <span>Inundation Vector Zone</span>
          </div>
        </div>
      </div>

      {/* Map Engine */}
      <MapContainer
        center={center}
        zoom={zoom}
        scrollWheelZoom={true}
        className="w-full h-full z-10"
        zoomControl={true}
      >
        <MapController center={center} zoom={zoom} />
        <CoordsTracker resolution={resolution} />

        {activeTileType === "dark" ? (
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            maxZoom={19}
          />
        ) : (
          <TileLayer
            attribution='Tiles &copy; Esri &mdash; Earth Observation'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            maxZoom={18}
          />
        )}

        {renderHeatmapPolygons()}

        {showBoxes &&
          groundingBoxes.map((box) => {
            const bounds: [[number, number], [number, number]] = [
              [box.bbox[0], box.bbox[1]],
              [box.bbox[2], box.bbox[3]],
            ];
            const style = getBoxStyle(box.category, box.confidence);

            return (
              <Rectangle key={box.id} bounds={bounds} pathOptions={style}>
                <Tooltip direction="top" offset={[0, -10]} opacity={0.95} className="!bg-charcoal-900 !border-charcoal-700 !text-charcoal-100 !font-mono text-xs">
                  <div className="p-1 space-y-1">
                    <div className="font-semibold text-coral-accent flex items-center justify-between gap-2">
                      <span>{box.label}</span>
                      <span className="text-[10px] text-charcoal-300">{box.confidence}%</span>
                    </div>
                    {box.attributes && (
                      <div className="text-[10px] text-charcoal-400 space-y-0.5 border-t border-charcoal-750 pt-1">
                        {Object.entries(box.attributes).map(([k, v]) => (
                          <div key={k} className="flex justify-between gap-2">
                            <span className="text-charcoal-500">{k}:</span>
                            <span className="text-charcoal-300">{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </Tooltip>
              </Rectangle>
            );
          })}
      </MapContainer>
    </div>
  );
};
