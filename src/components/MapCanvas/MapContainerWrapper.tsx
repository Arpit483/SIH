"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import { GroundingBox, GeoTIFFMetadata } from "../../types/satquery";
import { SwipeSlider } from "./SwipeSlider";
import { Loader2 } from "lucide-react";

// Dynamic import for LeafletMap to avoid SSR 'window is not defined' error
const DynamicLeafletMap = dynamic(
  () => import("./LeafletMap").then((mod) => mod.LeafletMap),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex flex-col items-center justify-center bg-[#070b14] text-slate-400 space-y-3 font-mono">
        <Loader2 className="w-8 h-8 text-cyan-accent animate-spin" />
        <div className="text-xs tracking-wider">INITIALIZING SATELLITE CANVAS ENGINE...</div>
      </div>
    ),
  }
);

interface MapContainerWrapperProps {
  center: [number, number];
  zoom: number;
  files: GeoTIFFMetadata[];
  groundingBoxes: GroundingBox[];
  showHeatmap: boolean;
  onToggleHeatmap: () => void;
  hasSwipeComparison: boolean;
  preImageUrl?: string;
  postImageUrl?: string;
  activeScenarioId?: string;
}

export const MapContainerWrapper: React.FC<MapContainerWrapperProps> = ({
  center,
  zoom,
  files,
  groundingBoxes,
  showHeatmap,
  onToggleHeatmap,
  hasSwipeComparison,
  preImageUrl,
  postImageUrl,
  activeScenarioId,
}) => {
  const [showSwipe, setShowSwipe] = useState(false);

  return (
    <div className="relative w-full h-full min-h-[400px]">
      <DynamicLeafletMap
        center={center}
        zoom={zoom}
        files={files}
        groundingBoxes={groundingBoxes}
        showHeatmap={showHeatmap}
        onToggleHeatmap={onToggleHeatmap}
        hasSwipeComparison={hasSwipeComparison}
        showSwipe={showSwipe}
        onToggleSwipe={() => setShowSwipe(!showSwipe)}
        activeScenarioId={activeScenarioId}
      />

      {/* Swipe Comparator Overlay */}
      {hasSwipeComparison && preImageUrl && postImageUrl && (
        <SwipeSlider
          preImageUrl={preImageUrl}
          postImageUrl={postImageUrl}
          isActive={showSwipe}
          onToggle={() => setShowSwipe(false)}
        />
      )}
    </div>
  );
};
