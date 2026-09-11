"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { ArrowLeftRight, SplitSquareVertical } from "lucide-react";

interface SwipeSliderProps {
  preImageUrl: string;
  postImageUrl: string;
  preLabel?: string;
  postLabel?: string;
  isActive: boolean;
  onToggle: () => void;
}

export const SwipeSlider: React.FC<SwipeSliderProps> = ({
  preImageUrl,
  postImageUrl,
  preLabel = "T1: Pre-Event",
  postLabel = "T2: Post-Event",
  isActive,
  onToggle,
}) => {
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const percentage = Math.min(Math.max((x / rect.width) * 100, 0), 100);
      setSliderPosition(percentage);
    },
    []
  );

  const handleTouchMove = useCallback(
    (e: TouchEvent) => {
      if (!isDragging) return;
      handleMove(e.touches[0].clientX);
    },
    [isDragging, handleMove]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      handleMove(e.clientX);
    },
    [isDragging, handleMove]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("touchmove", handleTouchMove);
      window.addEventListener("touchend", handleMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp, handleTouchMove]);

  if (!isActive) return null;

  return (
    <div className="absolute inset-0 z-20 pointer-events-none flex flex-col font-mono text-xs select-none">
      {/* Top Banner */}
      <div className="p-3 flex items-center justify-between pointer-events-auto bg-charcoal-950/80 border-b border-charcoal-700">
        <div className="flex items-center gap-2 text-charcoal-200">
          <SplitSquareVertical className="w-4 h-4 text-coral-accent" />
          <span className="font-semibold">Swipe Comparison Active</span>
          <span className="text-charcoal-400 text-[10px] hidden sm:inline">• Drag divider to inspect delta</span>
        </div>
        <button
          onClick={onToggle}
          className="px-2.5 py-1 rounded bg-charcoal-800 hover:bg-charcoal-700 border border-charcoal-600 text-charcoal-200 text-xs transition-colors"
        >
          Exit Swipe View
        </button>
      </div>

      {/* Swipe Canvas */}
      <div
        ref={containerRef}
        className="relative flex-1 m-3 rounded overflow-hidden border border-charcoal-700 pointer-events-auto cursor-ew-resize bg-black"
        onMouseDown={() => setIsDragging(true)}
        onTouchStart={() => setIsDragging(true)}
      >
        {/* Post Layer */}
        <div className="absolute inset-0 w-full h-full">
          <img src={postImageUrl} alt="Post event" className="w-full h-full object-cover" />
          <div className="absolute top-3 right-3 bg-charcoal-900/90 border border-charcoal-700 px-2 py-1 rounded text-[10px] text-charcoal-200">
            {postLabel}
          </div>
        </div>

        {/* Pre Layer */}
        <div
          className="absolute inset-0 w-full h-full overflow-hidden"
          style={{ width: `${sliderPosition}%` }}
        >
          <div
            className="absolute inset-0 w-full h-full"
            style={{ width: containerRef.current ? `${containerRef.current.clientWidth}px` : "100%" }}
          >
            <img src={preImageUrl} alt="Pre event" className="w-full h-full object-cover" />
            <div className="absolute top-3 left-3 bg-charcoal-900/90 border border-charcoal-700 px-2 py-1 rounded text-[10px] text-coral-accent">
              {preLabel}
            </div>
          </div>
        </div>

        {/* Divider Line with Coral Handle */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-coral-accent cursor-ew-resize flex items-center justify-center -ml-[1px]"
          style={{ left: `${sliderPosition}%` }}
        >
          <div className="w-6 h-6 rounded-full bg-coral-accent text-charcoal-950 flex items-center justify-center shadow font-bold">
            <ArrowLeftRight className="w-3.5 h-3.5" />
          </div>
        </div>
      </div>
    </div>
  );
};
