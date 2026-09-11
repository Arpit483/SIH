"use client";

import React, { useState } from "react";
import { 
  Send, 
  Paperclip, 
  Terminal, 
  Search, 
  Sliders, 
  ChevronDown, 
  Cpu, 
  Layers,
  Sparkles
} from "lucide-react";

interface ChatComposerProps {
  onSendMessage: (text: string) => void;
  isProcessing: boolean;
  onAttachClick?: () => void;
  onParamsClick?: () => void;
}

export const ChatComposer: React.FC<ChatComposerProps> = ({
  onSendMessage,
  isProcessing,
  onAttachClick,
  onParamsClick,
}) => {
  const [inputText, setInputText] = useState("");
  const [executionMode, setExecutionMode] = useState<"Agent" | "Manual tool">("Agent");
  const [selectedModel, setSelectedModel] = useState("isro-vit-large / vqa-geo-v2");
  const [showModelPicker, setShowModelPicker] = useState(false);

  const models = [
    "isro-vit-large / vqa-geo-v2",
    "sentinel-sam-geo-3b",
    "sar-cband-fusion-transformer",
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isProcessing) return;
    onSendMessage(inputText.trim());
    setInputText("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="p-3 bg-charcoal-900 border-t border-charcoal-700 font-mono text-xs select-none"
    >
      <div className="bg-charcoal-850 border border-charcoal-700 focus-within:border-charcoal-500 rounded p-2.5 transition-colors space-y-2">
        {/* Input field on top */}
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask SatQuery AI about ingested satellite imagery..."
          disabled={isProcessing}
          rows={2}
          className="w-full bg-transparent text-charcoal-100 placeholder-charcoal-400 focus:outline-none resize-none font-mono text-xs leading-relaxed disabled:opacity-50"
        />

        {/* Controls Row */}
        <div className="flex items-center justify-between pt-1 border-t border-charcoal-750 gap-2">
          {/* Below-Left: Small icon-buttons */}
          <div className="flex items-center gap-1 text-charcoal-300">
            <button
              type="button"
              onClick={onAttachClick}
              className="p-1.5 rounded hover:bg-charcoal-700 hover:text-charcoal-100 transition-colors"
              title="Attach GeoTIFF Raster"
            >
              <Paperclip className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={onParamsClick}
              className="p-1.5 rounded hover:bg-charcoal-700 hover:text-charcoal-100 transition-colors"
              title="View Raw Geospatial Parameters"
            >
              <Terminal className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              className="p-1.5 rounded hover:bg-charcoal-700 hover:text-charcoal-100 transition-colors"
              title="Search Catalog"
            >
              <Search className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Below-Right: Mode toggle + Model selector pill + Primary Send button */}
          <div className="flex items-center gap-2">
            {/* Mode toggle (Agent / Manual tool) */}
            <div className="flex items-center bg-charcoal-800 border border-charcoal-700 rounded p-0.5 text-[10px]">
              <button
                type="button"
                onClick={() => setExecutionMode("Agent")}
                className={`px-2 py-0.5 rounded transition-all ${
                  executionMode === "Agent"
                    ? "bg-charcoal-700 text-charcoal-100 font-bold"
                    : "text-charcoal-400 hover:text-charcoal-200"
                }`}
              >
                Agent
              </button>
              <button
                type="button"
                onClick={() => setExecutionMode("Manual tool")}
                className={`px-2 py-0.5 rounded transition-all ${
                  executionMode === "Manual tool"
                    ? "bg-charcoal-700 text-charcoal-100 font-bold"
                    : "text-charcoal-400 hover:text-charcoal-200"
                }`}
              >
                Manual tool
              </button>
            </div>

            {/* Model selector pill */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowModelPicker(!showModelPicker)}
                className="px-2 py-1 rounded bg-charcoal-800 border border-charcoal-700 hover:border-charcoal-600 text-[10px] text-charcoal-300 flex items-center gap-1.5 transition-colors"
              >
                <Cpu className="w-3 h-3 text-coral-accent" />
                <span className="truncate max-w-[130px]">{selectedModel}</span>
                <ChevronDown className="w-3 h-3 text-charcoal-400" />
              </button>

              {showModelPicker && (
                <div className="absolute right-0 bottom-full mb-1 w-60 bg-charcoal-800 border border-charcoal-600 rounded shadow-xl z-50 p-1 space-y-0.5">
                  {models.map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => {
                        setSelectedModel(m);
                        setShowModelPicker(false);
                      }}
                      className={`w-full text-left px-2 py-1.5 rounded text-[10px] transition-colors ${
                        selectedModel === m
                          ? "bg-charcoal-700 text-coral-accent font-semibold"
                          : "text-charcoal-300 hover:bg-charcoal-750 hover:text-charcoal-100"
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Primary Coral Send Button */}
            <button
              type="submit"
              disabled={!inputText.trim() || isProcessing}
              className="p-1.5 rounded bg-coral-accent hover:bg-coral-hover text-charcoal-950 disabled:opacity-30 transition-all font-bold"
              title="Execute Query"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </form>
  );
};
