"use client";

import React from "react";
import { 
  UploadCloud, 
  Map as MapIcon, 
  MessageSquare, 
  FileText, 
  Activity, 
  Terminal,
  Radio,
  Satellite,
  Compass
} from "lucide-react";
import { MissionScenario } from "../../types/satquery";
import { MOCK_SCENARIOS } from "../../constants/mockScenarios";

export type NavSection = "upload" | "map" | "chat" | "reports" | "trace";

interface SidebarNavProps {
  activeSection: NavSection;
  onSelectSection: (section: NavSection) => void;
  isProcessing: boolean;
  filesCount: number;
  currentScenario: MissionScenario;
  onSelectScenario: (scenario: MissionScenario) => void;
}

export const SidebarNav: React.FC<SidebarNavProps> = ({
  activeSection,
  onSelectSection,
  isProcessing,
  filesCount,
  currentScenario,
  onSelectScenario,
}) => {
  const navItems: {
    id: NavSection;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    stateText?: string;
    stateDotColor?: string;
  }[] = [
    {
      id: "upload",
      label: "Upload",
      icon: UploadCloud,
      stateText: filesCount > 0 ? `${filesCount} loaded` : "empty",
      stateDotColor: filesCount > 0 ? "bg-charcoal-400" : "bg-charcoal-600",
    },
    {
      id: "map",
      label: "Map",
      icon: MapIcon,
      stateText: "synced",
      stateDotColor: "bg-charcoal-400",
    },
    {
      id: "chat",
      label: "Chat",
      icon: MessageSquare,
      stateText: isProcessing ? "running" : "idle",
      stateDotColor: isProcessing ? "bg-coral-accent animate-pulse" : "bg-charcoal-500",
    },
    {
      id: "reports",
      label: "Reports",
      icon: FileText,
      stateText: "ready",
      stateDotColor: "bg-charcoal-500",
    },
    {
      id: "trace",
      label: "Trace",
      icon: Terminal,
      stateText: isProcessing ? "active" : "standby",
      stateDotColor: isProcessing ? "bg-coral-accent" : "bg-charcoal-600",
    },
  ];

  return (
    <aside className="w-56 h-full bg-charcoal-900 border-r border-charcoal-700 flex flex-col justify-between p-4 select-none font-mono text-xs flex-shrink-0 z-30">
      {/* Brand & Wordmark */}
      <div className="space-y-6">
        <div className="flex items-center gap-2.5 px-2">
          <div className="w-6 h-6 rounded bg-charcoal-800 border border-charcoal-600 flex items-center justify-center text-coral-accent">
            <Satellite className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="font-bold text-sm tracking-wide text-charcoal-100 flex items-center gap-1">
              SatQuery <span className="text-coral-accent font-extrabold">AI</span>
            </div>
            <div className="text-[10px] text-charcoal-400 tracking-tight">
              ISRO EO SIH26167
            </div>
          </div>
        </div>

        {/* Primary Navigation - Generous Vertical Spacing */}
        <nav className="space-y-2 pt-2">
          {navItems.map((item) => {
            const isActive = activeSection === item.id;
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                onClick={() => onSelectSection(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded transition-all group text-left ${
                  isActive
                    ? "bg-charcoal-800 text-charcoal-100 border-l-2 border-coral-accent font-semibold"
                    : "text-charcoal-300 hover:text-charcoal-100 hover:bg-charcoal-850"
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon
                    className={`w-4 h-4 transition-colors ${
                      isActive ? "text-coral-accent" : "text-charcoal-400 group-hover:text-charcoal-200"
                    }`}
                  />
                  <span className="tracking-wide text-xs">{item.label}</span>
                </div>

                {item.stateText && (
                  <div className="flex items-center gap-1.5 text-[10px] text-charcoal-400">
                    <span className={`w-1.5 h-1.5 rounded-full ${item.stateDotColor}`} />
                    <span className="lowercase">{item.stateText}</span>
                  </div>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Preset Switcher & Station Telemetry */}
      <div className="space-y-3 pt-4 border-t border-charcoal-700 text-[11px]">
        <div className="px-2 text-[10px] text-charcoal-400 uppercase tracking-wider">
          Preset Mission
        </div>
        <div className="space-y-1">
          {MOCK_SCENARIOS.map((s, idx) => {
            const isSelected = currentScenario.id === s.id;
            return (
              <button
                key={s.id}
                onClick={() => onSelectScenario(s)}
                className={`w-full text-left px-2 py-1.5 rounded transition-all text-[11px] truncate flex items-center justify-between ${
                  isSelected
                    ? "bg-charcoal-800 text-coral-accent font-medium border border-charcoal-600"
                    : "text-charcoal-300 hover:text-charcoal-100 hover:bg-charcoal-850"
                }`}
                title={s.description}
              >
                <span className="truncate">S0{idx + 1} {s.config_type.replace(" Pair", "").replace(" Scene", "")}</span>
                {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-coral-accent flex-shrink-0 ml-1" />}
              </button>
            );
          })}
        </div>

        <div className="px-2 pt-2 border-t border-charcoal-800 flex items-center justify-between text-[10px] text-charcoal-400">
          <span>NRSC-HYD-NODE</span>
          <span className="text-charcoal-300">ONLINE</span>
        </div>
      </div>
    </aside>
  );
};
