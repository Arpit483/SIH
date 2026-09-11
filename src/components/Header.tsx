"use client";

import React, { useState, useEffect } from "react";
import { Satellite, Radio, Cpu, ShieldCheck, Activity, Terminal, FileText, Sparkles } from "lucide-react";
import { MOCK_SCENARIOS } from "../constants/mockScenarios";
import { MissionScenario } from "../types/satquery";

interface HeaderProps {
  currentScenario: MissionScenario;
  onSelectScenario: (scenario: MissionScenario) => void;
  activeTab: "chat" | "report";
  onTabChange: (tab: "chat" | "report") => void;
  onReset: () => void;
  isProcessing: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentScenario,
  onSelectScenario,
  activeTab,
  onTabChange,
  onReset,
  isProcessing,
}) => {
  const [currentTime, setCurrentTime] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTime(now.toISOString().replace("T", " ").substring(0, 19) + " UTC");
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-panel-border bg-[#090e1a]/95 backdrop-blur px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs z-30 select-none">
      {/* Brand & Project Identity */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded bg-cyan-950/70 border border-cyan-500/50 text-cyan-accent shadow-[0_0_12px_rgba(0,240,255,0.3)]">
          <Satellite className="w-4 h-4 animate-pulse" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold tracking-wider text-sm text-white font-mono flex items-center gap-1.5">
              SATQUERY <span className="text-cyan-accent font-extrabold">AI</span>
            </span>
            <span className="px-1.5 py-0.5 rounded bg-amber-950/60 border border-amber-500/40 text-amber-300 font-mono text-[10px] tracking-wide font-medium">
              SIH26167
            </span>
            <span className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-blue-950/60 border border-blue-500/30 text-blue-300 text-[10px] font-mono">
              <ShieldCheck className="w-3 h-3 text-blue-400" />
              ISRO EO SCIENTIST SUITE
            </span>
          </div>
          <p className="text-[10px] text-slate-400 font-mono tracking-tight hidden md:block">
            Autonomous Agentic Remote-Sensing VQA & Multimodal Grounding Engine
          </p>
        </div>
      </div>

      {/* Preset Scenario Quick Switcher */}
      <div className="flex items-center gap-2 bg-[#0d1527] border border-panel-border rounded-lg p-1">
        <span className="text-[10px] text-slate-400 font-mono px-2 hidden lg:inline flex items-center gap-1">
          <Terminal className="w-3 h-3 text-cyan-accent" />
          SCENARIOS:
        </span>
        {MOCK_SCENARIOS.map((scenario, index) => {
          const isActive = currentScenario.id === scenario.id;
          return (
            <button
              key={scenario.id}
              onClick={() => onSelectScenario(scenario)}
              className={`px-2.5 py-1 rounded font-mono text-[11px] transition-all flex items-center gap-1.5 ${
                isActive
                  ? "bg-cyan-500/20 text-cyan-accent border border-cyan-500/60 shadow-[0_0_8px_rgba(0,240,255,0.2)]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent"
              }`}
              title={scenario.description}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isActive ? "bg-cyan-accent animate-ping" : "bg-slate-600"}`} />
              S0{index + 1}: {scenario.config_type.replace(" Pair", "").replace(" Scene", "")}
            </button>
          );
        })}
      </div>

      {/* Right Controls & Telemetry HUD */}
      <div className="flex items-center gap-3">
        {/* Live System Status */}
        <div className="hidden xl:flex items-center gap-3 font-mono text-[11px] text-slate-400 border-r border-panel-border pr-3">
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${isProcessing ? "bg-amber-400 animate-spin" : "bg-emerald-400 shadow-[0_0_6px_#10b981]"}`} />
            <span className="text-slate-300">
              AGENT: {isProcessing ? "COMPUTING" : "STANDBY"}
            </span>
          </div>
          <div className="flex items-center gap-1 text-slate-400">
            <Cpu className="w-3 h-3 text-cyan-accent" />
            <span>CUDA TENSOR-RT</span>
          </div>
          <span className="text-slate-500">|</span>
          <span className="text-slate-300">{currentTime || "LIVE TELEMETRY"}</span>
        </div>

        {/* View Toggle Tabs (Chat / Mission Report) */}
        <div className="flex items-center bg-[#070c18] border border-panel-border rounded-lg p-0.5">
          <button
            onClick={() => onTabChange("chat")}
            className={`px-3 py-1 rounded text-xs font-mono transition-all flex items-center gap-1.5 ${
              activeTab === "chat"
                ? "bg-cyan-950/80 text-cyan-accent border border-cyan-500/50"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Terminal className="w-3 h-3" />
            Agent Chat
          </button>
          <button
            onClick={() => onTabChange("report")}
            className={`px-3 py-1 rounded text-xs font-mono transition-all flex items-center gap-1.5 ${
              activeTab === "report"
                ? "bg-amber-950/80 text-amber-300 border border-amber-500/50"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileText className="w-3 h-3" />
            Mission Report
          </button>
        </div>

        {/* Reset Button */}
        <button
          onClick={onReset}
          className="px-2.5 py-1 text-xs font-mono text-slate-400 hover:text-white hover:bg-slate-800 rounded border border-panel-border transition-colors"
          title="Reset current session to initial state"
        >
          Reset
        </button>
      </div>
    </header>
  );
};
