"use client";

import React, { useRef, useEffect } from "react";
import { 
  Bot, 
  User, 
  Terminal, 
  Sparkles, 
  FileText, 
  Radio, 
  Check, 
  ChevronRight, 
  Loader2,
  Satellite
} from "lucide-react";
import { 
  ChatMessage, 
  AssistantResponse, 
  UserMessage, 
  MissionScenario, 
  GeoTIFFMetadata,
  ExecutionTraceStep
} from "../../types/satquery";
import { THEME_CONFIG } from "../../constants/theme";
import { ExecutionTrace } from "./ExecutionTrace";
import { ChatComposer } from "./ChatComposer";

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (query: string) => void;
  isProcessing: boolean;
  currentRunningStep?: ExecutionTraceStep | null;
  currentScenario: MissionScenario;
  onOpenReport: () => void;
  files: GeoTIFFMetadata[];
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  onSendMessage,
  isProcessing,
  currentRunningStep,
  currentScenario,
  onOpenReport,
  files,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing, currentRunningStep]);

  const quickPrompts = [
    {
      label: "Detect vessels & tanks",
      query: "Detect and isolate all maritime vessels, cargo berths, and fuel storage tanks in the port area.",
    },
    {
      label: "Assess flood inundation area",
      query: "Calculate total flood inundation area, compare pre/post flood boundaries, and quantify agricultural submergence.",
    },
    {
      label: "SAR all-weather cloud penetration",
      query: "Penetrate monsoon cloud cover using SAR cross-modal fusion to detect hidden maritime vessels and slick anomalies.",
    },
  ];

  return (
    <div className="flex flex-col h-full bg-charcoal-900 border-l border-charcoal-700 overflow-hidden">
      {/* Panel Top Header */}
      <div className="p-3 border-b border-charcoal-700 bg-charcoal-850 flex items-center justify-between font-mono text-xs select-none">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-coral-accent" />
          <span className="font-bold text-charcoal-100 tracking-wide">
            Conversational Agent Feed
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-charcoal-400">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isProcessing ? "bg-coral-accent animate-pulse" : "bg-charcoal-500"
            }`}
          />
          <span>{isProcessing ? "running" : "idle"}</span>
        </div>
      </div>

      {/* Suggested Query Chips */}
      <div className="px-3 py-2 bg-charcoal-900 border-b border-charcoal-800 space-y-1 select-none">
        <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar py-0.5">
          {quickPrompts.map((item, idx) => (
            <button
              key={idx}
              onClick={() => onSendMessage(item.query)}
              disabled={isProcessing}
              className="px-2 py-1 rounded bg-charcoal-850 hover:bg-charcoal-800 border border-charcoal-700 text-[10px] font-mono text-charcoal-300 hover:text-charcoal-100 whitespace-nowrap transition-colors flex items-center gap-1 disabled:opacity-40"
            >
              <Sparkles className="w-2.5 h-2.5 text-coral-accent" />
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Message Feed Area */}
      <div className="flex-1 p-3 overflow-y-auto space-y-4">
        {messages.length === 0 ? (
          /* Empty State per Odysseus style specification */
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3 font-mono select-none">
            <div className="w-10 h-10 rounded bg-charcoal-800 border border-charcoal-700 flex items-center justify-center text-coral-accent">
              <Satellite className="w-5 h-5" />
            </div>
            <div className="space-y-1">
              <h2 className="text-sm font-bold tracking-wide text-charcoal-100">
                SatQuery <span className="text-coral-accent">AI</span>
              </h2>
              <p className="text-xs text-charcoal-400">
                One prompt, any satellite
              </p>
            </div>
            <p className="text-[11px] text-charcoal-500 max-w-[240px] pt-2">
              Tip: Drag a GeoTIFF pair here to begin, or pick a prompt above
            </p>
          </div>
        ) : (
          messages.map((msg) => {
            if (msg.sender === "user") {
              return (
                <div key={msg.id} className="flex justify-end pl-8">
                  <div className="bg-charcoal-800 border border-charcoal-700 text-charcoal-100 rounded p-3 max-w-[92%] space-y-1">
                    <div className="flex items-center justify-between gap-3 text-[10px] font-mono text-charcoal-400">
                      <span className="font-semibold text-charcoal-300 flex items-center gap-1">
                        <User className="w-3 h-3 text-coral-accent" /> Scientist
                      </span>
                      <span>{msg.timestamp}</span>
                    </div>
                    <p className="text-xs font-sans leading-relaxed text-charcoal-200">{msg.text}</p>
                  </div>
                </div>
              );
            }

            // Assistant Response
            const asstMsg = msg as AssistantResponse;
            const badgeConfig = THEME_CONFIG.confidenceBadges[asstMsg.confidence_tier] || THEME_CONFIG.confidenceBadges.HIGH;

            return (
              <div key={asstMsg.id} className="flex justify-start pr-2 space-y-2">
                <div className="w-full bg-charcoal-850 border border-charcoal-700 rounded p-3.5 space-y-3">
                  {/* Header & Confidence Tier */}
                  <div className="flex items-center justify-between border-b border-charcoal-700/80 pb-2">
                    <div className="flex items-center gap-2">
                      <div className="w-5 h-5 rounded bg-charcoal-800 border border-charcoal-600 flex items-center justify-center text-coral-accent">
                        <Bot className="w-3 h-3" />
                      </div>
                      <span className="font-mono text-xs font-bold text-charcoal-100">
                        SatQuery Reasoner
                      </span>
                    </div>

                    {/* Shared Theme Constant Badge */}
                    <div
                      className="px-2 py-0.5 rounded border text-[10px] font-mono font-semibold flex items-center gap-1.5"
                      style={{
                        backgroundColor: badgeConfig.bgColor,
                        borderColor: badgeConfig.borderColor,
                        color: badgeConfig.color,
                      }}
                    >
                      <span
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: badgeConfig.dotColor }}
                      />
                      {badgeConfig.label} ({asstMsg.confidence_score}%)
                    </div>
                  </div>

                  {/* Prose Body (Plain Sans-Serif per typography spec) */}
                  <div className="text-xs text-charcoal-200 leading-relaxed font-sans whitespace-pre-line space-y-2">
                    {asstMsg.text}
                  </div>

                  {/* SAR Fusion Notice */}
                  {asstMsg.fusion_note && (
                    <div className="p-2.5 rounded bg-charcoal-900 border border-charcoal-700 text-[11px] font-mono text-charcoal-300 flex items-start gap-2">
                      <Radio className="w-4 h-4 text-coral-accent flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-bold text-coral-accent text-[10px]">RADAR CROSS-MODAL FUSION:</div>
                        <p className="text-[10px] text-charcoal-300 font-sans mt-0.5">{asstMsg.fusion_note}</p>
                      </div>
                    </div>
                  )}

                  {/* Execution Trace Accordion */}
                  {asstMsg.execution_trace && asstMsg.execution_trace.length > 0 && (
                    <ExecutionTrace steps={asstMsg.execution_trace} />
                  )}

                  {/* Action Link to Report */}
                  <div className="pt-2 border-t border-charcoal-750 flex items-center justify-between text-xs font-mono">
                    <button
                      onClick={onOpenReport}
                      className="text-[11px] text-coral-accent hover:underline flex items-center gap-1"
                    >
                      <FileText className="w-3 h-3" />
                      Open Mission Report →
                    </button>
                    <span className="text-[10px] text-charcoal-500">
                      ISRO-SIH26167
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}

        {/* Live Running Tool Step Pulse */}
        {isProcessing && (
          <div className="flex justify-start pr-2">
            <div className="w-full bg-charcoal-850 border border-charcoal-700 rounded p-3 space-y-2 font-mono text-xs">
              <div className="flex items-center justify-between text-charcoal-300">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-3.5 h-3.5 text-coral-accent animate-spin" />
                  <span className="font-semibold text-charcoal-100">Executing Agent Tools...</span>
                </div>
                <span className="text-[10px] text-charcoal-400">~400ms</span>
              </div>

              {currentRunningStep && (
                <div className="p-2 rounded bg-charcoal-900 border border-charcoal-750 flex items-center gap-2 text-[11px]">
                  <span className="w-1.5 h-1.5 rounded-full bg-coral-accent animate-ping" />
                  <span className="text-charcoal-400">Step {currentRunningStep.step}:</span>
                  <span className="text-coral-accent font-semibold">{currentRunningStep.tool}</span>
                  <span className="text-[10px] text-charcoal-400 truncate">
                    ({currentRunningStep.tool_display_name})
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Composer Bar */}
      <ChatComposer
        onSendMessage={onSendMessage}
        isProcessing={isProcessing}
        onAttachClick={() => {}}
        onParamsClick={() => {}}
      />
    </div>
  );
};
