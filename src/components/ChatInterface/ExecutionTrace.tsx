"use client";

import React, { useState } from "react";
import { 
  ChevronDown, 
  ChevronRight, 
  Terminal, 
  Check, 
  AlertCircle, 
  Clock, 
  Code2,
  Loader2
} from "lucide-react";
import { ExecutionTraceStep } from "../../types/satquery";

interface ExecutionTraceProps {
  steps: ExecutionTraceStep[];
  isStreaming?: boolean;
}

export const ExecutionTrace: React.FC<ExecutionTraceProps> = ({ steps }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  const totalDuration = steps.reduce((acc, step) => acc + (step.duration_ms || 0), 0);

  const toggleStep = (stepNumber: number) => {
    setExpandedStep(expandedStep === stepNumber ? null : stepNumber);
  };

  return (
    <div className="border border-charcoal-700 rounded bg-charcoal-850 overflow-hidden font-mono text-xs select-none">
      {/* Accordion Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2 bg-charcoal-800 hover:bg-charcoal-750 flex items-center justify-between transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          {isOpen ? (
            <ChevronDown className="w-3.5 h-3.5 text-charcoal-300" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-charcoal-400" />
          )}
          <span className="font-semibold text-charcoal-200 flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5 text-coral-accent" />
            Execution Trace
          </span>
          <span className="text-[10px] px-1.5 py-0.2 rounded bg-charcoal-700 text-charcoal-300 border border-charcoal-600">
            {steps.length} tools
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-charcoal-400">
          <Clock className="w-3 h-3 text-charcoal-400" />
          <span>{totalDuration}ms</span>
        </div>
      </button>

      {/* Accordion Content */}
      {isOpen && (
        <div className="p-2 space-y-1.5 bg-charcoal-900 border-t border-charcoal-700">
          {steps.map((step) => {
            const isStepExpanded = expandedStep === step.step;

            return (
              <div
                key={step.step}
                className="border border-charcoal-700 rounded bg-charcoal-850 overflow-hidden"
              >
                {/* Step Header */}
                <div
                  onClick={() => toggleStep(step.step)}
                  className="px-2.5 py-1.5 flex items-center justify-between cursor-pointer hover:bg-charcoal-800 text-[11px]"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="w-3.5 h-3.5 rounded bg-charcoal-700 text-charcoal-300 flex items-center justify-center text-[9px] font-bold">
                      {step.step}
                    </span>
                    {step.status === "running" ? (
                      <Loader2 className="w-3.5 h-3.5 text-status-amber animate-spin flex-shrink-0" />
                    ) : step.status === "success" ? (
                      <Check className="w-3.5 h-3.5 text-status-green flex-shrink-0" />
                    ) : (
                      <AlertCircle className="w-3.5 h-3.5 text-status-red flex-shrink-0" />
                    )}
                    <span className="font-semibold text-charcoal-200 truncate">
                      {step.tool}
                    </span>
                    <span className="text-[10px] text-charcoal-400 hidden sm:inline">
                      ({step.tool_display_name})
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-[10px] text-charcoal-400 flex-shrink-0">
                    <span>{step.duration_ms}ms</span>
                    {isStepExpanded ? (
                      <ChevronDown className="w-3 h-3 text-charcoal-400" />
                    ) : (
                      <ChevronRight className="w-3 h-3 text-charcoal-400" />
                    )}
                  </div>
                </div>

                {/* Step Parameter & Output Payloads */}
                {isStepExpanded && (
                  <div className="p-2.5 bg-charcoal-950 border-t border-charcoal-750 space-y-2 text-[10px]">
                    <div>
                      <div className="text-charcoal-400 font-semibold mb-1 flex items-center gap-1">
                        <Code2 className="w-3 h-3 text-coral-accent" />
                        INPUT PARAMS:
                      </div>
                      <pre className="p-2 bg-charcoal-900 border border-charcoal-750 rounded text-charcoal-300 overflow-x-auto leading-relaxed">
                        {JSON.stringify(step.params, null, 2)}
                      </pre>
                    </div>

                    {step.output && (
                      <div>
                        <div className="text-charcoal-400 font-semibold mb-1 flex items-center gap-1">
                          <Check className="w-3 h-3 text-status-green" />
                          OUTPUT:
                        </div>
                        <pre className="p-2 bg-charcoal-900 border border-charcoal-750 rounded text-charcoal-200 overflow-x-auto leading-relaxed">
                          {JSON.stringify(step.output, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
