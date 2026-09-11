"use client";

import React, { useState, useEffect, useCallback } from "react";
import { SidebarNav, NavSection } from "../components/Navigation/SidebarNav";
import { UploadZone } from "../components/UploadZone";
import { MapContainerWrapper } from "../components/MapCanvas/MapContainerWrapper";
import { ChatPanel } from "../components/ChatInterface/ChatPanel";
import { ReportView } from "../components/Report/ReportView";
import { MOCK_SCENARIOS, findMatchingScenario } from "../constants/mockScenarios";
import { 
  ChatMessage, 
  AssistantResponse, 
  UserMessage, 
  MissionScenario, 
  GeoTIFFMetadata,
  ExecutionTraceStep,
  InputConfiguration,
  GroundingBox
} from "../types/satquery";
import { checkBackendHealth, executeAgentQuery, uploadSatelliteImages } from "../lib/api";

export default function Home() {
  const [currentScenarioIndex, setCurrentScenarioIndex] = useState<number>(0);
  const scenario = MOCK_SCENARIOS[currentScenarioIndex];

  const [files, setFiles] = useState<GeoTIFFMetadata[]>(scenario.files);
  const [activeSection, setActiveSection] = useState<NavSection>("chat");

  // Backend connection state
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [backendDevice, setBackendDevice] = useState<string>("mps");

  // Map state
  const [mapCenter, setMapCenter] = useState<[number, number]>(scenario.files[0]?.center || [17.695, 83.300]);
  const [mapZoom, setMapZoom] = useState<number>(14);
  const [groundingBoxes, setGroundingBoxes] = useState<GroundingBox[]>([]);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [hasSwipe, setHasSwipe] = useState<boolean>(false);
  const [preImageUrl, setPreImageUrl] = useState<string | undefined>(undefined);
  const [postImageUrl, setPostImageUrl] = useState<string | undefined>(undefined);

  // Chat & Agent Pipeline State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [currentRunningStep, setCurrentRunningStep] = useState<ExecutionTraceStep | null>(null);
  const [latestResponse, setLatestResponse] = useState<AssistantResponse | null>(null);

  // Probe backend health
  useEffect(() => {
    const probe = async () => {
      const health = await checkBackendHealth();
      setBackendOnline(health.online);
      if (health.data?.device) {
        setBackendDevice(health.data.device);
      }
    };
    probe();
    const interval = setInterval(probe, 5000);
    return () => clearInterval(interval);
  }, []);

  const computeInputConfig = (currentFiles: GeoTIFFMetadata[]): InputConfiguration => {
    if (currentFiles.length === 0) return "Single Optical";
    if (currentFiles.length === 1) {
      return currentFiles[0].data_type === "SAR" ? "Single SAR" : "Single Optical";
    }
    const hasOptical = currentFiles.some((f) => f.data_type === "Optical");
    const hasSAR = currentFiles.some((f) => f.data_type === "SAR");
    if (hasOptical && hasSAR) {
      return "Cross-Modal Pair";
    }
    return "Bi-Temporal Pair";
  };

  const inputConfig = computeInputConfig(files);

  const handleSelectScenario = useCallback((newScenario: MissionScenario) => {
    const idx = MOCK_SCENARIOS.findIndex((s) => s.id === newScenario.id);
    setCurrentScenarioIndex(idx !== -1 ? idx : 0);
    setFiles(newScenario.files);
    setMapCenter(newScenario.files[0]?.center || [17.695, 83.300]);
    setMapZoom(newScenario.id === "scenario-2-flood-change" ? 12 : 14);
    setGroundingBoxes([]);
    setShowHeatmap(false);
    setHasSwipe(false);
    setPreImageUrl(undefined);
    setPostImageUrl(undefined);
    setMessages([]);
    setLatestResponse(null);
  }, []);

  const handleSelectPreset = (scenarioIndex: number) => {
    const targetScenario = MOCK_SCENARIOS[scenarioIndex];
    if (targetScenario) {
      handleSelectScenario(targetScenario);
    }
  };

  const handleSendMessage = async (queryText: string) => {
    if (isProcessing) return;

    const userMsg: UserMessage = {
      id: `usr_${Date.now()}`,
      sender: "user",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      text: queryText,
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsProcessing(true);

    const hasSarFile = files.some((f) => f.data_type === "SAR");
    const matchedScenario = findMatchingScenario(queryText, files.length, hasSarFile);

    const targetIdx = MOCK_SCENARIOS.findIndex((s) => s.id === matchedScenario.id);
    if (targetIdx !== -1 && targetIdx !== currentScenarioIndex) {
      setCurrentScenarioIndex(targetIdx);
      if (files.length === 0 || files[0].id.startsWith("custom_") === false) {
        setFiles(matchedScenario.files);
      }
    }

    let answerText = matchedScenario.response.text;
    let confidenceTier = matchedScenario.response.confidence_tier;
    let confidenceScore = matchedScenario.response.confidence_score;
    let traceSteps: ExecutionTraceStep[] = [];
    let boxes: GroundingBox[] = matchedScenario.response.grounding_boxes || [];
    let changeStats = matchedScenario.response.change_stats;
    let hasHeatmap = matchedScenario.response.has_heatmap;
    let hasSwipeComp = matchedScenario.response.has_swipe_comparison;

    // Try Live FastAPI Backend if available
    let usedLiveBackend = false;
    if (backendOnline) {
      try {
        const filePaths = files.map((f) => f.id);
        const liveRes = await executeAgentQuery(queryText, filePaths);
        if (liveRes.status === "success") {
          usedLiveBackend = true;
          answerText = liveRes.answer;
          confidenceTier = liveRes.tier;
          confidenceScore = Math.round(liveRes.confidence * 100);
          
          if (liveRes.execution_trace) {
            traceSteps = [
              {
                step: 1,
                tool: liveRes.execution_trace.selected_tool,
                tool_display_name: liveRes.execution_trace.model_name,
                params: liveRes.execution_trace.permitted_parameters || {},
                duration_ms: Math.round(liveRes.execution_trace.latency_seconds * 1000),
                status: "success",
                output: {
                  task: liveRes.execution_trace.selected_task,
                  confidence: liveRes.execution_trace.confidence,
                  orchestrator: liveRes.execution_trace.orchestrator,
                }
              }
            ];
          }

          if (liveRes.bounding_boxes && liveRes.bounding_boxes.length > 0) {
            boxes = liveRes.bounding_boxes.map((b, i) => ({
              id: `box_live_${i}`,
              label: queryText.slice(0, 24),
              confidence: confidenceScore,
              category: "anomaly" as const,
              bbox: [
                mapCenter[0] - 0.01 + (b[0] || 0) * 0.02,
                mapCenter[1] - 0.01 + (b[1] || 0) * 0.02,
                mapCenter[0] - 0.01 + (b[2] || 0.02) * 0.02,
                mapCenter[1] - 0.01 + (b[3] || 0.02) * 0.02,
              ],
            }));
          }
        }
      } catch (err) {
        console.warn("Live backend query returned error, falling back to simulated engine:", err);
      }
    }

    // If offline or fallback needed, simulate sequential tool execution
    if (!usedLiveBackend) {
      traceSteps = [];
      for (const step of matchedScenario.response.execution_trace) {
        setCurrentRunningStep({
          ...step,
          status: "running",
        });
        await new Promise((resolve) => setTimeout(resolve, step.duration_ms || 400));
        traceSteps.push({
          ...step,
          status: "success",
        });
      }
    }

    setCurrentRunningStep(null);

    const responseId = `asst_${Date.now()}`;
    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

    const newAssistantResponse: AssistantResponse = {
      id: responseId,
      sender: "assistant",
      timestamp,
      text: "",
      confidence_tier: confidenceTier,
      confidence_score: confidenceScore,
      scenario_id: matchedScenario.id,
      execution_trace: traceSteps,
      grounding_boxes: boxes,
      change_stats: changeStats,
      fusion_note: matchedScenario.response.fusion_note,
      has_heatmap: hasHeatmap,
      has_swipe_comparison: hasSwipeComp,
      pre_image_url: matchedScenario.response.pre_image_url,
      post_image_url: matchedScenario.response.post_image_url,
    };

    // Simulated token-by-token reveal
    const words = answerText.split(" ");
    let currentRevealed = "";

    setMessages((prev) => [...prev, newAssistantResponse]);

    for (let i = 0; i < words.length; i += 3) {
      const chunk = words.slice(i, i + 3).join(" ") + " ";
      currentRevealed += chunk;
      
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === responseId
            ? { ...(msg as AssistantResponse), text: currentRevealed }
            : msg
        )
      );
      await new Promise((resolve) => setTimeout(resolve, 30));
    }

    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === responseId
          ? { ...(msg as AssistantResponse), text: answerText }
          : msg
      )
    );

    // Update map state
    if (matchedScenario.files.length > 0) {
      setMapCenter(matchedScenario.files[0].center);
      setMapZoom(matchedScenario.id === "scenario-2-flood-change" ? 12 : 14);
    }
    if (boxes) {
      setGroundingBoxes(boxes);
    }
    if (hasHeatmap) {
      setShowHeatmap(true);
    }
    if (hasSwipeComp) {
      setHasSwipe(true);
      setPreImageUrl(matchedScenario.response.pre_image_url);
      setPostImageUrl(matchedScenario.response.post_image_url);
    } else {
      setHasSwipe(false);
    }

    setLatestResponse({ ...newAssistantResponse, text: answerText });
    setIsProcessing(false);
  };

  return (
    <div className="flex h-screen w-screen bg-charcoal-950 text-charcoal-100 overflow-hidden font-sans select-none">
      {/* Leftmost Sidebar Nav (Odysseus Navigation Pattern) */}
      <SidebarNav
        activeSection={activeSection}
        onSelectSection={setActiveSection}
        isProcessing={isProcessing}
        filesCount={files.length}
        currentScenario={scenario}
        onSelectScenario={handleSelectScenario}
        backendOnline={backendOnline}
        backendDevice={backendDevice}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {activeSection === "reports" ? (
          <ReportView
            scenario={scenario}
            files={files}
            latestResponse={latestResponse}
            onBackToChat={() => setActiveSection("chat")}
          />
        ) : (
          <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden">
            {/* Panel 1: Left Ingestion Zone */}
            <div className="w-full lg:w-[300px] xl:w-[340px] h-[260px] lg:h-full flex-shrink-0">
              <UploadZone
                files={files}
                onFilesChange={(newFiles) => setFiles(newFiles)}
                inputConfig={inputConfig}
                onSelectPreset={handleSelectPreset}
              />
            </div>

            {/* Panel 2: Center Map Canvas (Dominant Panel) */}
            <div className="flex-1 h-[380px] lg:h-full relative min-h-0 bg-charcoal-950">
              <MapContainerWrapper
                center={mapCenter}
                zoom={mapZoom}
                files={files}
                groundingBoxes={groundingBoxes}
                showHeatmap={showHeatmap}
                onToggleHeatmap={() => setShowHeatmap(!showHeatmap)}
                hasSwipeComparison={hasSwipe}
                preImageUrl={preImageUrl}
                postImageUrl={postImageUrl}
                activeScenarioId={scenario.id}
              />
            </div>

            {/* Panel 3: Right Conversational Feed */}
            <div className="w-full lg:w-[380px] xl:w-[420px] h-[360px] lg:h-full flex-shrink-0">
              <ChatPanel
                messages={messages}
                onSendMessage={handleSendMessage}
                isProcessing={isProcessing}
                currentRunningStep={currentRunningStep}
                currentScenario={scenario}
                onOpenReport={() => setActiveSection("reports")}
                files={files}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
