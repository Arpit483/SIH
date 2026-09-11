import { ConfidenceTier, InputConfiguration, SensorType } from "../types/satquery";

export const THEME_CONFIG = {
  colors: {
    bgApp: "#0d0f12",
    bgPanel: "#14171c",
    bgSurface: "#1a1d24",
    bgActive: "#22262f",
    border: "#252b36",
    borderSubtle: "#1c212a",
    textPrimary: "#e2e6ed",
    textSecondary: "#8f9aa9",
    textMuted: "#5a6474",
    coralAccent: "#e07856",
  },
  confidenceBadges: {
    HIGH: {
      tier: "HIGH" as ConfidenceTier,
      label: "CONFIDENCE: HIGH",
      color: "#e07856",
      bgColor: "rgba(224, 120, 86, 0.12)",
      borderColor: "rgba(224, 120, 86, 0.35)",
      dotColor: "#e07856",
      iconColor: "text-coral-accent",
    },
    MEDIUM: {
      tier: "MEDIUM" as ConfidenceTier,
      label: "CONFIDENCE: MED",
      color: "#d99b32",
      bgColor: "rgba(217, 155, 50, 0.12)",
      borderColor: "rgba(217, 155, 50, 0.35)",
      dotColor: "#d99b32",
      iconColor: "text-status-amber",
    },
    LOW: {
      tier: "LOW" as ConfidenceTier,
      label: "CONFIDENCE: LOW",
      color: "#d45353",
      bgColor: "rgba(212, 83, 83, 0.12)",
      borderColor: "rgba(212, 83, 83, 0.35)",
      dotColor: "#d45353",
      iconColor: "text-status-red",
    },
  },
  sensorBadges: {
    sentinel2: {
      label: "Sentinel-2 (MSI)",
      type: "Optical",
      bands: "12 Bands",
      res: "10m",
      tag: "OPTICAL",
    },
    landsat8: {
      label: "Landsat-8 (OLI)",
      type: "Optical",
      bands: "11 Bands",
      res: "15-30m",
      tag: "OPTICAL",
    },
    sentinel1_sar: {
      label: "Sentinel-1 (C-SAR)",
      type: "SAR",
      bands: "Dual Pol (VV/VH)",
      res: "10m",
      tag: "SAR",
    },
    cartosat3: {
      label: "Cartosat-3 (PAN/MX)",
      type: "Optical VHR",
      bands: "4 Bands + PAN",
      res: "0.28m PAN",
      tag: "ISRO VHR",
    },
    risat1_sar: {
      label: "RISAT-1A (C-Band)",
      type: "SAR ISRO",
      bands: "Circular Pol",
      res: "3-25m",
      tag: "ISRO SAR",
    },
    planetscope: {
      label: "PlanetScope",
      type: "Optical",
      bands: "8 Bands",
      res: "3m",
      tag: "OPTICAL",
    },
  },
  configBadges: {
    "Single Optical": {
      label: "Single Optical Scene",
      description: "Monoscopic VNIR analysis & zero-shot semantic grounding",
    },
    "Single SAR": {
      label: "Single SAR Scene",
      description: "Coherent radar backscatter & polarimetric filtering",
    },
    "Cross-Modal Pair": {
      label: "Cross-Modal Pair (Optical + SAR)",
      description: "All-weather fusion with deep cloud penetration",
    },
    "Bi-Temporal Pair": {
      label: "Bi-Temporal Pair (Pre/Post)",
      description: "Differential change detection & damage segmentation",
    },
  },
};
