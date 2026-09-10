"""
unified_confidence_engine.py
============================
SatQuery AI — Unified Confidence Engine

Provides calibrated confidence scores for all task heads:
  • VQA          — generative open-ended answers (T5 / language decoder)
  • Grounding    — bounding-box predictions (GroundingDINO-style)
  • Change       — binary change-probability maps
  • Fusion       — weighted harmonic mean across multiple tasks

Also includes:
  • TemperatureScaler — post-hoc calibration via learned scalar T
  • compute_ece()     — Expected Calibration Error metric

Dependencies: numpy, torch (no additional libraries required).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_THRESHOLD: float = 0.80
MEDIUM_THRESHOLD: float = 0.55
_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Tier helpers
# ---------------------------------------------------------------------------

def _tier(score: float) -> Tuple[str, str]:
    """Return (tier_label, icon) for a confidence score in [0, 1]."""
    if score >= HIGH_THRESHOLD:
        return "HIGH", "🟢"
    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM", "🟡"
    return "LOW", "🔴"


def _build_output(confidence: float, extra: Dict) -> Dict:
    """Merge base confidence fields with task-specific extras."""
    tier_label, icon = _tier(confidence)
    base = {
        "confidence": float(confidence),
        "tier": tier_label,
        "icon": icon,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Temperature Scaler
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration via a single learned scalar temperature T.

    Usage
    -----
    scaler = TemperatureScaler()
    scaler.fit(logits_val, labels_val)          # learn T
    calibrated_logits = scaler.transform(logits) # apply at inference

    Parameters
    ----------
    init_temperature : float
        Starting value for T (default 1.0 = no scaling).
    max_iter : int
        LBFGS optimisation steps (default 50).
    lr : float
        Learning rate for LBFGS (default 0.01).
    """

    def __init__(
        self,
        init_temperature: float = 1.0,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32)
        )
        self.max_iter = max_iter
        self.lr = lr

    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by 1/T."""
        return logits / self.temperature.clamp(min=_EPS)

    # ------------------------------------------------------------------
    def fit(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
    ) -> "TemperatureScaler":
        """
        Fit temperature T by minimising NLL on validation logits/labels.

        Parameters
        ----------
        logits : Tensor | ndarray, shape (N, C)
            Raw (un-scaled) logits from the model.
        labels : Tensor | ndarray, shape (N,)
            Integer class labels.

        Returns
        -------
        self
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        logits = logits.detach()
        labels = labels.detach()

        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=self.lr, max_iter=self.max_iter
        )

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        logger.info(
            "TemperatureScaler fitted: T = %.4f", self.temperature.item()
        )
        return self

    # ------------------------------------------------------------------
    def transform(
        self, logits: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Apply learned temperature scaling.

        Parameters
        ----------
        logits : Tensor | ndarray, shape (..., C)

        Returns
        -------
        Tensor : calibrated logits, same shape as input.
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        with torch.no_grad():
            return self.forward(logits)


# ---------------------------------------------------------------------------
# ECE metric
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: Union[np.ndarray, List[float]],
    accuracies: Union[np.ndarray, List[float]],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Splits predictions into ``n_bins`` equal-width confidence bins and
    computes the weighted absolute gap between mean confidence and mean
    accuracy in each bin.

    Parameters
    ----------
    confidences : array-like, shape (N,)
        Predicted confidence / probability for the *chosen* class, in [0, 1].
    accuracies : array-like, shape (N,)
        Binary correctness indicators (1 = correct, 0 = incorrect).
    n_bins : int
        Number of calibration bins (default 10).

    Returns
    -------
    float
        ECE in [0, 1]; lower is better.

    Notes
    -----
    ECE = Σ_b (|B_b| / N) * |acc(B_b) − conf(B_b)|
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    if confidences.shape != accuracies.shape:
        raise ValueError(
            f"Shape mismatch: confidences {confidences.shape} vs "
            f"accuracies {accuracies.shape}"
        )

    n = len(confidences)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include upper boundary in last bin
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class UnifiedConfidenceEngine:
    """
    Unified Confidence Engine for SatQuery AI.

    Computes calibrated confidence scores for four task types:
      - VQA (generative answers from a language decoder)
      - Grounding (bounding-box predictions)
      - Change Detection (binary change-probability maps)
      - Fusion (combining scores from multiple tasks)

    All public methods return a dict with at minimum::

        {
            'confidence': float,   # [0, 1]
            'tier':       str,     # 'HIGH' | 'MEDIUM' | 'LOW'
            'icon':       str,     # '🟢'  | '🟡'    | '🔴'
            # …task-specific keys
        }

    Parameters
    ----------
    temperature : float
        Temperature T for VQA token-probability scaling (default 1.0).
    high_threshold : float
        Lower bound for HIGH tier (default 0.80).
    medium_threshold : float
        Lower bound for MEDIUM tier (default 0.55).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        high_threshold: float = HIGH_THRESHOLD,
        medium_threshold: float = MEDIUM_THRESHOLD,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    # ------------------------------------------------------------------
    # Internal tier helper (instance-aware thresholds)
    # ------------------------------------------------------------------

    def _tier(self, score: float) -> Tuple[str, str]:
        if score >= self.high_threshold:
            return "HIGH", "🟢"
        if score >= self.medium_threshold:
            return "MEDIUM", "🟡"
        return "LOW", "🔴"

    def _out(self, confidence: float, extra: Dict) -> Dict:
        tier_label, icon = self._tier(confidence)
        return {"confidence": float(confidence), "tier": tier_label, "icon": icon, **extra}

    # ==================================================================
    # 1.  VQA Confidence
    # ==================================================================

    def vqa_confidence(
        self,
        transition_scores: torch.Tensor,
        apply_temperature: bool = True,
    ) -> Dict:
        """
        Compute confidence for a generative VQA answer.

        Uses log-probabilities obtained via
        ``model.compute_transition_scores()``.

        Composite formula::

            C_vqa = 0.7 * geometric_mean_prob + 0.3 * min_token_prob

        where::

            geometric_mean_prob = exp( mean(log_probs / T) )
            min_token_prob      = min( exp(log_probs / T) )

        Parameters
        ----------
        transition_scores : torch.Tensor, shape (seq_len,) or (1, seq_len)
            Log-probabilities from ``model.compute_transition_scores(
                sequences, scores, normalize_logits=True)``.
            Must contain non-positive finite values.
        apply_temperature : bool
            Whether to divide log-probs by ``self.temperature`` (default True).

        Returns
        -------
        dict with keys: confidence, tier, icon, geometric_mean_prob,
                        min_token_prob, num_tokens, temperature
        """
        # --- normalise shape to 1-D ---
        ts = transition_scores
        if ts.dim() == 2:
            ts = ts.squeeze(0)  # (seq_len,)
        if ts.dim() != 1:
            raise ValueError(
                f"transition_scores must be 1-D or 2-D, got shape {transition_scores.shape}"
            )

        ts = ts.float().detach()

        # Filter out padding tokens (−inf or very large negative)
        valid_mask = ts > -1e9
        if valid_mask.sum() == 0:
            logger.warning("All transition scores are −inf; returning zero confidence.")
            return self._out(0.0, {"geometric_mean_prob": 0.0, "min_token_prob": 0.0,
                                   "num_tokens": 0, "temperature": self.temperature})

        log_probs = ts[valid_mask]  # already log-probs from normalize_logits=True

        if apply_temperature:
            log_probs = log_probs / self.temperature

        # Clamp to avoid numerical issues
        log_probs = log_probs.clamp(max=0.0)

        geom_mean_prob = float(torch.exp(log_probs.mean()))
        min_token_prob = float(torch.exp(log_probs.min()))

        # Clamp to [0, 1] (floating-point safety)
        geom_mean_prob = max(0.0, min(1.0, geom_mean_prob))
        min_token_prob = max(0.0, min(1.0, min_token_prob))

        c_vqa = 0.7 * geom_mean_prob + 0.3 * min_token_prob

        return self._out(
            confidence=c_vqa,
            extra={
                "geometric_mean_prob": geom_mean_prob,
                "min_token_prob": min_token_prob,
                "num_tokens": int(valid_mask.sum()),
                "temperature": self.temperature,
            },
        )

    # ==================================================================
    # 2.  Grounding Confidence
    # ==================================================================

    def grounding_confidence(
        self,
        detector_logit: Optional[float] = None,
        coord_log_probs: Optional[Union[torch.Tensor, List[float]]] = None,
    ) -> Dict:
        """
        Compute confidence for a bounding-box prediction.

        Two sub-scores are supported and combined 50 / 50:

        **Detector-based** (GroundingDINO style)::

            detector_conf = sigmoid( max_text_box_alignment_logit )

        **Autoregressive coordinate tokens**::

            coord_conf = geometric_mean( exp(log_probs_of_4_coord_tokens) )

        If only one is provided, it is used directly as the final score.

        Parameters
        ----------
        detector_logit : float, optional
            Raw (un-sigmoided) alignment logit from the detection head.
        coord_log_probs : Tensor | list[float], optional
            Log-probabilities for the four coordinate tokens
            [x_min, y_min, x_max, y_max].

        Returns
        -------
        dict with keys: confidence, tier, icon, detector_conf,
                        coord_conf, combination_mode
        """
        if detector_logit is None and coord_log_probs is None:
            raise ValueError(
                "At least one of detector_logit or coord_log_probs must be provided."
            )

        detector_conf: Optional[float] = None
        coord_conf: Optional[float] = None

        # --- detector branch ---
        if detector_logit is not None:
            detector_conf = float(torch.sigmoid(torch.tensor(detector_logit)))

        # --- autoregressive coordinate branch ---
        if coord_log_probs is not None:
            if isinstance(coord_log_probs, list):
                coord_log_probs = torch.tensor(coord_log_probs, dtype=torch.float32)
            coord_log_probs = coord_log_probs.float().detach().clamp(max=0.0)
            coord_conf = float(torch.exp(coord_log_probs.mean()))
            coord_conf = max(0.0, min(1.0, coord_conf))

        # --- combine ---
        if detector_conf is not None and coord_conf is not None:
            combined = 0.5 * detector_conf + 0.5 * coord_conf
            mode = "detector+coord"
        elif detector_conf is not None:
            combined = detector_conf
            mode = "detector_only"
        else:
            combined = coord_conf  # type: ignore[assignment]
            mode = "coord_only"

        return self._out(
            confidence=combined,
            extra={
                "detector_conf": detector_conf,
                "coord_conf": coord_conf,
                "combination_mode": mode,
            },
        )

    # ==================================================================
    # 3.  Change Detection Confidence
    # ==================================================================

    def change_confidence(
        self,
        prob_map: Union[torch.Tensor, np.ndarray],
        ambiguity_lo: float = 0.35,
        ambiguity_hi: float = 0.65,
    ) -> Dict:
        """
        Compute confidence for a binary change-detection probability map.

        Formulae::

            pixel_conf_map   = 2 * |P - 0.5|          # ∈ [0, 1]
            ambiguity_ratio  = fraction of pixels in [ambiguity_lo, ambiguity_hi]
            img_conf_ambiguity = 1 - ambiguity_ratio

            changed_pixels    = P >= 0.5
            fg_confidence     = mean(pixel_conf_map[changed_pixels])
                                (= 0.0 if no changed pixels)

            image_conf = 0.6 * img_conf_ambiguity + 0.4 * fg_confidence

        Parameters
        ----------
        prob_map : Tensor | ndarray, shape (H, W) or (1, H, W)
            Per-pixel change probabilities in [0, 1].
        ambiguity_lo : float
            Lower boundary of the ambiguous zone (default 0.35).
        ambiguity_hi : float
            Upper boundary of the ambiguous zone (default 0.65).

        Returns
        -------
        dict with keys: confidence, tier, icon, img_conf_ambiguity,
                        fg_confidence, ambiguity_ratio, changed_pixel_fraction
        """
        if isinstance(prob_map, torch.Tensor):
            p = prob_map.float().detach().cpu().numpy()
        else:
            p = np.asarray(prob_map, dtype=np.float32)

        if p.ndim == 3:
            p = p.squeeze(0)
        if p.ndim != 2:
            raise ValueError(
                f"prob_map must be 2-D (H, W) or 3-D (1, H, W), got shape {prob_map.shape}"
            )

        p = np.clip(p, 0.0, 1.0)
        n_pixels = p.size

        # Distance from decision boundary
        pixel_conf_map = 2.0 * np.abs(p - 0.5)

        # Ambiguity: fraction of uncertain pixels
        ambiguous_mask = (p >= ambiguity_lo) & (p <= ambiguity_hi)
        ambiguity_ratio = float(ambiguous_mask.sum()) / n_pixels
        img_conf_ambiguity = 1.0 - ambiguity_ratio

        # Foreground (changed) confidence
        changed_mask = p >= 0.5
        changed_fraction = float(changed_mask.sum()) / n_pixels
        if changed_mask.any():
            fg_confidence = float(pixel_conf_map[changed_mask].mean())
        else:
            fg_confidence = 0.0

        image_conf = 0.6 * img_conf_ambiguity + 0.4 * fg_confidence

        return self._out(
            confidence=image_conf,
            extra={
                "img_conf_ambiguity": float(img_conf_ambiguity),
                "fg_confidence": float(fg_confidence),
                "ambiguity_ratio": float(ambiguity_ratio),
                "changed_pixel_fraction": changed_fraction,
            },
        )

    # ==================================================================
    # 4.  Fusion Confidence (Weighted Harmonic Mean)
    # ==================================================================

    def fusion_confidence(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Combine per-task confidence scores via a **weighted harmonic mean**.

        The harmonic mean strongly penalises the lowest-scoring component,
        making it ideal for multi-task fusion where all tasks must be
        reliable for the overall answer to be trustworthy.

        Formula::

            WHM = Σ w_i / Σ (w_i / (s_i + ε))

        Parameters
        ----------
        scores : dict[str, float]
            Per-task confidence scores, e.g.
            ``{'vqa': 0.9, 'grounding': 0.7, 'change': 0.85}``.
        weights : dict[str, float], optional
            Per-task weights.  If omitted, equal weights are used.
            Keys must be a superset of ``scores`` keys.

        Returns
        -------
        dict with keys: confidence, tier, icon, component_scores,
                        component_weights, bottleneck_task, harmonic_mean
        """
        if not scores:
            raise ValueError("scores dict must not be empty.")

        tasks = list(scores.keys())

        if weights is None:
            w = {t: 1.0 for t in tasks}
        else:
            missing = set(tasks) - set(weights.keys())
            if missing:
                raise ValueError(f"weights missing for tasks: {missing}")
            w = {t: float(weights[t]) for t in tasks}

        # Normalise weights
        w_total = sum(w.values())
        if w_total <= 0:
            raise ValueError("Sum of weights must be positive.")
        w_norm = {t: w[t] / w_total for t in tasks}

        numerator = sum(w_norm[t] for t in tasks)  # = 1.0 by construction
        denominator = sum(w_norm[t] / (scores[t] + _EPS) for t in tasks)

        whm = numerator / (denominator + _EPS)
        whm = max(0.0, min(1.0, whm))

        bottleneck = min(tasks, key=lambda t: scores[t])

        return self._out(
            confidence=whm,
            extra={
                "component_scores": dict(scores),
                "component_weights": w_norm,
                "bottleneck_task": bottleneck,
                "harmonic_mean": whm,
            },
        )

    # ==================================================================
    # 5.  Convenience: compute all tasks and fuse
    # ==================================================================

    def compute_all(
        self,
        vqa_transition_scores: Optional[torch.Tensor] = None,
        detector_logit: Optional[float] = None,
        coord_log_probs: Optional[Union[torch.Tensor, List[float]]] = None,
        change_prob_map: Optional[Union[torch.Tensor, np.ndarray]] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Compute individual confidences for available tasks, then fuse them.

        Only tasks whose required inputs are provided are computed.  At
        least one task input is required.

        Returns
        -------
        dict with keys:
            - 'fusion': fusion result dict
            - 'vqa':    VQA result dict (if inputs given)
            - 'grounding': grounding result dict (if inputs given)
            - 'change': change result dict (if inputs given)
        """
        results: Dict[str, Dict] = {}

        if vqa_transition_scores is not None:
            results["vqa"] = self.vqa_confidence(vqa_transition_scores)

        if detector_logit is not None or coord_log_probs is not None:
            results["grounding"] = self.grounding_confidence(
                detector_logit=detector_logit,
                coord_log_probs=coord_log_probs,
            )

        if change_prob_map is not None:
            results["change"] = self.change_confidence(change_prob_map)

        if not results:
            raise ValueError(
                "At least one task input must be provided to compute_all()."
            )

        scores = {task: r["confidence"] for task, r in results.items()}
        results["fusion"] = self.fusion_confidence(scores, weights=task_weights)

        return results
