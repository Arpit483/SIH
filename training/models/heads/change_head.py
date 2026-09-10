"""
ChangeHead: Siamese bi-temporal change detection head for SatQueryUnified.

Architecture
------------
  1. Reshape ViT patch tokens  [B, N, 1024] -> [B, 1024, 14, 14]
  2. Learned difference module : concat(f1, f2) -> [B, 256, 14, 14]
  3. U-Net decoder with 4 upsampling stages  -> change map [B, 1, 224, 224]
  4. Change features for VQA  : AdaptiveAvgPool2d(1) -> Linear -> [B, 1024]

Loss
----
  - BCE loss on change map pixel predictions
  - Optional CE loss for change-VQA answers
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net decoder block
# ---------------------------------------------------------------------------

class _UNetBlock(nn.Module):
    """Conv-BN-GELU x2 block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# ChangeHead
# ---------------------------------------------------------------------------

class ChangeHead(nn.Module):
    """
    Bi-temporal change detection head.

    Input
    -----
    f1, f2 : torch.Tensor [B, N, 1024]
        ViT patch features for each timestamp.
        N can be 196 (no CLS) or 197 (with CLS, auto-stripped).

    Output
    ------
    dict:
      'change_map'      : [B, 1, 224, 224]  pixel-level change probability [0, 1]
      'change_features' : [B, 1024]         change descriptor for VQAHead
      'change_prob'     : scalar            mean change probability

    Parameters
    ----------
    visual_dim : int   Input token dimension (1024 for ViT-L/14).
    patch_grid : int   Spatial patch grid size (14 for ViT-L/14 @ 224px).
    diff_ch    : int   Channels after the learned difference module.
    """

    PATCH_GRID: int = 14

    def __init__(
        self,
        visual_dim: int = 1024,
        patch_grid: int = 14,
        diff_ch   : int = 256,
    ) -> None:
        super().__init__()

        self.visual_dim = visual_dim
        self.patch_grid = patch_grid
        self.diff_ch    = diff_ch

        # ------------------------------------------------------------------
        # Learned difference module  (concat f1, f2 -> 2048 channels in)
        # ------------------------------------------------------------------
        self.diff_module = nn.Sequential(
            nn.Conv2d(visual_dim * 2, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, diff_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(diff_ch),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # U-Net decoder: 4 upsampling stages
        # 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        # channels: 256  -> 128   ->  64  ->    32  ->  1
        # ------------------------------------------------------------------
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _UNetBlock(diff_ch, 128),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _UNetBlock(128, 64),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _UNetBlock(64, 32),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 1),  # final 1x1 conv -> single-channel logit
        )

        # ------------------------------------------------------------------
        # Change features for downstream VQA
        # [B, 256, 14, 14] -> AdaptiveAvgPool -> [B, 256] -> Linear -> [B, 1024]
        # ------------------------------------------------------------------
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.feat_proj = nn.Linear(diff_ch, visual_dim)

    # ------------------------------------------------------------------
    # Internal: reshape patch tokens [B, N, D] -> [B, D, H, H]
    # ------------------------------------------------------------------

    def _to_feature_map(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        P = self.patch_grid

        if N == P * P + 1:   # CLS token present -> strip it
            tokens = tokens[:, 1:, :]
            N = P * P

        if N != P * P:
            raise ValueError(
                f"Expected {P*P} or {P*P+1} patch tokens, got {N}. "
                f"Adjust patch_grid (currently {P})."
            )

        return tokens.permute(0, 2, 1).reshape(B, D, P, P)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        f1: torch.Tensor,   # [B, N, 1024]
        f2: torch.Tensor,   # [B, N, 1024]
    ) -> dict:
        fm1 = self._to_feature_map(f1)              # [B, 1024, 14, 14]
        fm2 = self._to_feature_map(f2)              # [B, 1024, 14, 14]

        combined = torch.cat([fm1, fm2], dim=1)     # [B, 2048, 14, 14]
        diff     = self.diff_module(combined)        # [B,  256, 14, 14]

        x = self.up1(diff)   # [B, 128, 28, 28]
        x = self.up2(x)      # [B,  64, 56, 56]
        x = self.up3(x)      # [B,  32, 112,112]
        x = self.up4(x)      # [B,   1, 224,224]

        change_map  = torch.sigmoid(x)              # [B, 1, 224, 224]  in [0,1]
        change_prob = change_map.mean()             # scalar

        pooled          = self.pool(diff).flatten(1)   # [B, 256]
        change_features = self.feat_proj(pooled)        # [B, 1024]

        return {
            "change_map"     : change_map,
            "change_features": change_features,
            "change_prob"    : change_prob,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_map            : torch.Tensor,              # [B, 1, H, W] sigmoid
        target_mask         : torch.Tensor,              # [B, 1, H, W] binary
        change_answer_logits: Optional[torch.Tensor] = None,
        target_answers      : Optional[torch.Tensor] = None,
        bce_weight          : float = 1.0,
        ce_weight           : float = 0.5,
        pos_weight          : Optional[float] = None,
    ) -> dict:
        """
        Parameters
        ----------
        pred_map             : [B, 1, H, W]  sigmoid change probability map.
        target_mask          : [B, 1, H, W]  binary ground-truth change mask.
        change_answer_logits : [B, vocab_size] or None  for change-VQA.
        target_answers       : [B] int indices or None.
        bce_weight           : weight for pixel-level BCE loss.
        ce_weight            : weight for change-VQA CE loss.
        pos_weight           : positive-class weight for BCE (imbalance handling).

        Returns
        -------
        dict with keys 'total', 'bce', and optionally 'ce'.
        """
        if pred_map.shape[-2:] != target_mask.shape[-2:]:
            pred_map = F.interpolate(
                pred_map, size=target_mask.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        pw = None
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=pred_map.device)

        pred_logits = torch.logit(pred_map.clamp(1e-6, 1 - 1e-6))
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_mask.float(), pos_weight=pw
        )

        losses = {"bce": bce_loss}
        total  = bce_weight * bce_loss

        if change_answer_logits is not None and target_answers is not None:
            ce_loss    = F.cross_entropy(change_answer_logits, target_answers)
            losses["ce"] = ce_loss
            total = total + ce_weight * ce_loss

        losses["total"] = total
        return losses


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"ChangeHead smoke test  |  device: {device}")
    print(f"{'='*55}\n")

    BATCH = 2
    N     = 197  # ViT-L/14: 196 patches + 1 CLS

    head = ChangeHead(visual_dim=1024, patch_grid=14, diff_ch=256).to(device)

    f1 = torch.randn(BATCH, N, 1024, device=device)
    f2 = torch.randn(BATCH, N, 1024, device=device)

    with torch.no_grad():
        out = head(f1=f1, f2=f2)

    print(f"change_map      shape : {out['change_map'].shape}")
    print(f"change_features shape : {out['change_features'].shape}")
    print(f"change_prob           : {out['change_prob'].item():.4f}")

    assert out["change_map"].shape      == (BATCH, 1, 224, 224)
    assert out["change_features"].shape == (BATCH, 1024)
    assert 0.0 <= out["change_prob"].item() <= 1.0

    target_mask = (torch.rand(BATCH, 1, 224, 224, device=device) > 0.7).float()
    losses = head.compute_loss(pred_map=out["change_map"], target_mask=target_mask)
    print(f"\nLosses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")
    assert "total" in losses and "bce" in losses

    print("\nAll ChangeHead checks passed")
    print(f"{'='*55}\n")