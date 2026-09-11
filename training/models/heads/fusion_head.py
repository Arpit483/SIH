"""
FusionHead: Optical-SAR cross-modal attention fusion head for SatQueryUnified.

Architecture
------------
  - 2 bidirectional cross-attention layers:
      SAR tokens  attend to optical tokens  (SAR  <- optical context)
      Optical tokens attend to SAR tokens   (optical <- SAR context)
  - Residual connections + LayerNorm after each attention + FFN block
  - Output projection: 1024 -> 1024  (element-wise mean of two streams)

Input
-----
  optical_features : [B, N, 1024]
  sar_features     : [B, N, 1024]

Output
------
  dict with key 'fused_features' : [B, N, 1024]  ready for VQAHead
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cross-Modal Attention Layer
# ---------------------------------------------------------------------------

class CrossModalAttentionLayer(nn.Module):
    """
    One layer of bidirectional cross-modal attention.

    For a pair of feature sequences (a=optical, b=SAR):
      a' = norm(a + CrossAttn(Q=a, KV=b))  followed by FFN + norm
      b' = norm(b + CrossAttn(Q=b, KV=a))  followed by FFN + norm

    Parameters
    ----------
    dim       : int   Token dimension (1024 for ViT-L/14).
    num_heads : int   Multi-head attention heads.
    dropout   : float Dropout probability.
    """

    def __init__(
        self,
        dim      : int = 1024,
        num_heads: int = 8,
        dropout  : float = 0.0,
    ) -> None:
        super().__init__()

        # Optical (a) attends to SAR (b)
        self.cross_attn_a = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm  = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

        # SAR (b) attends to optical (a)
        self.cross_attn_b = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_b  = nn.LayerNorm(dim)
        self.ffn_b   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_b2 = nn.LayerNorm(dim)

    def forward(
        self,
        a: torch.Tensor,   # optical  [B, N, dim]
        b: torch.Tensor,   # SAR      [B, N, dim]
    ) -> tuple:
        """
        Returns
        -------
        a_out, b_out : updated feature tensors, each [B, N, dim]
        """
        # Optical attends to SAR
        ca_a, _ = self.cross_attn_a(query=a, key=b, value=b)
        a_out   = self.norm(a + ca_a)
        a_out   = self.norm2(a_out + self.ffn(a_out))

        # SAR attends to optical
        ca_b, _ = self.cross_attn_b(query=b, key=a, value=a)
        b_out   = self.norm_b(b + ca_b)
        b_out   = self.norm_b2(b_out + self.ffn_b(b_out))

        return a_out, b_out


# ---------------------------------------------------------------------------
# FusionHead
# ---------------------------------------------------------------------------

class FusionHead(nn.Module):
    """
    Cross-modal optical-SAR fusion head.

    Input
    -----
    optical_features : torch.Tensor [B, N, 1024]
    sar_features     : torch.Tensor [B, N, 1024]

    Output
    ------
    dict:
      'fused_features' : [B, N, 1024]  ready for VQAHead

    Parameters
    ----------
    visual_dim : int   Token dimension (1024 for ViT-L/14).
    num_layers : int   Number of bidirectional cross-attention layers.
    num_heads  : int   Attention heads per layer.
    dropout    : float Dropout probability.
    """

    def __init__(
        self,
        visual_dim: int = 1024,
        num_layers: int = 2,
        num_heads : int = 8,
        dropout   : float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(
                dim      =visual_dim,
                num_heads=num_heads,
                dropout  =dropout,
            )
            for _ in range(num_layers)
        ])

        # Merge optical + SAR via element-wise mean, then project 1024 -> 1024
        self.out_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.LayerNorm(visual_dim),
        )

    def forward(
        self,
        optical_features: torch.Tensor,   # [B, N, 1024]
        sar_features    : torch.Tensor,   # [B, N, 1024]
    ) -> dict:
        """
        Returns
        -------
        dict  {'fused_features': torch.Tensor [B, N, 1024]}
        """
        a = optical_features
        b = sar_features

        for layer in self.layers:
            a, b = layer(a, b)

        fused = (a + b) * 0.5          # element-wise mean
        fused = self.out_proj(fused)   # [B, N, 1024]

        return {"fused_features": fused}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"FusionHead smoke test  |  device: {device}")
    print(f"{'='*55}\n")

    BATCH = 2
    N     = 197   # ViT-L/14: 196 patches + 1 CLS

    head = FusionHead(visual_dim=1024, num_layers=2, num_heads=8).to(device)

    optical = torch.randn(BATCH, N, 1024, device=device)
    sar     = torch.randn(BATCH, N, 1024, device=device)

    with torch.no_grad():
        out = head(optical_features=optical, sar_features=sar)

    fused = out["fused_features"]
    print(f"fused_features shape : {fused.shape}")   # [2, 197, 1024]
    assert fused.shape == (BATCH, N, 1024)
    assert not torch.isnan(fused).any()

    print("\nAll FusionHead checks passed")
    print(f"{'='*55}\n")