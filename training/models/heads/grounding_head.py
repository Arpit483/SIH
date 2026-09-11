"""
GroundingHead: DETR-style text-guided region grounding head for SatQueryUnified.

Architecture
------------
  - Linear projection: 1024 -> 256 (visual tokens)
  - Linear projection: 1024 -> 256 (text embedding -> query initialisation)
  - 100 learned object queries
  - 3-layer transformer decoder (cross-attention with visual tokens)
  - MLP bbox head: 256 -> 256 -> 4  (cx, cy, w, h) in [0, 1] via sigmoid
  - Binary classification head: 256 -> 1  (object / no-object)

Loss
----
  Hungarian matching -> GIoU + L1 (boxes) + BCE (classification)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(x) if i < len(self.layers) - 1 else x
            x = layer(x)
        return x


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) -> (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalised IoU between two sets of boxes [N,4] and [M,4] (xyxy format).
    Returns [N, M] matrix.
    """
    b1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    b2 = boxes2.unsqueeze(0)  # [1, M, 4]

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter + 1e-6
    iou   = inter / union

    enc_x1 = torch.min(b1[..., 0], b2[..., 0])
    enc_y1 = torch.min(b1[..., 1], b2[..., 1])
    enc_x2 = torch.max(b1[..., 2], b2[..., 2])
    enc_y2 = torch.max(b1[..., 3], b2[..., 3])
    enc_area = ((enc_x2 - enc_x1).clamp(min=0) *
                (enc_y2 - enc_y1).clamp(min=0) + 1e-6)

    giou = iou - (enc_area - union) / enc_area
    return giou  # [N, M]


# ---------------------------------------------------------------------------
# Transformer decoder layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Single transformer decoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm1      = nn.LayerNorm(d_model)
        self.drop1      = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)
        self.drop2      = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        sa, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.drop1(sa))
        ca, _ = self.cross_attn(queries, memory, memory)
        queries = self.norm2(queries + self.drop2(ca))
        queries = self.norm3(queries + self.drop3(self.ffn(queries)))
        return queries


# ---------------------------------------------------------------------------
# GroundingHead
# ---------------------------------------------------------------------------

class GroundingHead(nn.Module):
    """
    Text-guided region grounding head.

    Input
    -----
    visual_tokens : [B, N, 1024]  ViT patch tokens from the shared backbone.
    text_emb      : [B, 1024]     Text embedding from encode_text.

    Output
    ------
    dict:
      'boxes'  : [B, N_queries, 4]  (cx, cy, w, h) normalised to [0, 1]
      'scores' : [B, N_queries]     confidence in [0, 1]
      'logits' : [B, N_queries, 1]  raw classification logits

    Parameters
    ----------
    visual_dim  : int   Input token dimension (1024 for ViT-L/14).
    d_model     : int   Internal transformer dimension.
    num_queries : int   Number of learnable object queries.
    num_layers  : int   Number of transformer decoder layers.
    num_heads   : int   Attention heads.
    dropout     : float Dropout probability.
    """

    def __init__(
        self,
        visual_dim : int = 1024,
        d_model    : int = 256,
        num_queries: int = 100,
        num_layers : int = 3,
        num_heads  : int = 8,
        dropout    : float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model     = d_model
        self.num_queries = num_queries

        # Project visual tokens: 1024 -> 256
        self.vis_proj = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Project text embedding -> query initialisation: 1024 -> 256
        self.text_to_query_init = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learned object queries [N_queries, d_model]
        self.query_embed = nn.Parameter(torch.zeros(num_queries, d_model))
        nn.init.normal_(self.query_embed)

        # Transformer decoder
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Bounding box MLP: 256 -> 256 -> 4
        self.bbox_mlp = MLP(d_model, d_model, 4, num_layers=2)

        # Classification head: 256 -> 1
        self.cls_head = nn.Linear(d_model, 1)

    def forward(
        self,
        visual_tokens: torch.Tensor,   # [B, N, 1024]
        text_emb     : torch.Tensor,   # [B, 1024]
    ) -> dict:
        B = visual_tokens.size(0)

        memory    = self.vis_proj(visual_tokens)             # [B, N, 256]
        text_bias = self.text_to_query_init(text_emb)        # [B, 256]
        text_bias = text_bias.unsqueeze(1)                   # [B, 1, 256]
        queries   = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N_q, 256]
        queries   = queries + text_bias                      # [B, N_q, 256]

        for layer in self.decoder:
            queries = layer(queries, memory)

        raw_boxes = self.bbox_mlp(queries)                   # [B, N_q, 4]
        boxes     = torch.sigmoid(raw_boxes)                 # (cx,cy,w,h) in [0,1]
        logits    = self.cls_head(queries)                   # [B, N_q, 1]
        scores    = torch.sigmoid(logits.squeeze(-1))        # [B, N_q]

        return {"boxes": boxes, "scores": scores, "logits": logits}

    def compute_loss(
        self,
        pred_boxes  : torch.Tensor,   # [B, N_q, 4]  cx,cy,w,h
        pred_logits : torch.Tensor,   # [B, N_q, 1]
        target_boxes: torch.Tensor,   # [B, N_gt, 4] cx,cy,w,h
        cost_bbox   : float = 5.0,
        cost_giou   : float = 2.0,
        cost_cls    : float = 1.0,
    ) -> dict:
        """
        DETR-style set-prediction loss with Hungarian matching.

        Returns dict with keys 'total', 'giou', 'l1', 'cls'.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError(
                "scipy is required for Hungarian matching: pip install scipy"
            )

        B, N_q, _ = pred_boxes.shape
        N_gt      = target_boxes.shape[1]
        device    = pred_boxes.device

        total_giou = torch.tensor(0.0, device=device)
        total_l1   = torch.tensor(0.0, device=device)
        total_cls  = torch.tensor(0.0, device=device)
        n_matched  = 0

        for b in range(B):
            pb = pred_boxes[b]    # [N_q, 4]
            tb = target_boxes[b]  # [N_gt, 4]
            pl = pred_logits[b]   # [N_q, 1]

            pb_xyxy  = box_cxcywh_to_xyxy(pb)
            tb_xyxy  = box_cxcywh_to_xyxy(tb)
            giou_mat = generalized_box_iou(pb_xyxy, tb_xyxy)   # [N_q, N_gt]
            l1_mat   = torch.cdist(pb, tb, p=1)                # [N_q, N_gt]
            cls_cost = torch.sigmoid(pl).expand(-1, N_gt)      # [N_q, N_gt]

            cost = (
                cost_bbox * l1_mat
                - cost_giou * giou_mat
                + cost_cls  * cls_cost
            ).detach().cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(cost)
            row_ind = torch.as_tensor(row_ind, device=device, dtype=torch.long)
            col_ind = torch.as_tensor(col_ind, device=device, dtype=torch.long)

            matched_pred = pb[row_ind]
            matched_tgt  = tb[col_ind]
            M = len(row_ind)
            n_matched += M

            mp_xyxy = box_cxcywh_to_xyxy(matched_pred)
            mt_xyxy = box_cxcywh_to_xyxy(matched_tgt)
            g = generalized_box_iou(mp_xyxy, mt_xyxy)           # [M, M]
            total_giou = total_giou + (1 - g.diagonal()).sum()
            total_l1   = total_l1   + F.l1_loss(matched_pred, matched_tgt,
                                                 reduction="sum")

            cls_labels = torch.zeros(N_q, 1, device=device)
            cls_labels[row_ind] = 1.0
            total_cls = total_cls + F.binary_cross_entropy_with_logits(
                pl, cls_labels, reduction="sum"
            )

        denom      = max(n_matched, 1)
        giou_loss  = total_giou / denom
        l1_loss    = total_l1   / denom
        cls_loss   = total_cls  / (B * N_q)
        total_loss = cost_giou * giou_loss + cost_bbox * l1_loss + cost_cls * cls_loss

        return {"total": total_loss, "giou": giou_loss, "l1": l1_loss, "cls": cls_loss}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"GroundingHead smoke test  |  device: {device}")
    print(f"{'='*55}\n")

    BATCH     = 2
    N_PATCHES = 197
    N_GT      = 3

    head = GroundingHead(
        visual_dim=1024, d_model=256, num_queries=100, num_layers=3
    ).to(device)

    vis_tokens = torch.randn(BATCH, N_PATCHES, 1024, device=device)
    text_emb   = torch.randn(BATCH, 1024, device=device)

    with torch.no_grad():
        out = head(visual_tokens=vis_tokens, text_emb=text_emb)

    print(f"boxes  shape : {out['boxes'].shape}")   # [2, 100, 4]
    print(f"scores shape : {out['scores'].shape}")  # [2, 100]
    print(f"logits shape : {out['logits'].shape}")  # [2, 100, 1]
    assert out["boxes"].shape  == (BATCH, 100, 4)
    assert out["scores"].shape == (BATCH, 100)
    assert 0.0 <= out["boxes"].min().item() and out["boxes"].max().item() <= 1.0

    target_boxes = torch.rand(BATCH, N_GT, 4, device=device)
    losses = head.compute_loss(
        pred_boxes=out["boxes"], pred_logits=out["logits"],
        target_boxes=target_boxes,
    )
    print(f"\nLosses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")
    assert all(k in losses for k in ("total", "giou", "l1", "cls"))

    print("\nAll GroundingHead checks passed")
    print(f"{'='*55}\n")