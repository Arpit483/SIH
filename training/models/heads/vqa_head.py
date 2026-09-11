"""
VQAHead: Visual Question Answering head for SatQueryUnified.

Architecture:
  - Projects ViT-L/14 patch features (dim=1024) to Flan-T5 encoder dim (dim=768)
    via a linear projection layer.
  - Cross-attention between projected visual tokens and T5 encoder hidden states.
  - T5 decoder generates free-form answers (open-ended) or scores candidates
    (closed-set VQA).
  - Shared between regular single-image VQA and change-VQA (which feeds
    difference features from ChangeHead).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-attention bridge: visual tokens → T5 encoder space
# ---------------------------------------------------------------------------

class VisualCrossAttention(nn.Module):
    """
    Multi-head cross-attention that lets T5 encoder hidden states
    attend to projected visual tokens.

    Args:
        embed_dim (int): T5 model dimension (768 for Flan-T5-base).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,          # [B, L_text, D]
        key_value: torch.Tensor,      # [B, L_vis,  D]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Text hidden states from T5 encoder, shape [B, L_text, D].
            key_value: Projected visual tokens, shape [B, L_vis, D].
            key_padding_mask: Bool mask for visual tokens, True = ignore.

        Returns:
            Updated query tensor, shape [B, L_text, D].
        """
        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(query + self.dropout(attn_out))


# ---------------------------------------------------------------------------
# VQAHead
# ---------------------------------------------------------------------------

class VQAHead(nn.Module):
    """
    VQA head using Flan-T5-base as the language decoder.

    Supports:
      - Open-ended generation (beam search / sampling).
      - Closed-set VQA: scores a fixed list of candidate answers by
        computing the per-token log-likelihood of each candidate and
        returning the highest-scoring one.
      - Training mode: teacher-forced with answer_input_ids, returns
        token-level cross-entropy loss.

    Args:
        vis_dim (int): Dimension of incoming visual tokens (1024 for ViT-L/14).
        t5_model_name (str): HuggingFace model identifier for the T5 backbone.
        num_xattn_layers (int): Number of cross-attention layers inserted
            between the projected visual tokens and T5 encoder output.
        dropout (float): Dropout probability.
        max_answer_length (int): Maximum tokens to generate during inference.
        freeze_t5_encoder (bool): Whether to freeze the T5 encoder weights.
        freeze_t5_decoder (bool): Whether to freeze the T5 decoder weights.
    """

    VIS_DIM: int = 1024          # ViT-L/14 token dimension
    T5_DIM: int = 768            # Flan-T5-base hidden dimension
    T5_MODEL: str = "google/flan-t5-base"

    def __init__(
        self,
        vis_dim: int = 1024,
        t5_model_name: str = "google/flan-t5-base",
        num_xattn_layers: int = 2,
        dropout: float = 0.1,
        max_answer_length: int = 64,
        freeze_t5_encoder: bool = False,
        freeze_t5_decoder: bool = False,
    ) -> None:
        super().__init__()

        self.vis_dim = vis_dim
        self.max_answer_length = max_answer_length

        # ------------------------------------------------------------------
        # Load Flan-T5-base
        # ------------------------------------------------------------------
        logger.info("Loading %s …", t5_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_dim: int = self.t5.config.d_model  # 768

        # ------------------------------------------------------------------
        # Visual projection: ViT-L dim → T5 dim
        # ------------------------------------------------------------------
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, self.t5_dim),
            nn.LayerNorm(self.t5_dim),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Cross-attention layers (visual → text)
        # ------------------------------------------------------------------
        self.xattn_layers = nn.ModuleList([
            VisualCrossAttention(
                embed_dim=self.t5_dim,
                num_heads=8,
                dropout=dropout,
            )
            for _ in range(num_xattn_layers)
        ])

        # ------------------------------------------------------------------
        # Optional freeze
        # ------------------------------------------------------------------
        if freeze_t5_encoder:
            for p in self.t5.encoder.parameters():
                p.requires_grad = False
            logger.info("T5 encoder frozen.")
        if freeze_t5_decoder:
            for p in self.t5.decoder.parameters():
                p.requires_grad = False
            logger.info("T5 decoder frozen.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_visual(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Project raw visual tokens to T5 embedding space.

        Args:
            visual_features: Shape [B, N_patches, vis_dim].

        Returns:
            Projected tokens, shape [B, N_patches, t5_dim].
        """
        return self.vis_proj(visual_features)  # [B, N, T5_DIM]

    def _encode_with_visual(
        self,
        question_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_vis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run T5 encoder on question tokens, then apply cross-attention layers
        so that text hidden states attend to visual tokens.

        Args:
            question_input_ids: [B, L_q].
            attention_mask: [B, L_q].
            projected_vis: [B, N_vis, t5_dim].

        Returns:
            Visually-augmented encoder hidden states, [B, L_q, t5_dim].
        """
        encoder_out = self.t5.encoder(
            input_ids=question_input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # [B, L_q, T5_DIM]

        for xattn in self.xattn_layers:
            encoder_out = xattn(
                query=encoder_out,
                key_value=projected_vis,
            )  # [B, L_q, T5_DIM]

        return encoder_out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        visual_features: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
        answer_input_ids: Optional[torch.Tensor] = None,
        answer_attention_mask: Optional[torch.Tensor] = None,
        candidates: Optional[List[str]] = None,
        tokenizer: Optional[T5Tokenizer] = None,
        generate_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Forward pass for VQAHead.

        Three operating modes determined by which optional args are provided:

        1. **Training** (``answer_input_ids`` supplied):
           Returns ``{"loss": scalar}``.

        2. **Open-ended inference** (no ``answer_input_ids``, no ``candidates``):
           Returns ``{"logits": [B, L_ans, vocab_size]}`` when
           ``generate_kwargs`` is None, or ``{"generated_ids": [B, L_ans]}``
           when ``generate_kwargs`` is provided.

        3. **Closed-set inference** (``candidates`` + ``tokenizer`` supplied):
           Returns ``{"answer": str, "scores": List[float]}``.

        Args:
            visual_features:         [B, N_vis, vis_dim]
            question_input_ids:      [B, L_q]
            question_attention_mask: [B, L_q] – auto-generated if None.
            answer_input_ids:        [B, L_a] – teacher-forced labels (training).
            answer_attention_mask:   [B, L_a] – mask for label padding.
            candidates:              List of candidate answer strings (closed-set).
            tokenizer:               Tokenizer for closed-set scoring.
            generate_kwargs:         Dict of kwargs forwarded to ``t5.generate``.

        Returns:
            Dict with keys depending on operating mode.
        """
        B = visual_features.size(0)

        if question_attention_mask is None:
            question_attention_mask = (question_input_ids != 0).long()

        projected_vis = self._project_visual(visual_features)  # [B, N, T5_DIM]

        encoder_hidden = self._encode_with_visual(
            question_input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            projected_vis=projected_vis,
        )  # [B, L_q, T5_DIM]

        # ---- Training mode -----------------------------------------------
        if answer_input_ids is not None:
            if answer_attention_mask is None:
                answer_attention_mask = (answer_input_ids != -100).long()

            # T5 uses -100 as ignore index for labels
            labels = answer_input_ids.clone()
            labels[labels == 0] = -100  # pad tokens ignored

            out = self.t5(
                encoder_outputs=(encoder_hidden,),
                attention_mask=question_attention_mask,
                labels=labels,
                decoder_attention_mask=answer_attention_mask,
            )
            return {"loss": out.loss, "logits": out.logits}

        # ---- Closed-set inference -----------------------------------------
        if candidates is not None:
            assert tokenizer is not None, (
                "tokenizer must be provided for closed-set VQA."
            )
            return self._closed_set_inference(
                encoder_hidden=encoder_hidden,
                encoder_attention_mask=question_attention_mask,
                candidates=candidates,
                tokenizer=tokenizer,
            )

        # ---- Open-ended generation ----------------------------------------
        if generate_kwargs is not None:
            gen_ids = self.t5.generate(
                encoder_outputs=(encoder_hidden,),
                attention_mask=question_attention_mask,
                max_new_tokens=self.max_answer_length,
                **generate_kwargs,
            )
            return {"generated_ids": gen_ids}

        # ---- Return raw logits (for custom decoding) ----------------------
        dummy_decoder_ids = torch.zeros(
            B, 1, dtype=torch.long, device=visual_features.device
        )
        out = self.t5(
            encoder_outputs=(encoder_hidden,),
            attention_mask=question_attention_mask,
            decoder_input_ids=dummy_decoder_ids,
        )
        return {"logits": out.logits}

    @torch.no_grad()
    def _closed_set_inference(
        self,
        encoder_hidden: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        candidates: List[str],
        tokenizer: T5Tokenizer,
    ) -> dict:
        """
        Score each candidate answer by its conditional log-likelihood and
        return the highest-scoring one.

        Args:
            encoder_hidden:        [B, L_q, T5_DIM]
            encoder_attention_mask:[B, L_q]
            candidates:            List[str] – answer candidates.
            tokenizer:             T5Tokenizer for encoding candidates.

        Returns:
            Dict with keys:
              - "answer"  (str): Best candidate.
              - "scores"  (List[float]): Per-candidate log-likelihoods.
        """
        B = encoder_hidden.size(0)
        assert B == 1, "Closed-set inference supports batch_size=1 only."

        device = encoder_hidden.device
        best_score = float("-inf")
        best_ans = candidates[0]
        scores: List[float] = []

        for cand in candidates:
            cand_ids = tokenizer(
                cand,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).input_ids.to(device)  # [1, L_c]

            labels = cand_ids.clone()

            out = self.t5(
                encoder_outputs=(encoder_hidden,),
                attention_mask=encoder_attention_mask,
                labels=labels,
            )
            # Negative cross-entropy → log-likelihood
            score = -out.loss.item()
            scores.append(score)
            if score > best_score:
                best_score = score
                best_ans = cand

        return {"answer": best_ans, "scores": scores}

    # ------------------------------------------------------------------
    # Convenience: generate text (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
        tokenizer: Optional[T5Tokenizer] = None,
        **generate_kwargs,
    ) -> List[str]:
        """
        Generate answer strings given visual features and question tokens.

        Args:
            visual_features:         [B, N_vis, vis_dim].
            question_input_ids:      [B, L_q].
            question_attention_mask: [B, L_q] or None.
            tokenizer:               If provided, decode output ids to strings.
            **generate_kwargs:       Forwarded to ``t5.generate``.

        Returns:
            If tokenizer is given: list of decoded answer strings.
            Otherwise: raw generated token-id tensors.
        """
        out = self.forward(
            visual_features=visual_features,
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            generate_kwargs=generate_kwargs or {"num_beams": 4},
        )
        gen_ids = out["generated_ids"]  # [B, L_out]
        if tokenizer is not None:
            return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return gen_ids


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    BATCH = 2
    N_VIS = 196 + 1   # 14×14 patches + CLS token for ViT-L/14
    L_Q = 20
    L_A = 12

    head = VQAHead(
        vis_dim=1024,
        t5_model_name="google/flan-t5-base",
        num_xattn_layers=2,
        freeze_t5_encoder=False,
    ).to(device)

    vis = torch.randn(BATCH, N_VIS, 1024, device=device)
    q_ids = torch.randint(0, 32000, (BATCH, L_Q), device=device)
    a_ids = torch.randint(0, 32000, (BATCH, L_A), device=device)

    # --- Training ---
    out_train = head(
        visual_features=vis,
        question_input_ids=q_ids,
        answer_input_ids=a_ids,
    )
    print("Training loss:", out_train["loss"].item())
    print("Logits shape: ", out_train["logits"].shape)

    # --- Open-ended generation ---
    out_gen = head(
        visual_features=vis[:1],
        question_input_ids=q_ids[:1],
        generate_kwargs={"num_beams": 2, "max_new_tokens": 10},
    )
    print("Generated ids shape:", out_gen["generated_ids"].shape)

    print("\nAll VQAHead checks passed ✓")
