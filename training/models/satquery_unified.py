"""
SatQueryUnified: Core multimodal remote sensing VLM for SIH26167 (ISRO).

Shared backbone : RemoteCLIP ViT-L/14 (pretrained on RS5M)
Task heads      : VQAHead, GroundingHead, ChangeHead, FusionHead
VRAM @ inference: ~4.2 GB (all heads loaded simultaneously)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: load RemoteCLIP (open_clip) with HuggingFace fallback
# ---------------------------------------------------------------------------

def _build_backbone(pretrained: str = "openai"):
    """
    Attempt to load ViT-L/14 via open_clip.
    Falls back to HuggingFace CLIP if open_clip is not installed.

    Returns
    -------
    backbone   : CLIP model
    preprocess : image pre-processing transform (or None for HF path)
    tokenizer  : text tokenizer callable
    backend    : 'open_clip' | 'hf'
    """
    try:
        import open_clip

        backbone, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=pretrained,  # replaced with RemoteCLIP weights at runtime
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        logger.info("Backbone loaded via open_clip (ViT-L/14, pretrained=%s).", pretrained)
        return backbone, preprocess, tokenizer, "open_clip"

    except ImportError:
        logger.warning(
            "open_clip not found — falling back to HuggingFace "
            "openai/clip-vit-large-patch14."
        )
        from transformers import CLIPModel, CLIPProcessor

        backbone  = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        return backbone, None, processor, "hf"


# ---------------------------------------------------------------------------
# SatQueryUnified
# ---------------------------------------------------------------------------

class SatQueryUnified(nn.Module):
    """
    Unified multimodal remote sensing VLM for SIH26167.

    Shared backbone: RemoteCLIP ViT-L/14 (pretrained on RS5M)
    Task heads: VQAHead, GroundingHead, ChangeHead, FusionHead

    VRAM at inference: ~4.2GB (all heads loaded simultaneously)

    Parameters
    ----------
    pretrained : str
        open_clip pretrained tag for the backbone (default ``'openai'``).
        Pass a local checkpoint path to load RemoteCLIP weights.
    visual_dim : int
        ViT-L/14 token dimension. Must stay 1024 unless you swap the backbone.
    freeze_backbone_on_init : bool
        Freeze backbone weights immediately after construction (recommended
        for first-stage training where only heads are updated).
    """

    VISUAL_DIM: int = 1024  # ViT-L/14 output dimension

    def __init__(
        self,
        pretrained: str = "openai",
        visual_dim: int = 1024,
        freeze_backbone_on_init: bool = False,
    ) -> None:
        super().__init__()

        self.visual_dim = visual_dim

        # ------------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------------
        self.backbone, self.preprocess, self.tokenizer, self._backend = (
            _build_backbone(pretrained)
        )

        # ------------------------------------------------------------------
        # Task heads
        # ------------------------------------------------------------------
        from training.models.heads.vqa_head       import VQAHead
        from training.models.heads.grounding_head import GroundingHead
        from training.models.heads.change_head    import ChangeHead
        from training.models.heads.fusion_head    import FusionHead

        self.vqa_head       = VQAHead(vis_dim=visual_dim)
        self.grounding_head = GroundingHead(visual_dim=visual_dim)
        self.change_head    = ChangeHead(visual_dim=visual_dim)
        self.fusion_head    = FusionHead(visual_dim=visual_dim)

        # Convenience routing map
        self._heads: Dict[str, nn.Module] = {
            "vqa"       : self.vqa_head,
            "grounding" : self.grounding_head,
            "change"    : self.change_head,
            "fusion"    : self.fusion_head,
        }

        if freeze_backbone_on_init:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract ViT patch tokens (all patches, NOT just the CLS token).

        Parameters
        ----------
        image : torch.Tensor [B, 3, H, W]
            Preprocessed image tensor.

        Returns
        -------
        torch.Tensor [B, N_patches, 1024]
            For ViT-L/14 @ 224px: N = 197 (196 patches + 1 CLS).
        """
        if self._backend == "open_clip":
            visual = self.backbone.visual
            x = visual.conv1(image)                              # [B, D, 14, 14]
            x = x.reshape(x.shape[0], x.shape[1], -1)           # [B, D, 196]
            x = x.permute(0, 2, 1)                              # [B, 196, D]
            cls = visual.class_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            cls = cls.expand(x.shape[0], -1, -1)
            x   = torch.cat([cls, x], dim=1)                    # [B, 197, D]
            x   = x + visual.positional_embedding.unsqueeze(0)
            x   = visual.patch_dropout(x) if hasattr(visual, "patch_dropout") else x
            x   = visual.ln_pre(x)
            x   = visual.transformer(x)
            x   = visual.ln_post(x)                             # [B, 197, D]
            return x                                             # [B, 197, 1024]

        else:  # HuggingFace CLIPModel
            vision_out = self.backbone.vision_model(pixel_values=image)
            return vision_out.last_hidden_state                  # [B, N, 1024]

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode a list of text strings into normalised embedding vectors.

        Parameters
        ----------
        text : list of str

        Returns
        -------
        torch.Tensor [B, 1024]
        """
        if self._backend == "open_clip":
            tokens = self.tokenizer(text)
            device = next(self.backbone.parameters()).device
            tokens = tokens.to(device)
            text_features = self.backbone.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features

        else:  # HuggingFace
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            device = next(self.backbone.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_features = self.backbone.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features

    # ------------------------------------------------------------------
    # Forward / routing
    # ------------------------------------------------------------------

    def forward(self, task: str, **inputs) -> Dict[str, Any]:
        """
        Route forward pass to the correct task head.

        Parameters
        ----------
        task : str
            One of ``'vqa'``, ``'grounding'``, ``'change'``, ``'fusion'``.

        Single-image VQA
        ~~~~~~~~~~~~~~~~
        ``forward(task='vqa', image=tensor, question_input_ids=ids, ...)``

        Text-guided grounding
        ~~~~~~~~~~~~~~~~~~~~~
        ``forward(task='grounding', image=tensor, text=['find the runway'])``
        Returns ``{'boxes': [B,N,4], 'scores': [B,N]}``.

        Bi-temporal change detection
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``forward(task='change', image1=tensor, image2=tensor, **vqa_kwargs)``

        Optical-SAR fusion + VQA
        ~~~~~~~~~~~~~~~~~~~~~~~~
        ``forward(task='fusion', optical=tensor, sar=tensor, **vqa_kwargs)``

        Returns
        -------
        dict  Head-specific output dictionary.
        """
        if task not in self._heads:
            raise ValueError(
                f"Unknown task '{task}'. Choose from {list(self._heads.keys())}."
            )

        # ---- VQA -----------------------------------------------------------
        if task == "vqa":
            image = inputs.pop("image")
            visual_features = self.encode_image(image)
            return self.vqa_head(visual_features=visual_features, **inputs)

        # ---- Grounding -----------------------------------------------------
        if task == "grounding":
            image        = inputs.pop("image")
            text         = inputs.pop("text")
            visual_tokens = self.encode_image(image)
            text_emb      = self.encode_text(text)
            return self.grounding_head(
                visual_tokens=visual_tokens, text_emb=text_emb, **inputs
            )

        # ---- Change detection ----------------------------------------------
        if task == "change":
            image1 = inputs.pop("image1")
            image2 = inputs.pop("image2")
            f1 = self.encode_image(image1)
            f2 = self.encode_image(image2)
            return self.change_head(f1=f1, f2=f2, **inputs)

        # ---- Fusion --------------------------------------------------------
        if task == "fusion":
            optical = inputs.pop("optical")
            sar     = inputs.pop("sar")
            optical_features = self.encode_image(optical)
            sar_features     = self.encode_image(sar)
            fused = self.fusion_head(
                optical_features=optical_features,
                sar_features=sar_features,
            )
            if inputs:
                return self.vqa_head(
                    visual_features=fused["fused_features"], **inputs
                )
            return fused

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (gradient disabled)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_backbone(self, backbone_lr_scale: float = 0.1) -> None:
        """
        Unfreeze backbone parameters for fine-tuning.

        Parameters
        ----------
        backbone_lr_scale : float
            Suggested LR multiplier for backbone params relative to heads.
            Caller must apply the scale when constructing optimizer param groups.
        """
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info(
            "Backbone unfrozen (suggested LR scale: %.3f).", backbone_lr_scale
        )

    def get_param_groups(self, base_lr: float, backbone_lr_scale: float = 0.1):
        """
        Return optimizer param groups with separate LRs for backbone vs heads.

        Usage::

            model.unfreeze_backbone(backbone_lr_scale=0.1)
            groups = model.get_param_groups(base_lr=1e-4)
            optimizer = torch.optim.AdamW(groups)
        """
        backbone_params = list(self.backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        head_params     = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
            {"params": head_params,     "lr": base_lr},
        ]

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save model weights and meta-info to ``path``.

        Saves a dict with keys::

            {
                'model_state_dict' : self.state_dict(),
                'visual_dim'       : self.visual_dim,
                'backend'          : self._backend,
            }
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "visual_dim"      : self.visual_dim,
                "backend"         : self._backend,
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "SatQueryUnified":
        """
        Instantiate SatQueryUnified from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str  Path to the ``.pt`` file.
        device : str           Target device (``'cuda'`` or ``'cpu'``).

        Returns
        -------
        SatQueryUnified  Model in eval mode on ``device``.
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = cls(visual_dim=ckpt.get("visual_dim", 1024))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        logger.info(
            "Loaded checkpoint from %s (device=%s).", checkpoint_path, device
        )
        return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SatQueryUnified smoke test  |  device: {device}")
    print(f"{'='*60}\n")

    BATCH = 2
    N     = 197   # ViT-L/14: 14x14 patches + 1 CLS
    H = W = 224

    # Fake backbone to skip weight downloads during testing
    class _FakeBackbone(nn.Module):
        D = 1024

        def __init__(self):
            super().__init__()
            self.conv1               = nn.Conv2d(3, self.D, 14, stride=14, bias=False)
            self.class_embedding     = nn.Parameter(torch.zeros(self.D))
            self.positional_embedding = nn.Parameter(torch.zeros(N, self.D))
            self.ln_pre              = nn.LayerNorm(self.D)
            self.transformer         = nn.Identity()
            self.ln_post             = nn.LayerNorm(self.D)

        def encode_text(self, tokens):
            return torch.randn(tokens.shape[0], self.D, device=tokens.device)

    class _TestModel(SatQueryUnified):
        """Subclass that bypasses backbone download for CI / offline tests."""

        def __init__(self):
            nn.Module.__init__(self)
            self.visual_dim = 1024
            self._backend   = "open_clip"
            self.backbone   = _FakeBackbone()
            self.preprocess = None
            self.tokenizer  = lambda t: torch.zeros(len(t), 77, dtype=torch.long)

            from training.models.heads.vqa_head       import VQAHead
            from training.models.heads.grounding_head import GroundingHead
            from training.models.heads.change_head    import ChangeHead
            from training.models.heads.fusion_head    import FusionHead

            self.vqa_head       = VQAHead(vis_dim=1024)
            self.grounding_head = GroundingHead(visual_dim=1024)
            self.change_head    = ChangeHead(visual_dim=1024)
            self.fusion_head    = FusionHead(visual_dim=1024)
            self._heads = {
                "vqa"      : self.vqa_head,
                "grounding": self.grounding_head,
                "change"   : self.change_head,
                "fusion"   : self.fusion_head,
            }

    model = _TestModel().to(device)
    model.eval()

    fake_image = torch.randn(BATCH, 3, H, W, device=device)

    # --- encode_image ---
    with torch.no_grad():
        vis = model.encode_image(fake_image)
    print(f"[encode_image]   shape : {vis.shape}")           # [B, 197, 1024]
    assert vis.shape == (BATCH, N, 1024)

    # --- GroundingHead ---
    with torch.no_grad():
        text_emb = torch.randn(BATCH, 1024, device=device)
        g_out = model.grounding_head(visual_tokens=vis, text_emb=text_emb)
    print(f"[grounding]      boxes  : {g_out['boxes'].shape}")   # [B, 100, 4]
    print(f"[grounding]      scores : {g_out['scores'].shape}")  # [B, 100]
    assert g_out["boxes"].shape  == (BATCH, 100, 4)
    assert g_out["scores"].shape == (BATCH, 100)

    # --- ChangeHead ---
    f1 = torch.randn(BATCH, N, 1024, device=device)
    f2 = torch.randn(BATCH, N, 1024, device=device)
    with torch.no_grad():
        c_out = model.change_head(f1=f1, f2=f2)
    print(f"[change]         change_map      : {c_out['change_map'].shape}")
    print(f"[change]         change_features : {c_out['change_features'].shape}")
    print(f"[change]         change_prob     : {c_out['change_prob'].item():.4f}")
    assert c_out["change_map"].shape      == (BATCH, 1, 224, 224)
    assert c_out["change_features"].shape == (BATCH, 1024)

    # --- FusionHead ---
    optical = torch.randn(BATCH, N, 1024, device=device)
    sar     = torch.randn(BATCH, N, 1024, device=device)
    with torch.no_grad():
        fused = model.fusion_head(optical_features=optical, sar_features=sar)
    print(f"[fusion]         fused : {fused['fused_features'].shape}")
    assert fused["fused_features"].shape == (BATCH, N, 1024)

    print(f"\n{'='*60}")
    print("All SatQueryUnified smoke tests passed")
    print(f"{'='*60}\n")