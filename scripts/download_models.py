"""
download_models.py: Download fine-tuned model checkpoints from Kaggle Models or Hugging Face.
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_via_kaggle(model_slug: str):
    print(f"Downloading {model_slug} using Kaggle API...")
    try:
        import kaggle
        kaggle.api.model_instance_version_download(model_slug, path=str(MODELS_DIR), untar=True)
        print(f"Successfully downloaded to {MODELS_DIR}")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Please ensure ~/.kaggle/kaggle.json exists with valid credentials.")

def download_fallback_weights():
    print("Pre-downloading base RemoteCLIP weights for offline demo...")
    try:
        import open_clip
        _, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        print("RemoteCLIP / CLIP base weights cached successfully!")
    except Exception as e:
        print(f"Pre-download notice: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-model", type=str, default=None, help="Kaggle model handle, e.g. username/satquery-unified/pyTorch/v1")
    args = parser.parse_args()

    if args.kaggle_model:
        download_via_kaggle(args.kaggle_model)
    else:
        download_fallback_weights()
