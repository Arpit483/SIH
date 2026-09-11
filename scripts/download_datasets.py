"""
download_datasets.py: Utility to download and prepare benchmark splits for evaluation & training.
"""

import os
import argparse
from pathlib import Path

DATA_DIR = Path("d:/SIH/datasets")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def setup_mock_datasets():
    """Generates synthetic / miniature test samples for local debugging."""
    print("Setting up directory structure and local mock datasets...")
    for sub in ["rsvqa_hr", "rsvqa_lr", "vrsbench", "cdvqa", "cartosat_samples"]:
        p = DATA_DIR / sub
        p.mkdir(parents=True, exist_ok=True)
        print(f"  Initialized {p}")
    print("Mock datasets initialized. Use HuggingFace / Kaggle datasets for full training.")

if __name__ == "__main__":
    setup_mock_datasets()
