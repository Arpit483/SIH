import json
import os
from pathlib import Path

notebooks_dir = Path("d:/SIH/training/notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

def create_nb(cells, filename):
    nb = {
        "cells": [
            {
                "cell_type": "code" if c["type"] == "code" else "markdown",
                "metadata": {},
                "execution_count": None,
                "outputs": [] if c["type"] == "code" else None,
                "source": [line + "\n" for line in c["source"].split("\n")]
            }
            for c in cells
        ],
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"accelerator": "gpuT4_x2"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    target_path = notebooks_dir / filename
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {target_path}")

# 01_stage1_heads.ipynb
cells1 = [
    {"type": "markdown", "source": "# SatQuery AI - Stage 1: Freeze Backbone & Train Task Heads\n**SIH26167 (ISRO Space Technology)**\n- Accelerators: 2x NVIDIA T4 (Kaggle Free Tier)\n- Model: SatQueryUnified (RemoteCLIP ViT-L/14 Shared Backbone)\n- Goal: Train VQA, Grounding, Change, and Fusion heads while keeping backbone frozen."},
    {"type": "code", "source": "!pip install -q open-clip-torch transformers datasets peft rasterio torchgeo timm"},
    {"type": "code", "source": "import torch\nprint(f'CUDA Available: {torch.cuda.is_available()}')\nprint(f'GPU Count: {torch.cuda.device_count()}')\nfor i in range(torch.cuda.device_count()):\n    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')"},
    {"type": "code", "source": "# Clone or copy SatQuery repository into Kaggle working environment\n!git clone https://github.com/your-repo/satquery-ai.git /kaggle/working/satquery || cp -r /kaggle/input/satquery-src /kaggle/working/satquery\n%cd /kaggle/working/satquery"},
    {"type": "code", "source": "# Run Stage 1 Training (~3 hours on 2xT4)\n!python training/train.py --stage 1 --config training/configs/unified_config.yaml"},
    {"type": "code", "source": "# Verify checkpoint saved\n!ls -lh /kaggle/working/satquery/models/"}
]
create_nb(cells1, "01_stage1_heads.ipynb")

# 02_stage2_joint.ipynb
cells2 = [
    {"type": "markdown", "source": "# SatQuery AI - Stage 2: Full Joint Multi-Task Training\n**SIH26167 (ISRO Space Technology)**\n- Unfreeze RemoteCLIP backbone with differential learning rates (1e-5 backbone, 5e-5 heads)\n- Joint multi-task optimization across BigEarthNet.txt, RSVQA, VRSBench, and CDVQA."},
    {"type": "code", "source": "!pip install -q open-clip-torch transformers datasets peft rasterio torchgeo timm"},
    {"type": "code", "source": "%cd /kaggle/working/satquery\n# Run Stage 2 Training (~5 hours on 2xT4)\n!python training/train.py --stage 2 --config training/configs/unified_config.yaml"},
    {"type": "code", "source": "# Check checkpoint\n!ls -lh /kaggle/working/satquery/models/"}
]
create_nb(cells2, "02_stage2_joint.ipynb")

# 03_stage3_finetune.ipynb
cells3 = [
    {"type": "markdown", "source": "# SatQuery AI - Stage 3: Benchmark Polish & Final Evaluation\n**SIH26167 (ISRO Space Technology)**\n- Polish individual task heads for public benchmark metrics (RSVQA, VRSBench, CDVQA)\n- Test Cartosat-2S & RISAT domain adaptation\n- Run evaluation harness and export model artifact."},
    {"type": "code", "source": "!pip install -q open-clip-torch transformers datasets peft rasterio torchgeo timm"},
    {"type": "code", "source": "%cd /kaggle/working/satquery\n# Stage 3 Fine-tuning (~3 hours on 2xT4)\n!python training/train.py --stage 3 --config training/configs/unified_config.yaml"},
    {"type": "code", "source": "# Run Complete Benchmark Evaluation\n!python training/evaluate.py --checkpoint /kaggle/working/satquery/models/satquery_stage3_best.pt --output /kaggle/working/satquery/eval_report.json"},
    {"type": "code", "source": "# Display evaluation results\nimport json\nwith open('/kaggle/working/satquery/eval_report.json') as f:\n    report = json.load(f)\nprint(json.dumps(report, indent=2))"}
]
create_nb(cells3, "03_stage3_finetune.ipynb")
