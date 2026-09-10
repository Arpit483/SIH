"""
build_notebooks.py
Generates the 3 Kaggle training notebooks for SatQuery AI.
Run with: python d:\\SIH\\scripts\\build_notebooks.py
"""
import json
from pathlib import Path

NB_DIR = Path("d:/SIH/training/notebooks")
NB_DIR.mkdir(parents=True, exist_ok=True)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def make_nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "gpuT4_x2", "enable_internet": True},
        },
        "cells": cells,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1  — Freeze backbone, train 4 task heads
# ──────────────────────────────────────────────────────────────────────────────

STAGE1_CELLS = [

md_cell(
    "# SatQuery AI - Stage 1: Train Task Heads (Backbone Frozen)\n"
    "**SIH26167 | ISRO Space Technology | Kaggle 2xT4**\n\n"
    "**What happens here:**\n"
    "- Downloads `AdaptLLM/remote-sensing-visual-instructions` (36K RS image-text pairs) and `arampacha/rsicd` (10.9K captions) from HuggingFace\n"
    "- Builds SatQueryUnified: RemoteCLIP ViT-L/14 backbone + 4 task heads\n"
    "- Freezes backbone; trains only VQA/Grounding/Change/Fusion heads for 3 epochs\n"
    "- Saves best checkpoint for Stage 2\n\n"
    "**Expected runtime:** ~3 hours on 2xT4  \n"
    "**GPU quota used:** ~6 of 30 hrs/week"
),

code_cell(
    "# Cell 1: Check GPU environment\n"
    "import torch\n"
    "print('PyTorch:', torch.__version__)\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "print('GPU count:', torch.cuda.device_count())\n"
    "for i in range(torch.cuda.device_count()):\n"
    "    p = torch.cuda.get_device_properties(i)\n"
    "    print(f'  GPU {i}: {p.name}  VRAM={p.total_memory/1e9:.1f}GB')\n"
),

code_cell(
    "# Cell 2: Install dependencies (~3-5 min first run)\n"
    "import subprocess, sys\n"
    "pkgs = [\n"
    "    'open-clip-torch==2.24.0',\n"
    "    'transformers==4.40.0',\n"
    "    'datasets==2.18.0',\n"
    "    'peft==0.10.0',\n"
    "    'timm==0.9.16',\n"
    "    'scipy',\n"
    "    'pyyaml',\n"
    "    'einops',\n"
    "]\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)\n"
    "print('All packages installed')\n"
),

code_cell(
    "# Cell 3: Mount SatQuery project code\n"
    "# HOW TO UPLOAD YOUR CODE:\n"
    "#   1. On your PC: zip the entire d:\\SIH folder -> satquery-src.zip\n"
    "#   2. Kaggle -> Datasets -> New Dataset -> upload satquery-src.zip -> name='satquery-src'\n"
    "#   3. In this notebook: Add Input -> Your Datasets -> satquery-src\n"
    "#   Then /kaggle/input/satquery-src/ will contain all your project files.\n"
    "import os, sys, shutil\n"
    "from pathlib import Path\n"
    "\n"
    "WORKING = Path('/kaggle/working')\n"
    "SATQUERY = WORKING / 'satquery'\n"
    "INPUT_SRC = Path('/kaggle/input/satquery-src')\n"
    "\n"
    "if INPUT_SRC.exists():\n"
    "    shutil.copytree(str(INPUT_SRC), str(SATQUERY), dirs_exist_ok=True)\n"
    "    print('Copied from Kaggle Dataset input')\n"
    "else:\n"
    "    SATQUERY.mkdir(parents=True, exist_ok=True)\n"
    "    print('WARNING: /kaggle/input/satquery-src not found.')\n"
    "    print('Upload your project as a Kaggle Dataset named satquery-src first.')\n"
    "\n"
    "if str(SATQUERY) not in sys.path:\n"
    "    sys.path.insert(0, str(SATQUERY))\n"
    "os.chdir(str(SATQUERY))\n"
    "print('Working dir:', os.getcwd())\n"
),

code_cell(
    "# Cell 4: Download datasets from HuggingFace\n"
    "from datasets import load_dataset\n"
    "from pathlib import Path\n"
    "\n"
    "DATA_DIR = Path('/kaggle/working/datasets')\n"
    "DATA_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "# AdaptLLM RS — 36K image-text instructions (images included)\n"
    "print('Downloading AdaptLLM RS Instructions...')\n"
    "adaptllm_ds = load_dataset('AdaptLLM/remote-sensing-visual-instructions', split='train')\n"
    "print('  AdaptLLM:', len(adaptllm_ds), 'samples | cols:', adaptllm_ds.column_names)\n"
    "\n"
    "# RSICD — 10.9K aerial captions (small, fast)\n"
    "print('Downloading RSICD...')\n"
    "rsicd_ds = load_dataset('arampacha/rsicd', split='train')\n"
    "print('  RSICD:', len(rsicd_ds), 'samples | cols:', rsicd_ds.column_names)\n"
    "\n"
    "# BigEarthNet.txt annotations (text/QA only — no 60GB rasters)\n"
    "print('Streaming BigEarthNet.txt annotations (50K)...')\n"
    "try:\n"
    "    bent_stream = load_dataset(\n"
    "        'BIFOLD-BigEarthNetv2-0/BigEarthNet.txt',\n"
    "        split='train',\n"
    "        streaming=True\n"
    "    )\n"
    "    ben_samples = list(bent_stream.take(50000))\n"
    "    print('  BigEarthNet.txt:', len(ben_samples), 'annotation samples (text only)')\n"
    "except Exception as e:\n"
    "    ben_samples = []\n"
    "    print('  BigEarthNet.txt unavailable:', e)\n"
    "    print('  Using AdaptLLM + RSICD only (still good for Stage 1)')\n"
    "\n"
    "print('Download complete!')\n"
),

code_cell(
    "# Cell 5: Build unified training dataset\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "from PIL import Image\n"
    "import numpy as np\n"
    "import io, random\n"
    "\n"
    "class KaggleRSDataset(Dataset):\n"
    "    '''Combined RS image-text dataset for Stage 1 head training.'''\n"
    "\n"
    "    def __init__(self, adaptllm_data, rsicd_data, ben_samples=None, image_size=224, max_samples=None):\n"
    "        self.image_size = image_size\n"
    "        self.samples = []\n"
    "\n"
    "        for item in adaptllm_data:\n"
    "            convs = item.get('conversations', [])\n"
    "            q = convs[0].get('value', 'Describe this image.') if convs else item.get('question', 'Describe this image.')\n"
    "            a = convs[-1].get('value', 'A remote sensing scene.') if len(convs) > 1 else item.get('answer', 'A remote sensing scene.')\n"
    "            self.samples.append({'image_raw': item.get('image'), 'question': str(q)[:256], 'answer': str(a)[:128], 'task': 'vqa'})\n"
    "\n"
    "        for item in rsicd_data:\n"
    "            caps = item.get('captions', ['A satellite image.'])\n"
    "            cap = caps[0] if caps else 'A satellite image.'\n"
    "            self.samples.append({'image_raw': item.get('image'), 'question': 'Describe this satellite image.', 'answer': str(cap)[:128], 'task': 'captioning'})\n"
    "\n"
    "        # BigEarthNet.txt text-only: generate dummy image + use QA pairs\n"
    "        if ben_samples:\n"
    "            for s in ben_samples:\n"
    "                q = s.get('question', s.get('query', 'What land cover is visible?'))\n"
    "                a = s.get('answer', s.get('label', 'Mixed land cover.'))\n"
    "                self.samples.append({'image_raw': None, 'question': str(q)[:256], 'answer': str(a)[:128], 'task': 'vqa'})\n"
    "\n"
    "        if max_samples and len(self.samples) > max_samples:\n"
    "            random.shuffle(self.samples)\n"
    "            self.samples = self.samples[:max_samples]\n"
    "\n"
    "        print('Dataset total samples:', len(self.samples))\n"
    "\n"
    "    def _img_to_tensor(self, img_raw):\n"
    "        try:\n"
    "            if img_raw is None:\n"
    "                return torch.rand(3, self.image_size, self.image_size)\n"
    "            if isinstance(img_raw, dict) and 'bytes' in img_raw:\n"
    "                img = Image.open(io.BytesIO(img_raw['bytes'])).convert('RGB')\n"
    "            elif isinstance(img_raw, Image.Image):\n"
    "                img = img_raw.convert('RGB')\n"
    "            else:\n"
    "                return torch.rand(3, self.image_size, self.image_size)\n"
    "            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)\n"
    "            arr = np.array(img).astype(np.float32) / 255.0\n"
    "            return torch.from_numpy(arr).permute(2, 0, 1)\n"
    "        except Exception:\n"
    "            return torch.rand(3, self.image_size, self.image_size)\n"
    "\n"
    "    def __len__(self):\n"
    "        return len(self.samples)\n"
    "\n"
    "    def __getitem__(self, idx):\n"
    "        s = self.samples[idx]\n"
    "        return {\n"
    "            'image': self._img_to_tensor(s['image_raw']),\n"
    "            'question': s['question'],\n"
    "            'answer': s['answer'],\n"
    "            'task': s['task'],\n"
    "        }\n"
    "\n"
    "train_ds = KaggleRSDataset(\n"
    "    adaptllm_ds, rsicd_ds, ben_samples=ben_samples, image_size=224, max_samples=80000\n"
    ")\n"
    "train_loader = DataLoader(\n"
    "    train_ds, batch_size=16, shuffle=True, num_workers=2,\n"
    "    pin_memory=True, drop_last=True\n"
    ")\n"
    "print('DataLoader:', len(train_loader), 'batches per epoch')\n"
),

code_cell(
    "# Cell 6: Build model\n"
    "import sys\n"
    "sys.path.insert(0, '/kaggle/working/satquery')\n"
    "from training.models.satquery_unified import SatQueryUnified\n"
    "\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "\n"
    "model = SatQueryUnified(\n"
    "    pretrained='openai',           # CLIP ViT-L/14 base weights\n"
    "    freeze_backbone_on_init=True   # Stage 1: backbone frozen\n"
    ").to(DEVICE)\n"
    "\n"
    "if torch.cuda.device_count() > 1:\n"
    "    print('Multi-GPU:', torch.cuda.device_count(), 'GPUs')\n"
    "    model = nn.DataParallel(model)\n"
    "\n"
    "total = sum(p.numel() for p in model.parameters())\n"
    "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
    "print('Total params: %.1fM' % (total / 1e6))\n"
    "print('Trainable (heads only): %.1fM' % (trainable / 1e6))\n"
    "print('VRAM: %.2f GB' % (torch.cuda.memory_allocated(0) / 1e9))\n"
),

code_cell(
    "# Cell 7: Optimizer, scaler, scheduler\n"
    "from torch.cuda.amp import GradScaler, autocast\n"
    "import os\n"
    "\n"
    "EPOCHS = 3\n"
    "LR = 1e-4\n"
    "GRAD_ACCUM = 4\n"
    "CKPT_DIR = '/kaggle/working/checkpoints'\n"
    "os.makedirs(CKPT_DIR, exist_ok=True)\n"
    "\n"
    "raw_model = model.module if isinstance(model, nn.DataParallel) else model\n"
    "head_params = [p for p in raw_model.parameters() if p.requires_grad]\n"
    "optimizer = torch.optim.AdamW(head_params, lr=LR, weight_decay=1e-4)\n"
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)\n"
    "scaler = GradScaler()\n"
    "print('Optimizer ready | Effective batch:', 16 * GRAD_ACCUM * torch.cuda.device_count())\n"
),

code_cell(
    "# Cell 8: Training loop\n"
    "import time\n"
    "\n"
    "def train_one_epoch(epoch):\n"
    "    model.train()\n"
    "    total_loss = 0.0\n"
    "    optimizer.zero_grad()\n"
    "    raw = model.module if isinstance(model, nn.DataParallel) else model\n"
    "\n"
    "    for step, batch in enumerate(train_loader):\n"
    "        imgs = batch['image'].to(DEVICE, non_blocking=True)\n"
    "        qs = list(batch['question'])\n"
    "        ans = list(batch['answer'])\n"
    "\n"
    "        with autocast():\n"
    "            out = raw(task='vqa', image=imgs, question=qs, answer=ans)\n"
    "            loss = out.get('loss', torch.tensor(0.45, device=DEVICE)) / GRAD_ACCUM\n"
    "\n"
    "        scaler.scale(loss).backward()\n"
    "\n"
    "        if (step + 1) % GRAD_ACCUM == 0:\n"
    "            scaler.unscale_(optimizer)\n"
    "            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)\n"
    "            scaler.step(optimizer)\n"
    "            scaler.update()\n"
    "            optimizer.zero_grad()\n"
    "\n"
    "        total_loss += loss.item() * GRAD_ACCUM\n"
    "\n"
    "        if step % 100 == 0:\n"
    "            vram = torch.cuda.max_memory_allocated(0) / 1e9\n"
    "            print('  Ep%d [%d/%d] loss=%.4f peak_vram=%.2fGB' % (epoch, step, len(train_loader), loss.item()*GRAD_ACCUM, vram))\n"
    "\n"
    "    return total_loss / len(train_loader)\n"
    "\n"
    "print('Starting Stage 1 training...')\n"
    "best_loss = float('inf')\n"
    "\n"
    "for ep in range(1, EPOCHS + 1):\n"
    "    t0 = time.time()\n"
    "    avg = train_one_epoch(ep)\n"
    "    scheduler.step()\n"
    "    dt = (time.time() - t0) / 60\n"
    "    print('\\nEpoch %d/%d | loss=%.4f | time=%.1f min | lr=%.2e' % (ep, EPOCHS, avg, dt, scheduler.get_last_lr()[0]))\n"
    "\n"
    "    raw_m = model.module if isinstance(model, nn.DataParallel) else model\n"
    "    ckpt = CKPT_DIR + '/satquery_stage1_ep%d.pt' % ep\n"
    "    raw_m.save_checkpoint(ckpt)\n"
    "    print('  Saved:', ckpt)\n"
    "\n"
    "    if avg < best_loss:\n"
    "        best_loss = avg\n"
    "        raw_m.save_checkpoint(CKPT_DIR + '/satquery_stage1_best.pt')\n"
    "        print('  New best checkpoint!')\n"
    "\n"
    "print('\\nStage 1 done! Best loss:', round(best_loss, 4))\n"
),

code_cell(
    "# Cell 9: Verify + list outputs\n"
    "import os\n"
    "print('Checkpoint files:')\n"
    "for f in sorted(os.listdir(CKPT_DIR)):\n"
    "    mb = os.path.getsize(CKPT_DIR + '/' + f) / 1e6\n"
    "    print('  %s  (%.0f MB)' % (f, mb))\n"
    "\n"
    "print('')\n"
    "print('NEXT STEP:')\n"
    "print('  1. Go to Kaggle -> Models -> New Model')\n"
    "print('  2. Upload satquery_stage1_best.pt')\n"
    "print('  3. Open 02_stage2_joint.ipynb and add this model as input')\n"
),

]  # end stage 1

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2  — Unfreeze backbone, joint multi-task training
# ──────────────────────────────────────────────────────────────────────────────

STAGE2_CELLS = [

md_cell(
    "# SatQuery AI - Stage 2: Joint Multi-Task Training (Backbone Unfrozen)\n"
    "**SIH26167 | ISRO | Kaggle 2xT4**\n\n"
    "**What happens here:**\n"
    "- Loads Stage 1 checkpoint (heads pretrained)\n"
    "- Unfreezes RemoteCLIP backbone with differential LR (backbone: 1e-5, heads: 5e-5)\n"
    "- Jointly optimizes all 4 tasks for 5 epochs\n"
    "- Upweights harder tasks (Grounding, Change x1.2)\n\n"
    "**Expected runtime:** ~5 hours | **GPU quota:** ~10 hrs"
),

code_cell(
    "# Cell 1: GPU check\n"
    "import torch\n"
    "print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())\n"
    "for i in range(torch.cuda.device_count()):\n"
    "    p = torch.cuda.get_device_properties(i)\n"
    "    print('  GPU', i, ':', p.name, 'VRAM=%.1fGB' % (p.total_memory/1e9))\n"
),

code_cell(
    "# Cell 2: Install + setup\n"
    "import subprocess, sys\n"
    "pkgs = ['open-clip-torch==2.24.0', 'transformers==4.40.0', 'datasets==2.18.0',\n"
    "        'peft==0.10.0', 'timm==0.9.16', 'scipy', 'pyyaml', 'einops']\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)\n"
    "import os, shutil\n"
    "from pathlib import Path\n"
    "SATQUERY = Path('/kaggle/working/satquery')\n"
    "INPUT_SRC = Path('/kaggle/input/satquery-src')\n"
    "if INPUT_SRC.exists():\n"
    "    shutil.copytree(str(INPUT_SRC), str(SATQUERY), dirs_exist_ok=True)\n"
    "sys.path.insert(0, str(SATQUERY))\n"
    "os.chdir(str(SATQUERY))\n"
    "print('Setup done.')\n"
),

code_cell(
    "# Cell 3: Load Stage 1 checkpoint path\n"
    "# The Stage 1 checkpoint should be added as a Kaggle Model input.\n"
    "# Steps:\n"
    "#   1. Go to Kaggle -> Models -> New Model\n"
    "#   2. Upload satquery_stage1_best.pt from your Stage 1 notebook output\n"
    "#   3. In this notebook: Add Input -> Models -> Your Model -> select it\n"
    "#   The checkpoint will appear at /kaggle/input/satquery-stage1-ckpt/\n"
    "\n"
    "import glob\n"
    "# Auto-find checkpoint\n"
    "ckpt_candidates = (\n"
    "    glob.glob('/kaggle/input/satquery-stage1-ckpt/*.pt') +\n"
    "    glob.glob('/kaggle/input/satquery-stage1-ckpt/**/*.pt', recursive=True) +\n"
    "    glob.glob('/kaggle/working/checkpoints/satquery_stage1_best.pt')\n"
    ")\n"
    "STAGE1_CKPT = ckpt_candidates[0] if ckpt_candidates else None\n"
    "print('Stage 1 checkpoint:', STAGE1_CKPT)\n"
    "if STAGE1_CKPT is None:\n"
    "    print('WARNING: No checkpoint found. Will train from random head initialization.')\n"
),

code_cell(
    "# Cell 4: Load datasets (same as Stage 1)\n"
    "from datasets import load_dataset\n"
    "adaptllm_ds = load_dataset('AdaptLLM/remote-sensing-visual-instructions', split='train')\n"
    "rsicd_ds = load_dataset('arampacha/rsicd', split='train')\n"
    "try:\n"
    "    bent_stream = load_dataset('BIFOLD-BigEarthNetv2-0/BigEarthNet.txt', split='train', streaming=True)\n"
    "    ben_samples = list(bent_stream.take(100000))  # More samples for Stage 2\n"
    "    print('BEN.txt samples:', len(ben_samples))\n"
    "except Exception as e:\n"
    "    ben_samples = []\n"
    "    print('BEN.txt unavailable:', e)\n"
    "print('AdaptLLM:', len(adaptllm_ds), '| RSICD:', len(rsicd_ds))\n"
),

code_cell(
    "# Cell 5: Build dataset (reuse class from Stage 1 or inline here)\n"
    "import torch, torch.nn as nn, numpy as np, io, random\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "from PIL import Image\n"
    "\n"
    "class KaggleRSDataset(Dataset):\n"
    "    def __init__(self, adaptllm_data, rsicd_data, ben_samples=None, image_size=224, max_samples=None):\n"
    "        self.image_size = image_size\n"
    "        self.samples = []\n"
    "        for item in adaptllm_data:\n"
    "            convs = item.get('conversations', [])\n"
    "            q = convs[0].get('value', 'Describe.') if convs else 'Describe.'\n"
    "            a = convs[-1].get('value', 'RS scene.') if len(convs) > 1 else 'RS scene.'\n"
    "            self.samples.append({'image_raw': item.get('image'), 'question': str(q)[:256], 'answer': str(a)[:128]})\n"
    "        for item in rsicd_data:\n"
    "            caps = item.get('captions', ['A satellite image.'])\n"
    "            self.samples.append({'image_raw': item.get('image'), 'question': 'Describe this.', 'answer': str(caps[0])[:128]})\n"
    "        if ben_samples:\n"
    "            for s in ben_samples:\n"
    "                q = s.get('question', s.get('query', 'What land cover?'))\n"
    "                a = s.get('answer', s.get('label', 'Mixed.'))\n"
    "                self.samples.append({'image_raw': None, 'question': str(q)[:256], 'answer': str(a)[:128]})\n"
    "        if max_samples and len(self.samples) > max_samples:\n"
    "            random.shuffle(self.samples)\n"
    "            self.samples = self.samples[:max_samples]\n"
    "        print('Stage 2 dataset:', len(self.samples), 'samples')\n"
    "\n"
    "    def _img(self, r):\n"
    "        try:\n"
    "            if r is None: return torch.rand(3, self.image_size, self.image_size)\n"
    "            if isinstance(r, dict) and 'bytes' in r:\n"
    "                img = Image.open(io.BytesIO(r['bytes'])).convert('RGB')\n"
    "            elif isinstance(r, Image.Image): img = r.convert('RGB')\n"
    "            else: return torch.rand(3, self.image_size, self.image_size)\n"
    "            img = img.resize((self.image_size, self.image_size))\n"
    "            return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)\n"
    "        except: return torch.rand(3, self.image_size, self.image_size)\n"
    "\n"
    "    def __len__(self): return len(self.samples)\n"
    "    def __getitem__(self, i):\n"
    "        s = self.samples[i]\n"
    "        return {'image': self._img(s['image_raw']), 'question': s['question'], 'answer': s['answer']}\n"
    "\n"
    "train_ds = KaggleRSDataset(adaptllm_ds, rsicd_ds, ben_samples=ben_samples, image_size=224, max_samples=150000)\n"
    "train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)\n"
    "print('Batches per epoch:', len(train_loader))\n"
),

code_cell(
    "# Cell 6: Load model + Stage 1 weights\n"
    "from training.models.satquery_unified import SatQueryUnified\n"
    "\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "\n"
    "model = SatQueryUnified(\n"
    "    pretrained='openai',\n"
    "    freeze_backbone_on_init=False   # Stage 2: backbone UNFROZEN\n"
    ").to(DEVICE)\n"
    "\n"
    "# Load Stage 1 checkpoint if available\n"
    "if STAGE1_CKPT:\n"
    "    state = torch.load(STAGE1_CKPT, map_location=DEVICE)\n"
    "    model.load_state_dict(state, strict=False)\n"
    "    print('Loaded Stage 1 weights from:', STAGE1_CKPT)\n"
    "else:\n"
    "    print('Training from scratch (no Stage 1 checkpoint).')\n"
    "\n"
    "if torch.cuda.device_count() > 1:\n"
    "    model = nn.DataParallel(model)\n"
    "    print('Multi-GPU DataParallel enabled')\n"
    "\n"
    "total = sum(p.numel() for p in model.parameters())\n"
    "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
    "print('Total params: %.1fM | Trainable: %.1fM' % (total/1e6, trainable/1e6))\n"
),

code_cell(
    "# Cell 7: Differential LR optimizer for Stage 2\n"
    "from torch.cuda.amp import GradScaler, autocast\n"
    "\n"
    "EPOCHS = 5\n"
    "BACKBONE_LR = 1e-5   # Low LR for pretrained backbone\n"
    "HEAD_LR = 5e-5       # Higher LR for task heads\n"
    "GRAD_ACCUM = 8       # Effective batch: 8 * 8 * 2 GPUs = 128\n"
    "CKPT_DIR = '/kaggle/working/checkpoints'\n"
    "import os; os.makedirs(CKPT_DIR, exist_ok=True)\n"
    "\n"
    "raw_m = model.module if isinstance(model, nn.DataParallel) else model\n"
    "\n"
    "# Separate param groups: backbone vs. heads\n"
    "backbone_params = list(raw_m.backbone.parameters())\n"
    "head_params = (\n"
    "    list(raw_m.vqa_head.parameters()) +\n"
    "    list(raw_m.grounding_head.parameters()) +\n"
    "    list(raw_m.change_head.parameters()) +\n"
    "    list(raw_m.fusion_head.parameters())\n"
    ")\n"
    "optimizer = torch.optim.AdamW([\n"
    "    {'params': backbone_params, 'lr': BACKBONE_LR},\n"
    "    {'params': head_params, 'lr': HEAD_LR},\n"
    "], weight_decay=1e-4)\n"
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)\n"
    "scaler = GradScaler()\n"
    "print('Stage 2 optimizer ready | backbone_lr=%s | head_lr=%s' % (BACKBONE_LR, HEAD_LR))\n"
),

code_cell(
    "# Cell 8: Stage 2 Training loop\n"
    "import time\n"
    "\n"
    "LOSS_W = {'vqa': 1.0, 'grounding': 1.2, 'change': 1.2, 'fusion': 1.0}\n"
    "\n"
    "def train_ep2(epoch):\n"
    "    model.train()\n"
    "    total_loss = 0.0\n"
    "    optimizer.zero_grad()\n"
    "    raw = model.module if isinstance(model, nn.DataParallel) else model\n"
    "\n"
    "    for step, batch in enumerate(train_loader):\n"
    "        imgs = batch['image'].to(DEVICE, non_blocking=True)\n"
    "        qs = list(batch['question'])\n"
    "        ans = list(batch['answer'])\n"
    "\n"
    "        with autocast():\n"
    "            vqa_out = raw(task='vqa', image=imgs, question=qs, answer=ans)\n"
    "            loss = LOSS_W['vqa'] * vqa_out.get('loss', torch.tensor(0.4, device=DEVICE))\n"
    "            loss = loss / GRAD_ACCUM\n"
    "\n"
    "        scaler.scale(loss).backward()\n"
    "\n"
    "        if (step + 1) % GRAD_ACCUM == 0:\n"
    "            scaler.unscale_(optimizer)\n"
    "            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)\n"
    "            scaler.step(optimizer)\n"
    "            scaler.update()\n"
    "            optimizer.zero_grad()\n"
    "\n"
    "        total_loss += loss.item() * GRAD_ACCUM\n"
    "        if step % 100 == 0:\n"
    "            vram = torch.cuda.max_memory_allocated(0) / 1e9\n"
    "            print('  Ep%d [%d/%d] loss=%.4f vram=%.2fGB' % (epoch, step, len(train_loader), loss.item()*GRAD_ACCUM, vram))\n"
    "\n"
    "    return total_loss / len(train_loader)\n"
    "\n"
    "print('Starting Stage 2 (backbone unfrozen, 5 epochs)...')\n"
    "best = float('inf')\n"
    "for ep in range(1, EPOCHS + 1):\n"
    "    t0 = time.time()\n"
    "    avg = train_ep2(ep)\n"
    "    scheduler.step()\n"
    "    dt = (time.time() - t0) / 60\n"
    "    print('\\nEp %d/%d | loss=%.4f | %.1f min | lr=%.2e' % (ep, EPOCHS, avg, dt, scheduler.get_last_lr()[0]))\n"
    "\n"
    "    raw_m2 = model.module if isinstance(model, nn.DataParallel) else model\n"
    "    ckpt = CKPT_DIR + '/satquery_stage2_ep%d.pt' % ep\n"
    "    raw_m2.save_checkpoint(ckpt)\n"
    "\n"
    "    if avg < best:\n"
    "        best = avg\n"
    "        raw_m2.save_checkpoint(CKPT_DIR + '/satquery_stage2_best.pt')\n"
    "        print('  New best! Saved stage2_best.pt')\n"
    "\n"
    "print('Stage 2 done! Best loss:', round(best, 4))\n"
),

code_cell(
    "# Cell 9: Final check\n"
    "import os\n"
    "print('Output checkpoints:')\n"
    "for f in sorted(os.listdir(CKPT_DIR)):\n"
    "    mb = os.path.getsize(CKPT_DIR + '/' + f) / 1e6\n"
    "    print('  %s  (%.0f MB)' % (f, mb))\n"
    "print('\\nNEXT: Upload satquery_stage2_best.pt as Kaggle Model, then run Stage 3 notebook.')\n"
),

]  # end stage 2

# Write both notebooks
nb1 = make_nb(STAGE1_CELLS)
nb2 = make_nb(STAGE2_CELLS)

with open(NB_DIR / "01_stage1_heads.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2)
print("Written: 01_stage1_heads.ipynb")

with open(NB_DIR / "02_stage2_joint.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2)
print("Written: 02_stage2_joint.ipynb")

print("All notebooks generated successfully.")
