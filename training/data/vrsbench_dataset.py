"""
vrsbench_dataset.py
-------------------
VRSBench Multi-Task Remote Sensing Dataset for the SatQuery AI project
(SIH26167 - ISRO remote sensing VLM).

Tasks: VQA, Grounding (bbox), Captioning

Reference : https://github.com/lx709/VRSBench  (Li et al., 2024)
Dataset   : https://huggingface.co/datasets/lx709/VRSBench

Expected directory layout:
  root_dir/
    images/          *.jpg / *.tif
    VQA/             {train,val,test}.json
    Grounding/       {train,val,test}.json
    Captioning/      {train,val,test}.json

JSON schemas:
  VQA:         [{"image":str, "question":str, "answer":str}, ...]
  Grounding:   [{"image":str, "question":str, "answer":str,
                 "bbox":[x,y,w,h], "width":int, "height":int}, ...]
  Captioning:  [{"image":str, "caption":str}, ...]

Bounding boxes are stored as absolute pixel [x, y, w, h] and converted
to normalised [cx, cy, w, h] format (all values in [0, 1]).
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.utils.data as data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_tensor(path: str) -> torch.Tensor:
    """Load RGB image -> [3, H, W] float32 in [0, 1]."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
    except ImportError as exc:
        raise ImportError("Pillow required: pip install Pillow") from exc


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _xywh_abs_to_cxcywh_norm(
    bbox: List[float], img_w: int, img_h: int
) -> List[float]:
    """
    Convert absolute pixel [x_min, y_min, w, h] to normalised [cx, cy, w, h].
    All output values clamped to [0, 1].
    """
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return [max(0., min(1., v)) for v in (cx, cy, nw, nh)]


# ---------------------------------------------------------------------------
# Sample builders
# ---------------------------------------------------------------------------

def _build_vqa_samples(json_path: str, image_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            "task":       "vqa",
            "image_file": str(image_dir / r["image"]),
            "question":   r.get("question", ""),
            "answer":     r.get("answer",   ""),
            "caption":    None,
            "bbox":       None,
            "img_w":      None,
            "img_h":      None,
        }
        for r in _load_json(json_path)
    ]


def _build_grounding_samples(json_path: str, image_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            "task":       "grounding",
            "image_file": str(image_dir / r["image"]),
            "question":   r.get("question", ""),
            "answer":     r.get("answer",   ""),
            "caption":    None,
            "bbox":       r.get("bbox", None),
            "img_w":      r.get("width",  None),
            "img_h":      r.get("height", None),
        }
        for r in _load_json(json_path)
    ]


def _build_captioning_samples(json_path: str, image_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            "task":       "captioning",
            "image_file": str(image_dir / r["image"]),
            "question":   "Describe the image.",
            "answer":     r.get("caption", ""),
            "caption":    r.get("caption", ""),
            "bbox":       None,
            "img_w":      None,
            "img_h":      None,
        }
        for r in _load_json(json_path)
    ]


# ---------------------------------------------------------------------------
# VRSBenchDataset
# ---------------------------------------------------------------------------

class VRSBenchDataset(data.Dataset):
    """
    VRSBench multi-task remote sensing dataset.

    Combines VQA, grounding, and captioning into one unified Dataset.

    Parameters
    ----------
    root_dir    : str   Path to VRSBench root directory.
    task        : str   'vqa', 'grounding', 'captioning', or 'all'.
    split       : str   'train', 'val', or 'test'.
    transform   : callable, optional  Applied to image tensor [3,H,W].
    debug       : bool  Synthetic data, no disk reads.
    max_samples : int, optional  Truncate across all tasks.
    """

    VALID_TASKS  = {"vqa", "grounding", "captioning", "all"}
    VALID_SPLITS = {"train", "val", "test"}

    _TASK_DIRS = {
        "vqa":        "VQA",
        "grounding":  "Grounding",
        "captioning": "Captioning",
    }

    def __init__(
        self,
        root_dir: str,
        task: str = "all",
        split: str = "train",
        transform: Optional[Callable] = None,
        debug: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        task  = task.lower()
        split = split.lower()

        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}")
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}")

        self.root_dir  = root_dir
        self.task      = task
        self.split     = split
        self.transform = transform
        self.debug     = debug

        active_tasks = (
            list(self._TASK_DIRS.keys()) if task == "all" else [task]
        )

        if debug:
            self._samples = self._build_debug_samples(active_tasks, max_samples or 64)
        else:
            root      = Path(root_dir)
            image_dir = root / "images"
            self._samples: List[Dict[str, Any]] = []

            _builders = {
                "vqa":        _build_vqa_samples,
                "grounding":  _build_grounding_samples,
                "captioning": _build_captioning_samples,
            }

            for t in active_tasks:
                json_path = root / self._TASK_DIRS[t] / f"{split}.json"
                if not json_path.exists():
                    warnings.warn(
                        f"VRSBench {t} JSON not found: '{json_path}'. Skipping.",
                        RuntimeWarning, stacklevel=2,
                    )
                    continue
                try:
                    self._samples.extend(_builders[t](str(json_path), image_dir))
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load VRSBench '{t}' split '{split}': {exc}",
                        RuntimeWarning, stacklevel=2,
                    )

            if max_samples is not None:
                self._samples = self._samples[:max_samples]

        logger.info(
            "VRSBenchDataset task=%s split=%s %s: %d samples",
            task, split, "(debug)" if debug else "", len(self._samples),
        )

    @staticmethod
    def _build_debug_samples(
        active_tasks: List[str], n_total: int
    ) -> List[Dict[str, Any]]:
        n_per = max(1, n_total // len(active_tasks))
        samples: List[Dict[str, Any]] = []
        for t in active_tasks:
            for i in range(n_per):
                samples.append({
                    "task":       t,
                    "image_file": "",
                    "question":   f"[DEBUG-{t}] Question {i}?",
                    "answer":     f"[DEBUG] answer {i}",
                    "caption":    f"[DEBUG] Caption {i}." if t == "captioning" else None,
                    "bbox":       [0.4, 0.4, 0.1, 0.1]  if t == "grounding"  else None,
                    "img_w":      256,
                    "img_h":      256,
                })
        return samples[:n_total]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict with keys:
          image    : FloatTensor [3, H, W] in [0, 1]
          question : str
          answer   : str
          task     : str  ('vqa'|'grounding'|'captioning')
          bbox     : Optional[List[float]]  [cx,cy,w,h] normalised
          caption  : Optional[str]
        """
        meta = self._samples[idx]
        task = meta["task"]

        if self.debug or not meta["image_file"]:
            image = torch.rand(3, 256, 256)
        else:
            try:
                image = _load_image_tensor(meta["image_file"])
            except Exception as exc:
                warnings.warn(
                    f"VRSBench: could not load '{meta['image_file']}' (idx={idx}): {exc}. "
                    "Using zeros.",
                    RuntimeWarning, stacklevel=2,
                )
                image = torch.zeros(3, 256, 256)

        if self.transform is not None:
            image = self.transform(image)

        bbox: Optional[List[float]] = None
        if task == "grounding" and meta["bbox"] is not None:
            if self.debug:
                bbox = meta["bbox"]  # already normalised [cx,cy,w,h]
            else:
                img_w = meta["img_w"] or image.shape[-1]
                img_h = meta["img_h"] or image.shape[-2]
                try:
                    bbox = _xywh_abs_to_cxcywh_norm(
                        meta["bbox"], int(img_w), int(img_h)
                    )
                except Exception as exc:
                    warnings.warn(
                        f"BBox conversion failed idx={idx}: {exc}",
                        RuntimeWarning, stacklevel=2,
                    )

        return {
            "image":    image,
            "question": meta["question"],
            "answer":   meta["answer"],
            "task":     task,
            "bbox":     bbox,
            "caption":  meta["caption"],
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"task='{self.task}', split='{self.split}', "
            f"n={len(self)}, debug={self.debug})"
        )


# ---------------------------------------------------------------------------
# Collate / DataLoader
# ---------------------------------------------------------------------------

def vrsbench_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate list of VRSBench samples; keeps nullable bbox/caption as lists."""
    return {
        "image":    torch.stack([b["image"]    for b in batch]),
        "question": [b["question"] for b in batch],
        "answer":   [b["answer"]   for b in batch],
        "task":     [b["task"]     for b in batch],
        "bbox":     [b["bbox"]     for b in batch],
        "caption":  [b["caption"]  for b in batch],
    }


def create_vrsbench_dataloader(
    dataset: VRSBenchDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
) -> data.DataLoader:
    """Wrap VRSBenchDataset in a DataLoader with sensible defaults."""
    if shuffle is None:
        shuffle = dataset.split == "train"
    if num_workers > 0 and os.name == "nt":
        warnings.warn(
            "On Windows consider num_workers=0 to avoid multiprocessing issues.",
            RuntimeWarning, stacklevel=2,
        )
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=(dataset.split == "train"),
        collate_fn=vrsbench_collate_fn,
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for task in ("vqa", "grounding", "captioning", "all"):
        ds = VRSBenchDataset(
            root_dir="/nonexistent", task=task,
            split="train", debug=True, max_samples=12,
        )
        s = ds[0]
        print(
            f"  task={task:12s}  n={len(ds):4d}  "
            f"image={tuple(s['image'].shape)}  bbox={s['bbox']}  OK"
        )
    print("VRSBench smoke-test PASSED")
