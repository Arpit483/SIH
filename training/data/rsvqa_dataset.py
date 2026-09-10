"""
rsvqa_dataset.py
----------------
RSVQA (Remote Sensing Visual Question Answering) PyTorch Dataset
for the SatQuery AI project (SIH26167 - ISRO remote sensing VLM).

Supported variants:
  RSVQA-LR  - Low Resolution  (Sentinel-2, ~772 images)
  RSVQA-HR  - High Resolution (USGS NAIP, ~10,659 images)

Question types: presence, comparison, count, area

Expected directory layouts
--------------------------
RSVQA-LR root/
  Images/  *.tif
  LR_split_{train,val,test}_{questions,answers,images}.json

RSVQA-HR root/
  Data/  *.tif
  USGS_split_{train,val}_*{questions,answers,images}.json
  USGS_split_test_phili_{questions,answers,images}.json

JSON schemas:
  Questions: {"questions": [{"id":int,"question":str,"type":str,"img_id":int},...]}
  Answers:   {"answers":   [{"id":int,"answer":str,"active":bool,"question_id":int},...]}
  Images:    {"images":    [{"id":int,"filename":str,"active":bool},...]}
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
# Image loading
# ---------------------------------------------------------------------------

def _load_pil_image(path: str) -> torch.Tensor:
    """Load image -> [3, H, W] float32 in [0, 1]."""
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


# ---------------------------------------------------------------------------
# Internal split helper
# ---------------------------------------------------------------------------

class _RSVQASplit:
    """Resolves file paths and builds an index for one variant/split."""

    _TEMPLATES: Dict[str, Dict[str, Dict[str, str]]] = {
        "lr": {
            "train": {
                "questions": "LR_split_train_questions.json",
                "answers":   "LR_split_train_answers.json",
                "images":    "LR_split_train_images.json",
            },
            "val": {
                "questions": "LR_split_val_questions.json",
                "answers":   "LR_split_val_answers.json",
                "images":    "LR_split_val_images.json",
            },
            "test": {
                "questions": "LR_split_test_questions.json",
                "answers":   "LR_split_test_answers.json",
                "images":    "LR_split_test_images.json",
            },
        },
        "hr": {
            "train": {
                "questions": "USGS_split_train_questions.json",
                "answers":   "USGS_split_train_answers.json",
                "images":    "USGS_split_train_images.json",
            },
            "val": {
                "questions": "USGS_split_val_questions.json",
                "answers":   "USGS_split_val_answers.json",
                "images":    "USGS_split_val_images.json",
            },
            "test": {
                "questions": "USGS_split_test_phili_questions.json",
                "answers":   "USGS_split_test_phili_answers.json",
                "images":    "USGS_split_test_phili_images.json",
            },
        },
    }

    _IMAGE_DIRS: Dict[str, str] = {"lr": "Images", "hr": "Data"}

    def __init__(self, root_dir: str, variant: str, split: str):
        root = Path(root_dir)
        tmpl = self._TEMPLATES[variant][split]

        q_data = _load_json(root / tmpl["questions"])
        a_data = _load_json(root / tmpl["answers"])
        i_data = _load_json(root / tmpl["images"])

        self.image_map: Dict[int, str] = {
            img["id"]: img["filename"]
            for img in i_data["images"]
            if img.get("active", True)
        }
        self.answer_map: Dict[int, str] = {
            ans["question_id"]: ans["answer"]
            for ans in a_data["answers"]
            if ans.get("active", True)
        }
        self.samples: List[Dict[str, Any]] = [
            q for q in q_data["questions"]
            if q["id"] in self.answer_map and q["img_id"] in self.image_map
        ]
        self.image_dir = root / self._IMAGE_DIRS[variant]

    def get_image_path(self, img_id: int) -> str:
        filename = self.image_map[img_id]
        base = self.image_dir / filename
        if base.exists():
            return str(base)
        for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
            for candidate in [
                self.image_dir / (filename + ext),
                self.image_dir / (Path(filename).stem + ext),
            ]:
                if candidate.exists():
                    return str(candidate)
        return str(base)


# ---------------------------------------------------------------------------
# RSVQADataset
# ---------------------------------------------------------------------------

class RSVQADataset(data.Dataset):
    """
    RSVQA dataset (HR and LR variants).

    Loads image-question-answer triplets from the official RSVQA JSON
    files and images from the accompanying image directory.

    Parameters
    ----------
    root_dir             : str   Path to RSVQA-LR or RSVQA-HR root.
    variant              : str   'hr' or 'lr'.
    split                : str   'train', 'val', or 'test'.
    transform            : callable, optional  Applied to image tensor [3,H,W].
    debug                : bool  Return synthetic data without disk reads.
    question_type_filter : str, optional  Filter by type: presence/comparison/count/area.
    max_samples          : int, optional  Truncate dataset.
    """

    VALID_VARIANTS = {"hr", "lr"}
    VALID_SPLITS   = {"train", "val", "test"}
    VALID_QTYPES   = {"presence", "comparison", "count", "area"}

    def __init__(
        self,
        root_dir: str,
        variant: str = "hr",
        split: str = "train",
        transform: Optional[Callable] = None,
        debug: bool = False,
        question_type_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        variant = variant.lower()
        split   = split.lower()

        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {self.VALID_VARIANTS}")
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}")
        if question_type_filter is not None:
            qf = question_type_filter.lower()
            if qf not in self.VALID_QTYPES:
                raise ValueError(f"question_type_filter must be one of {self.VALID_QTYPES}")
            question_type_filter = qf

        self.root_dir             = root_dir
        self.variant              = variant
        self.split                = split
        self.transform            = transform
        self.debug                = debug
        self.question_type_filter = question_type_filter
        self.max_samples          = max_samples

        if debug:
            n = max_samples or 64
            self._samples: List[Dict[str, Any]] = [
                {
                    "id":       i,
                    "question": f"[DEBUG] Is there a building in image {i}?",
                    "type":     "presence",
                    "img_id":   i,
                }
                for i in range(n)
            ]
            self._split_obj = None
        else:
            try:
                self._split_obj = _RSVQASplit(root_dir, variant, split)
                self._samples   = self._split_obj.samples
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"RSVQA-{variant.upper()} JSON files not found in '{root_dir}'.\n"
                    f"Original: {exc}"
                ) from exc

            if question_type_filter is not None:
                self._samples = [
                    s for s in self._samples
                    if s.get("type", "").lower() == question_type_filter
                ]

            if max_samples is not None:
                self._samples = self._samples[:max_samples]

        logger.info(
            "RSVQADataset [%s-%s] %s: %d samples",
            variant.upper(), split, "(debug)" if debug else "", len(self._samples),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict with keys:
          image          : FloatTensor [3, H, W] in [0, 1]
          question       : str
          answer         : str
          question_type  : str  ('presence'|'comparison'|'count'|'area')
          image_path     : str
        """
        sample        = self._samples[idx]
        question      = sample.get("question", "")
        question_type = sample.get("type", "presence").lower()
        img_id        = sample.get("img_id", 0)
        q_id          = sample.get("id", idx)

        if self.debug:
            image      = torch.rand(3, 256, 256)
            answer     = "yes" if idx % 2 == 0 else "no"
            image_path = ""
        else:
            answer     = self._split_obj.answer_map.get(q_id, "")
            image_path = self._split_obj.get_image_path(img_id)
            try:
                image = _load_pil_image(image_path)
            except Exception as exc:
                warnings.warn(
                    f"Could not load '{image_path}' (idx={idx}): {exc}. Using zeros.",
                    RuntimeWarning, stacklevel=2,
                )
                image = torch.zeros(3, 256, 256)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image":         image,
            "question":      question,
            "answer":        answer,
            "question_type": question_type,
            "image_path":    image_path,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"variant='{self.variant}', split='{self.split}', "
            f"n={len(self)}, debug={self.debug})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_rsvqa_dataloader(
    dataset: RSVQADataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
) -> data.DataLoader:
    """Wrap RSVQADataset in a DataLoader with sensible defaults."""
    if shuffle is None:
        shuffle = dataset.split == "train"
    if num_workers > 0 and os.name == "nt":
        warnings.warn(
            "On Windows num_workers>0 may cause issues. Consider num_workers=0.",
            RuntimeWarning, stacklevel=2,
        )
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=(dataset.split == "train"),
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for variant in ("lr", "hr"):
        for split in ("train", "val", "test"):
            ds = RSVQADataset(
                root_dir="/nonexistent", variant=variant,
                split=split, debug=True, max_samples=8,
            )
            s = ds[0]
            assert s["image"].shape[0] == 3
            print(
                f"  RSVQA-{variant.upper()} [{split:5s}]  "
                f"n={len(ds)}  image={tuple(s['image'].shape)}  "
                f"q_type={s['question_type']}  OK"
            )
    print("RSVQA smoke-test PASSED")
