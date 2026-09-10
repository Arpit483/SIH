"""
cdvqa_dataset.py
----------------
CDVQA - Change Detection Visual Question Answering Dataset for the
SatQuery AI project (SIH26167 - ISRO remote sensing VLM).

CDVQA is a VQA dataset on bi-temporal remote sensing image pairs
for change detection and change-related question answering.

Reference : CDVQA (Yuan et al., IEEE GRSL 2022)
Dataset   : https://github.com/YZHJesse/CDVQA

Dataset statistics:
  ~2,968 bi-temporal image pairs (512x512 RGB)
  ~122,000 QA pairs (change existence, type, count, colour, size, location)
  Binary change masks per image pair

Expected directory layout:
  root_dir/
    image1/         earlier-timestamp images  *.png
    image2/         later-timestamp images    *.png
    change_mask/    binary change masks       *.png
    QA/
      train.json
      val.json
      test.json

JSON schema (per scene, flattened to one row per QA pair):
{
  "scene_id": str,
  "image1": "filename.png",
  "image2": "filename.png",
  "mask":   "filename.png",   (optional key)
  "QA": [{"question": str, "answer": str}, ...]
}
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
# Image / mask loading helpers
# ---------------------------------------------------------------------------

def _load_rgb_tensor(path: str) -> torch.Tensor:
    """Load RGB image -> [3, H, W] float32 in [0, 1]."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
    except ImportError as exc:
        raise ImportError("Pillow required: pip install Pillow") from exc


def _load_mask_tensor(path: str) -> torch.Tensor:
    """Load greyscale mask -> [1, H, W] float32 binary {0, 1}."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("L")
        arr = (np.array(img, dtype=np.float32) > 0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)
    except ImportError as exc:
        raise ImportError("Pillow required: pip install Pillow") from exc


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sample builder (JSON -> flat list)
# ---------------------------------------------------------------------------

def _build_samples(
    json_path: str,
    image1_dir: Path,
    image2_dir: Path,
    mask_dir: Path,
) -> List[Dict[str, Any]]:
    """Flatten CDVQA JSON: one dict per (scene, question) pair."""
    records = _load_json(json_path)
    samples: List[Dict[str, Any]] = []
    for rec in records:
        scene_id  = rec.get("scene_id", "")
        img1_file = str(image1_dir / rec.get("image1", ""))
        img2_file = str(image2_dir / rec.get("image2", ""))
        mask_key  = rec.get("mask", rec.get("image1", ""))
        mask_file = str(mask_dir / mask_key) if mask_key else None
        for qa in rec.get("QA", []):
            samples.append({
                "scene_id": scene_id,
                "image1":   img1_file,
                "image2":   img2_file,
                "mask":     mask_file,
                "question": qa.get("question", ""),
                "answer":   qa.get("answer",   ""),
            })
    return samples


# ---------------------------------------------------------------------------
# CDVQADataset
# ---------------------------------------------------------------------------

class CDVQADataset(data.Dataset):
    """
    CDVQA - Change Detection VQA on bi-temporal image pairs.

    Each sample: earlier image, later image, QA pair, optional change mask.

    Parameters
    ----------
    root_dir            : str   Path to CDVQA root directory.
    split               : str   'train', 'val', or 'test'.
    return_change_mask  : bool  Include binary change mask [1,H,W] in output.
    transform           : callable, optional  Applied to each image tensor.
    debug               : bool  Synthetic data, no disk reads.
    max_samples         : int, optional  Truncate samples.
    image1_subdir       : str   Sub-dir for earlier images (default 'image1').
    image2_subdir       : str   Sub-dir for later images  (default 'image2').
    mask_subdir         : str   Sub-dir for masks (default 'change_mask').
    qa_subdir           : str   Sub-dir for JSON QA files (default 'QA').
    """

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        return_change_mask: bool = True,
        transform: Optional[Callable] = None,
        debug: bool = False,
        max_samples: Optional[int] = None,
        image1_subdir: str = "image1",
        image2_subdir: str = "image2",
        mask_subdir: str = "change_mask",
        qa_subdir: str = "QA",
    ):
        super().__init__()
        split = split.lower()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.root_dir           = root_dir
        self.split              = split
        self.return_change_mask = return_change_mask
        self.transform          = transform
        self.debug              = debug
        self.max_samples        = max_samples

        if debug:
            n = max_samples or 64
            self._samples: List[Dict[str, Any]] = [
                {
                    "scene_id": f"scene_{i:04d}",
                    "image1":   "",
                    "image2":   "",
                    "mask":     None,
                    "question": f"[DEBUG] Is there any change in scene {i}?",
                    "answer":   "yes" if i % 2 == 0 else "no",
                }
                for i in range(n)
            ]
            logger.info("CDVQADataset [DEBUG] split=%s n=%d", split, n)
        else:
            root       = Path(root_dir)
            img1_dir   = root / image1_subdir
            img2_dir   = root / image2_subdir
            mask_dir   = root / mask_subdir
            json_path  = root / qa_subdir / f"{split}.json"

            if not json_path.exists():
                raise FileNotFoundError(
                    f"CDVQA QA JSON not found: '{json_path}'. "
                    f"Please download the dataset to '{root_dir}'."
                )

            logger.info("Loading CDVQA %s split from '%s' ...", split, json_path)
            try:
                self._samples = _build_samples(
                    str(json_path), img1_dir, img2_dir, mask_dir
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse CDVQA JSON '{json_path}': {exc}"
                ) from exc

            if max_samples is not None:
                self._samples = self._samples[:max_samples]

            logger.info(
                "CDVQADataset [%s]: %d (scene, question) pairs loaded.",
                split, len(self._samples),
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict with keys:
          image1       : FloatTensor [C, H, W]  earlier timestamp, in [0,1]
          image2       : FloatTensor [C, H, W]  later  timestamp, in [0,1]
          question     : str
          answer       : str
          change_mask  : FloatTensor [1, H, W] binary {0,1}  (if return_change_mask)
          scene_id     : str
        """
        meta = self._samples[idx]

        if self.debug or not meta["image1"]:
            image1 = torch.rand(3, 512, 512)
            image2 = torch.rand(3, 512, 512)
        else:
            image1 = self._safe_load_image(meta["image1"], idx, "image1")
            image2 = self._safe_load_image(meta["image2"], idx, "image2")

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        result: Dict[str, Any] = {
            "image1":   image1,
            "image2":   image2,
            "question": meta["question"],
            "answer":   meta["answer"],
            "scene_id": meta["scene_id"],
        }

        if self.return_change_mask:
            if self.debug or not meta.get("mask"):
                h, w = image1.shape[-2], image1.shape[-1]
                mask = (torch.rand(1, h, w) > 0.5).float()
            else:
                mask = self._safe_load_mask(meta["mask"], idx)
            result["change_mask"] = mask

        return result

    def _safe_load_image(self, path: str, idx: int, role: str) -> torch.Tensor:
        try:
            return _load_rgb_tensor(path)
        except Exception as exc:
            warnings.warn(
                f"CDVQA: cannot load {role} '{path}' (idx={idx}): {exc}. Using zeros.",
                RuntimeWarning, stacklevel=3,
            )
            return torch.zeros(3, 512, 512)

    def _safe_load_mask(self, path: str, idx: int) -> torch.Tensor:
        try:
            return _load_mask_tensor(path)
        except Exception as exc:
            warnings.warn(
                f"CDVQA: cannot load mask '{path}' (idx={idx}): {exc}. Using zeros.",
                RuntimeWarning, stacklevel=3,
            )
            return torch.zeros(1, 512, 512)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split='{self.split}', n={len(self)}, "
            f"return_change_mask={self.return_change_mask}, "
            f"debug={self.debug})"
        )


# ---------------------------------------------------------------------------
# Collate / DataLoader
# ---------------------------------------------------------------------------

def cdvqa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate CDVQA samples. Stacks tensors; keeps text as lists."""
    result: Dict[str, Any] = {
        "image1":   torch.stack([b["image1"]   for b in batch]),
        "image2":   torch.stack([b["image2"]   for b in batch]),
        "question": [b["question"] for b in batch],
        "answer":   [b["answer"]   for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
    }
    if "change_mask" in batch[0]:
        result["change_mask"] = torch.stack([b["change_mask"] for b in batch])
    return result


def create_cdvqa_dataloader(
    dataset: CDVQADataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
) -> data.DataLoader:
    """
    Wrap CDVQADataset in a DataLoader.

    Defaults to batch_size=8 (bi-temporal pairs are memory-intensive).
    """
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
        collate_fn=cdvqa_collate_fn,
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for split in ("train", "val", "test"):
        for mask in (True, False):
            ds = CDVQADataset(
                root_dir="/nonexistent", split=split,
                return_change_mask=mask, debug=True, max_samples=8,
            )
            s = ds[0]
            assert s["image1"].shape[0] == 3
            assert s["image2"].shape == s["image1"].shape
            if mask:
                assert "change_mask" in s
            print(
                f"  split={split:5s}  mask={str(mask):5s}  "
                f"n={len(ds):4d}  image1={tuple(s['image1'].shape)}  "
                f"scene_id={s['scene_id']}  OK"
            )
    loader = create_cdvqa_dataloader(
        CDVQADataset(
            root_dir="/nonexistent", split="train", debug=True, max_samples=16
        ),
        batch_size=4, num_workers=0,
    )
    batch = next(iter(loader))
    print(
        f"  Batch image1={tuple(batch['image1'].shape)}  "
        f"mask={tuple(batch['change_mask'].shape)}"
    )
    print("CDVQA smoke-test PASSED")
