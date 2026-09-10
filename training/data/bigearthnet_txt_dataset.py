"""
bigearthnet_txt_dataset.py
--------------------------
BigEarthNet.txt PyTorch Dataset for the SatQuery AI project (SIH26167).

HuggingFace Hub ID : BIFOLD-BigEarthNetv2-0/BigEarthNet.txt
GitHub reference   : rsim-tu-berlin/BigEarthNet.txt (BENTxTDataset)

Dataset statistics:
  - 464,044 co-registered Sentinel-1 SAR + Sentinel-2 MSI image pairs
  - 9.6M text annotations spanning VQA, captioning, grounding, and retrieval tasks

Image preprocessing pipeline (external):
  Images are downloaded from bigearth.net and preprocessed with rico-hdl
  into LMDB or safetensors format prior to training.

Normalization conventions:
  Sentinel-2 (S2): clip((DN - 1000) / 10000, 0, 1)  -- bands B2,B3,B4,B8
  Sentinel-1 (S1): VV dB in [-25, 0]  -> [0, 1]
                   VH dB in [-32, -5] -> [0, 1]

DEBUG mode:
  Text annotations are loaded from HuggingFace; image tensors are replaced
  with torch.rand() fakes so the pipeline can be exercised without images.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.utils.data as data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------

def _try_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
        return load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for BigEarthNetTxtDataset. "
            "Install it with:  pip install datasets"
        ) from exc


def _try_import_lmdb():
    try:
        import lmdb  # type: ignore
        return lmdb
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_s2(tensor: torch.Tensor) -> torch.Tensor:
    """Sentinel-2 DN normalization: clip((DN - 1000) / 10000, 0, 1)."""
    tensor = tensor.float()
    tensor = (tensor - 1000.0) / 10_000.0
    return tensor.clamp(0.0, 1.0)


def _normalize_s1(tensor: torch.Tensor) -> torch.Tensor:
    """
    Sentinel-1 dB backscatter normalization.
    VV: clip([-25, 0]) -> [0, 1]
    VH: clip([-32, -5]) -> [0, 1]
    """
    tensor = tensor.float()
    channels = []
    if tensor.shape[0] >= 1:
        vv = (tensor[0:1].clamp(-25.0, 0.0) + 25.0) / 25.0
        channels.append(vv)
    if tensor.shape[0] >= 2:
        vh = (tensor[1:2].clamp(-32.0, -5.0) + 32.0) / 27.0
        channels.append(vh)
    for c in range(2, tensor.shape[0]):
        channels.append(tensor[c:c+1].clamp(0.0, 1.0))
    if channels:
        tensor = torch.cat(channels, dim=0)
    return tensor.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# LMDB image loader
# ---------------------------------------------------------------------------

class _LMDBImageLoader:
    """Thin wrapper around a rico-hdl LMDB environment."""

    def __init__(self, lmdb_path: str):
        lmdb = _try_import_lmdb()
        if lmdb is None:
            raise ImportError("The `lmdb` package is required: pip install lmdb")
        self._env = lmdb.open(
            lmdb_path, readonly=True, lock=False,
            readahead=False, meminit=False,
        )

    def read(self, patch_id: str) -> Optional[torch.Tensor]:
        import numpy as np  # type: ignore
        key = patch_id.encode("utf-8")
        with self._env.begin(write=False) as txn:
            raw = txn.get(key)
        if raw is None:
            return None
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        try:
            tensor = torch.from_numpy(arr).reshape(-1, 120, 120)
        except RuntimeError:
            tensor = torch.from_numpy(arr)
        return tensor

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class BigEarthNetTxtDataset(data.Dataset):
    """
    BigEarthNet.txt dataset wrapper.

    Loads text annotations from HuggingFace Hub
    (``BIFOLD-BigEarthNetv2-0/BigEarthNet.txt``) and images from local
    LMDB files prepared by rico-hdl.

    In DEBUG mode the HF text partition is still loaded but image tensors
    are replaced with ``torch.rand`` fakes of the expected shape.

    Parameters
    ----------
    hf_split : str
        HuggingFace split: 'train', 'validation', or 'test'.
    task_filter : str
        'vqa', 'captioning', 'grounding', 'retrieval', or 'all'.
    s1_lmdb_path : str, optional
        Path to the Sentinel-1 LMDB created by rico-hdl.
    s2_lmdb_path : str, optional
        Path to the Sentinel-2 LMDB created by rico-hdl.
    max_samples : int, optional
        Truncate dataset to this many samples.
    debug : bool
        Use torch.rand image tensors; still fetches HF annotations.
    transform_s1 : callable, optional
        Extra transform applied after built-in S1 normalization.
    transform_s2 : callable, optional
        Extra transform applied after built-in S2 normalization.
    """

    _S2_CHANNELS = 4    # B2, B3, B4, B8
    _S1_CHANNELS = 2    # VV, VH
    _DEFAULT_HW  = 120  # BigEarthNet default spatial resolution

    _HF_REPO_ID    = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
    _COL_PATCH_ID  = "patch_id"
    _COL_QUESTION  = "question"
    _COL_ANSWER    = "answer"
    _COL_TASK_TYPE = "task_type"
    _COL_BBOX      = "bbox"

    def __init__(
        self,
        hf_split: str = "train",
        task_filter: str = "all",
        s1_lmdb_path: Optional[str] = None,
        s2_lmdb_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        debug: bool = False,
        transform_s1=None,
        transform_s2=None,
    ):
        super().__init__()
        self.hf_split      = hf_split
        self.task_filter   = task_filter.lower()
        self.s1_lmdb_path  = s1_lmdb_path
        self.s2_lmdb_path  = s2_lmdb_path
        self.max_samples   = max_samples
        self.debug         = debug
        self.transform_s1  = transform_s1
        self.transform_s2  = transform_s2

        valid_splits = {"train", "validation", "test"}
        if hf_split not in valid_splits:
            raise ValueError(f"hf_split must be one of {valid_splits}, got '{hf_split}'")

        valid_tasks = {"all", "vqa", "captioning", "grounding", "retrieval"}
        if self.task_filter not in valid_tasks:
            raise ValueError(f"task_filter must be one of {valid_tasks}")

        # ------------------------------------------------------------------
        # Load HuggingFace annotations
        # ------------------------------------------------------------------
        logger.info("Loading BigEarthNet.txt annotations (split=%s) ...", hf_split)
        load_dataset = _try_import_datasets()
        hf_dataset = load_dataset(self._HF_REPO_ID, split=hf_split)

        if self.task_filter != "all":
            hf_dataset = hf_dataset.filter(
                lambda row: (row.get(self._COL_TASK_TYPE) or "").lower()
                == self.task_filter
            )

        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        self._hf_dataset = hf_dataset
        logger.info("BigEarthNetTxtDataset: %d rows loaded.", len(hf_dataset))

        # ------------------------------------------------------------------
        # Image loaders (skipped in debug mode)
        # ------------------------------------------------------------------
        self._s1_loader: Optional[_LMDBImageLoader] = None
        self._s2_loader: Optional[_LMDBImageLoader] = None

        if not debug:
            for attr, path, name in [
                ("_s2_loader", s2_lmdb_path, "S2"),
                ("_s1_loader", s1_lmdb_path, "S1"),
            ]:
                if path:
                    try:
                        setattr(self, attr, _LMDBImageLoader(path))
                        logger.info("%s LMDB opened: %s", name, path)
                    except Exception as exc:
                        warnings.warn(
                            f"Could not open {name} LMDB at '{path}': {exc}. "
                            "Falling back to zero tensors.",
                            RuntimeWarning, stacklevel=2,
                        )

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict with keys:
          s2_tensor  : FloatTensor [4, H, W]   bands B2,B3,B4,B8  in [0,1]
          s1_tensor  : FloatTensor [2, H, W]   channels VV,VH     in [0,1]
          question   : str
          answer     : str
          task_type  : str  ('vqa'/'captioning'/'grounding'/'retrieval')
          bbox       : Optional[List[float]]  [cx,cy,w,h] normalised (grounding)
          patch_id   : str
        """
        row = self._hf_dataset[idx]

        patch_id  = str(row.get(self._COL_PATCH_ID,  ""))
        question  = str(row.get(self._COL_QUESTION,  ""))
        answer    = str(row.get(self._COL_ANSWER,    ""))
        task_type = str(row.get(self._COL_TASK_TYPE, "vqa")).lower()
        raw_bbox  = row.get(self._COL_BBOX, None)

        bbox: Optional[List[float]] = None
        if raw_bbox is not None:
            try:
                bbox = [float(v) for v in raw_bbox]
            except (TypeError, ValueError):
                bbox = None

        s2_tensor = self._load_s2(patch_id)
        s1_tensor = self._load_s1(patch_id)

        if self.transform_s2 is not None:
            s2_tensor = self.transform_s2(s2_tensor)
        if self.transform_s1 is not None:
            s1_tensor = self.transform_s1(s1_tensor)

        return {
            "s2_tensor": s2_tensor,
            "s1_tensor": s1_tensor,
            "question":  question,
            "answer":    answer,
            "task_type": task_type,
            "bbox":      bbox,
            "patch_id":  patch_id,
        }

    def _load_s2(self, patch_id: str) -> torch.Tensor:
        """Load and normalise the 4-band S2 patch; fallback to zeros."""
        H, W = self._DEFAULT_HW, self._DEFAULT_HW
        shape = (self._S2_CHANNELS, H, W)
        if self.debug:
            return torch.rand(*shape)
        if self._s2_loader is not None:
            try:
                raw = self._s2_loader.read(patch_id)
                if raw is not None:
                    if raw.shape[0] >= 8:
                        raw = raw[[1, 2, 3, 7]]
                    elif raw.shape[0] == 4:
                        pass
                    return _normalize_s2(raw)
            except Exception as exc:
                warnings.warn(
                    f"Failed to read S2 patch '{patch_id}': {exc}. Using zeros.",
                    RuntimeWarning, stacklevel=3,
                )
        logger.debug("S2 image unavailable for patch '%s', using zeros.", patch_id)
        return torch.zeros(*shape)

    def _load_s1(self, patch_id: str) -> torch.Tensor:
        """Load and normalise the 2-channel S1 patch; fallback to zeros."""
        H, W = self._DEFAULT_HW, self._DEFAULT_HW
        shape = (self._S1_CHANNELS, H, W)
        if self.debug:
            return torch.rand(*shape)
        if self._s1_loader is not None:
            try:
                raw = self._s1_loader.read(patch_id)
                if raw is not None:
                    if raw.shape[0] > 2:
                        raw = raw[:2]
                    return _normalize_s1(raw)
            except Exception as exc:
                warnings.warn(
                    f"Failed to read S1 patch '{patch_id}': {exc}. Using zeros.",
                    RuntimeWarning, stacklevel=3,
                )
        logger.debug("S1 image unavailable for patch '%s', using zeros.", patch_id)
        return torch.zeros(*shape)

    def __del__(self):
        for attr in ("_s1_loader", "_s2_loader"):
            loader = getattr(self, attr, None)
            if loader is not None:
                try:
                    loader.close()
                except Exception:
                    pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split='{self.hf_split}', "
            f"task_filter='{self.task_filter}', "
            f"n_samples={len(self)}, "
            f"debug={self.debug})"
        )


# ---------------------------------------------------------------------------
# Custom collate for variable bbox
# ---------------------------------------------------------------------------

def _ben_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate keeping bbox as a list (may be None per sample)."""
    return {
        "s2_tensor": torch.stack([b["s2_tensor"] for b in batch]),
        "s1_tensor": torch.stack([b["s1_tensor"] for b in batch]),
        "question":  [b["question"]  for b in batch],
        "answer":    [b["answer"]    for b in batch],
        "task_type": [b["task_type"] for b in batch],
        "bbox":      [b["bbox"]      for b in batch],
        "patch_id":  [b["patch_id"]  for b in batch],
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    dataset: BigEarthNetTxtDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> data.DataLoader:
    """
    Convenience factory wrapping dataset in a DataLoader.

    Parameters
    ----------
    dataset    : BigEarthNetTxtDataset
    batch_size : int  (default 8)
    num_workers: int  (default 4; auto-set to 0 on Windows with LMDB)
    shuffle    : bool (default: True for train split)
    pin_memory : bool
    drop_last  : bool

    Returns
    -------
    torch.utils.data.DataLoader
    """
    if shuffle is None:
        shuffle = dataset.hf_split == "train"

    if num_workers > 0 and os.name == "nt":
        if dataset._s1_loader is not None or dataset._s2_loader is not None:
            warnings.warn(
                "LMDB loaders are not fork-safe on Windows. "
                "Falling back to num_workers=0.",
                RuntimeWarning, stacklevel=2,
            )
            num_workers = 0

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=_ben_collate_fn,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("BigEarthNetTxtDataset smoke-test requires HF credentials.")
    print("Run with debug=True after instantiation.")
