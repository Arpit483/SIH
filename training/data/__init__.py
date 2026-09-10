"""
training/data/__init__.py
==========================
SatQuery AI — Training Data Package

Public API
----------
Dataset classes
~~~~~~~~~~~~~~~
BigEarthNetDataset
    Loads BigEarthNet v2 (``BIFOLD-BigEarthNetv2-0/BigEarthNet.txt``)
    for VQA, captioning, and grounding tasks.

RSVQAHRDataset
    High-resolution RSVQA benchmark dataset (local path).

RSVQALRDataset
    Low-resolution RSVQA benchmark dataset (local path).

VRSBenchDataset
    VRSBench multi-task remote-sensing benchmark (local path).

CDVQADataset
    Change-Detection VQA dataset — pairs of bi-temporal images with
    natural-language change questions (local path).

SARLANGDataset
    SARLANG-1M SAR image-language dataset
    (``YiminJimmy/SARLANG-1M``), used as SAR supplement.

RS5MDataset
    RS5M large-scale remote-sensing image-text dataset
    (``Zilun/RS5M``), used as fallback/pre-training corpus.

AdaptLLMRSDataset
    AdaptLLM remote-sensing visual instruction dataset
    (``AdaptLLM/remote-sensing-visual-instructions``), used as fallback.

Collators / utilities
~~~~~~~~~~~~~~~~~~~~~
MultiTaskCollator
    Pads and batches samples from heterogeneous task datasets.

SensorAwareTransform
    Applies per-sensor augmentation and normalisation pipelines
    defined in ``unified_config.yaml``.

Usage
-----
>>> from training.data import BigEarthNetDataset, MultiTaskCollator
>>> ds = BigEarthNetDataset(hf_id='BIFOLD-BigEarthNetv2-0/BigEarthNet.txt',
...                         max_samples=200_000)
>>> loader = torch.utils.data.DataLoader(ds, collate_fn=MultiTaskCollator())
"""

from .bigearthnet_dataset import BigEarthNetDataset
from .rsvqa_dataset import RSVQAHRDataset, RSVQALRDataset
from .vrsbench_dataset import VRSBenchDataset
from .cdvqa_dataset import CDVQADataset
from .sarlang_dataset import SARLANGDataset
from .rs5m_dataset import RS5MDataset
from .adaptllm_rs_dataset import AdaptLLMRSDataset
from .collator import MultiTaskCollator
from .transforms import SensorAwareTransform

__all__ = [
    # Dataset classes
    "BigEarthNetDataset",
    "RSVQAHRDataset",
    "RSVQALRDataset",
    "VRSBenchDataset",
    "CDVQADataset",
    "SARLANGDataset",
    "RS5MDataset",
    "AdaptLLMRSDataset",
    # Utilities
    "MultiTaskCollator",
    "SensorAwareTransform",
]
