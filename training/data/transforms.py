"""
transforms.py
-------------
Per-sensor image transform pipelines for the SatQuery AI project (SIH26167).

Supported sensors
-----------------
  Sentinel-2 MSI   - Sentinel2Transform
  Sentinel-1 SAR   - Sentinel1Transform
  Cartosat-2S VNIR - Cartosat2STransform
  RISAT SAR        - RISATTransform

Training augmentation
---------------------
  TrainingAugmentation - random flips, 90 deg rotations, scale jitter

Factory
-------
  get_transform(sensor_type, split) -> composed transform callable
"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared resize helper (bilinear, align_corners=False)
# ---------------------------------------------------------------------------

def _resize(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Resize [C, H, W] tensor to [C, size, size] via bilinear interpolation."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(
            tensor.float(), size=(size, size),
            mode="bilinear", align_corners=False,
        )
        return tensor.squeeze(0)
    return F.interpolate(
        tensor.float(), size=(size, size),
        mode="bilinear", align_corners=False,
    )


# ---------------------------------------------------------------------------
# Sentinel-2 Transform
# ---------------------------------------------------------------------------

class Sentinel2Transform:
    """
    Normalize Sentinel-2 multi-spectral imagery and select VNIR bands.

    Steps:
      1. Select bands by index from the full 13-band tensor.
      2. Normalize: clip((DN - 1000) / 10000, 0, 1)
      3. Resize to image_size x image_size (bilinear).

    Parameters
    ----------
    image_size : int   Output spatial resolution (default 224).
    bands      : list  0-based band indices to select. Default [1,2,3,7]
                       = B2 (Blue), B3 (Green), B4 (Red), B8 (NIR).
    """

    def __init__(
        self,
        image_size: int = 224,
        bands: List[int] = None,
    ):
        self.image_size = image_size
        self.bands = bands if bands is not None else [1, 2, 3, 7]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tensor : torch.Tensor  Shape [C, H, W] raw S2 DN values.

        Returns
        -------
        torch.Tensor  Shape [len(bands), image_size, image_size] in [0, 1].
        """
        tensor = tensor.float()
        if tensor.shape[0] > len(self.bands):
            indices = torch.tensor(self.bands, dtype=torch.long)
            tensor = tensor[indices]
        tensor = (tensor - 1000.0) / 10_000.0
        tensor = tensor.clamp(0.0, 1.0)
        tensor = _resize(tensor, self.image_size)
        return tensor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"image_size={self.image_size}, bands={self.bands})"
        )


# ---------------------------------------------------------------------------
# Sentinel-1 Transform
# ---------------------------------------------------------------------------

class Sentinel1Transform:
    """
    Normalize Sentinel-1 SAR backscatter (dB) to [0, 1].

    Steps:
      1. Ch 0 (VV): clip(val, vv_min, vv_max) -> [0, 1]
      2. Ch 1 (VH): clip(val, vh_min, vh_max) -> [0, 1]
      3. Optional speckle augmentation (training).
      4. Resize to image_size x image_size.

    Parameters
    ----------
    image_size      : int    Output resolution (default 224).
    vv_range        : tuple  (min_dB, max_dB) for VV (default (-25, 0)).
    vh_range        : tuple  (min_dB, max_dB) for VH (default (-32, -5)).
    speckle_augment : bool   Multiplicative speckle noise for training.
    """

    def __init__(
        self,
        image_size: int = 224,
        vv_range: Tuple[float, float] = (-25.0, 0.0),
        vh_range: Tuple[float, float] = (-32.0, -5.0),
        speckle_augment: bool = False,
    ):
        self.image_size      = image_size
        self.vv_range        = vv_range
        self.vh_range        = vh_range
        self.speckle_augment = speckle_augment

    @staticmethod
    def _linear_norm(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return (x.clamp(lo, hi) - lo) / (hi - lo)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tensor : torch.Tensor  Shape [2, H, W] dB backscatter (VV, VH).

        Returns
        -------
        torch.Tensor  Shape [2, image_size, image_size] in [0, 1].
        """
        tensor = tensor.float()

        if self.speckle_augment:
            lin   = 10.0 ** (tensor / 10.0)
            noise = torch.randn_like(lin).mul_(0.1).exp_()
            tensor = 10.0 * torch.log10((lin * noise).clamp(min=1e-10))

        channels = []
        if tensor.shape[0] >= 1:
            channels.append(self._linear_norm(tensor[0:1], *self.vv_range))
        if tensor.shape[0] >= 2:
            channels.append(self._linear_norm(tensor[1:2], *self.vh_range))
        for c in range(2, tensor.shape[0]):
            channels.append(tensor[c:c+1].clamp(0.0, 1.0))

        tensor = torch.cat(channels, dim=0)
        tensor = _resize(tensor, self.image_size)
        return tensor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"image_size={self.image_size}, "
            f"vv_range={self.vv_range}, "
            f"vh_range={self.vh_range}, "
            f"speckle_augment={self.speckle_augment})"
        )


# ---------------------------------------------------------------------------
# Cartosat-2S Transform
# ---------------------------------------------------------------------------

class Cartosat2STransform:
    """
    Radiometric correction and normalization for Cartosat-2S VNIR data.

    Band order: [B, G, R, NIR] (11-bit DN, range 0-2047).

    Applies simplified DOS-1 atmospheric correction then rescales to
    reflectance proxy in [0, 1], compatible with S2 VNIR conventions.

    Parameters
    ----------
    image_size : int    Output resolution (default 224).
    dn_max     : float  Max 11-bit DN value (default 2047).
    apply_dos  : bool   Apply Dark Object Subtraction (default True).
    """

    def __init__(
        self,
        image_size: int = 224,
        dn_max: float = 2047.0,
        apply_dos: bool = True,
    ):
        self.image_size = image_size
        self.dn_max     = dn_max
        self.apply_dos  = apply_dos

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tensor : torch.Tensor  Shape [4, H, W] Cartosat-2S DN [B,G,R,NIR].

        Returns
        -------
        torch.Tensor  Shape [4, image_size, image_size] in [0, 1].
        """
        tensor = tensor.float()
        if self.apply_dos:
            band_min = tensor.flatten(1).min(dim=1).values.view(-1, 1, 1).clamp(min=0.0)
            tensor = tensor - band_min
        tensor = (tensor / self.dn_max).clamp(0.0, 1.0)
        tensor = _resize(tensor, self.image_size)
        return tensor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"image_size={self.image_size}, "
            f"dn_max={self.dn_max}, "
            f"apply_dos={self.apply_dos})"
        )


# ---------------------------------------------------------------------------
# RISAT Transform
# ---------------------------------------------------------------------------

_RISAT_RANGES: dict = {
    "c": (-30.0, 5.0),    # RISAT-1 C-band (5.35 GHz)
    "x": (-25.0, 10.0),   # RISAT-2/2B X-band (9.5 GHz)
}


class RISATTransform:
    """
    Normalize RISAT SAR backscatter (dB) to [0, 1].

    Parameters
    ----------
    band       : str    'c' (C-band, RISAT-1) or 'x' (X-band, RISAT-2/2B).
    image_size : int    Output resolution (default 224).
    db_range   : tuple  Override default dB range (lo, hi).
    """

    def __init__(
        self,
        band: str = "c",
        image_size: int = 224,
        db_range: Optional[Tuple[float, float]] = None,
    ):
        band = band.lower()
        if band not in _RISAT_RANGES:
            raise ValueError(f"band must be 'c' or 'x', got '{band}'")
        self.band       = band
        self.image_size = image_size
        self.db_range   = db_range if db_range is not None else _RISAT_RANGES[band]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tensor : torch.Tensor  Shape [C, H, W] SAR backscatter in dB.

        Returns
        -------
        torch.Tensor  Shape [C, image_size, image_size] in [0, 1].
        """
        tensor = tensor.float()
        lo, hi = self.db_range
        tensor = (tensor.clamp(lo, hi) - lo) / (hi - lo)
        tensor = _resize(tensor, self.image_size)
        return tensor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"band='{self.band}', "
            f"image_size={self.image_size}, "
            f"db_range={self.db_range})"
        )


# ---------------------------------------------------------------------------
# Training Augmentation
# ---------------------------------------------------------------------------

class TrainingAugmentation:
    """
    Geometric augmentations for remote-sensing training imagery.

    Operations (all random):
      - Horizontal flip (p=0.5)
      - Vertical flip   (p=0.5)
      - 90 deg-multiple rotation (k in {0,1,2,3})
      - Scale jitter (random resized crop in scale_range, then resize back)

    Radiometric augmentations are omitted to preserve physical DN meaning.

    Parameters
    ----------
    image_size         : int    Output resolution (default 224).
    scale_range        : tuple  (min_scale, max_scale) for random crop (default (0.8, 1.2)).
    horizontal_flip_p  : float  Flip probability (default 0.5).
    vertical_flip_p    : float  Flip probability (default 0.5).
    rotation           : bool   Enable 90 deg rotations (default True).
    """

    def __init__(
        self,
        image_size: int = 224,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation: bool = True,
    ):
        self.image_size        = image_size
        self.scale_range       = scale_range
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p   = vertical_flip_p
        self.rotation          = rotation

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tensor : torch.Tensor  Shape [C, H, W].

        Returns
        -------
        torch.Tensor  Shape [C, image_size, image_size].
        """
        if random.random() < self.horizontal_flip_p:
            tensor = tensor.flip(-1)
        if random.random() < self.vertical_flip_p:
            tensor = tensor.flip(-2)
        if self.rotation:
            k = random.randint(0, 3)
            if k > 0:
                tensor = torch.rot90(tensor, k=k, dims=[-2, -1])
        tensor = self._scale_jitter(tensor)
        return tensor

    def _scale_jitter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Random-crop then resize to image_size."""
        C, H, W = tensor.shape
        scale  = random.uniform(*self.scale_range)
        crop_h = min(H, max(1, int(round(H / scale))))
        crop_w = min(W, max(1, int(round(W / scale))))

        if scale > 1.0:
            pad_h = max(0, int(math.ceil((H * scale - H) / 2)))
            pad_w = max(0, int(math.ceil((W * scale - W) / 2)))
            tensor = F.pad(
                tensor.unsqueeze(0),
                (pad_w, pad_w, pad_h, pad_h),
                mode="reflect",
            ).squeeze(0)
            C, H, W = tensor.shape
            crop_h = min(H, int(round(H / scale)))
            crop_w = min(W, int(round(W / scale)))

        top  = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))
        tensor = tensor[:, top:top + crop_h, left:left + crop_w]
        tensor = _resize(tensor, self.image_size)
        return tensor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"image_size={self.image_size}, "
            f"scale_range={self.scale_range})"
        )


# ---------------------------------------------------------------------------
# Compose helper
# ---------------------------------------------------------------------------

class _Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

    def __repr__(self) -> str:
        inner = "\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {inner}\n])"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_transform(
    sensor_type: str,
    split: str = "train",
    image_size: int = 224,
) -> Callable:
    """
    Return the appropriate transform for a given sensor and dataset split.

    Parameters
    ----------
    sensor_type : str  One of 'sentinel2', 'sentinel1', 'cartosat2s',
                        'risat_c', 'risat_x'.
    split       : str  'train' (augmentation on) | 'val'/'test' (det. only).
    image_size  : int  Output spatial resolution (default 224).

    Returns
    -------
    callable  Composed transform: [C, H, W] tensor -> [C', size, size] tensor.
    """
    sensor_type = sensor_type.lower().replace("-", "_").replace(" ", "_")
    is_train    = split.lower() == "train"

    if sensor_type == "sentinel2":
        base = Sentinel2Transform(image_size=image_size)
        if is_train:
            return _Compose([base, TrainingAugmentation(image_size=image_size)])
        return base

    elif sensor_type == "sentinel1":
        base = Sentinel1Transform(image_size=image_size, speckle_augment=is_train)
        if is_train:
            return _Compose([base, TrainingAugmentation(image_size=image_size)])
        return base

    elif sensor_type in ("cartosat2s", "cartosat_2s", "cartosat2"):
        base = Cartosat2STransform(image_size=image_size)
        if is_train:
            return _Compose([base, TrainingAugmentation(image_size=image_size)])
        return base

    elif sensor_type == "risat_c":
        base = RISATTransform(band="c", image_size=image_size)
        if is_train:
            return _Compose([base, TrainingAugmentation(image_size=image_size)])
        return base

    elif sensor_type == "risat_x":
        base = RISATTransform(band="x", image_size=image_size)
        if is_train:
            return _Compose([base, TrainingAugmentation(image_size=image_size)])
        return base

    else:
        valid = ["sentinel2", "sentinel1", "cartosat2s", "risat_c", "risat_x"]
        raise ValueError(f"Unknown sensor_type '{sensor_type}'. Valid: {valid}")


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sensors = [
        ("sentinel2",  (13, 120, 120)),
        ("sentinel1",  (2,  120, 120)),
        ("cartosat2s", (4,  512, 512)),
        ("risat_c",    (2,  256, 256)),
        ("risat_x",    (1,  256, 256)),
    ]
    for sensor, shape in sensors:
        for split in ("train", "val"):
            t = get_transform(sensor, split=split, image_size=224)
            x = torch.rand(*shape) * 2000
            y = t(x)
            assert y.shape[-1] == 224 and y.shape[-2] == 224
            print(f"  {sensor:14s} [{split:5s}]  {tuple(shape)} -> {tuple(y.shape)}  OK")
    print("All transform smoke-tests PASSED")
