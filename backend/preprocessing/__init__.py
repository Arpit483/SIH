"""
backend/preprocessing/__init__.py
==================================
SatQuery AI — Preprocessing Package

Public API
----------
CompatibilityChecker
    Validates sensor type, band layout, and metadata compatibility
    before feeding imagery into the model pipeline.

SensorNormalizer
    Applies per-sensor normalisation (Sentinel-2 reflectance scaling,
    Sentinel-1 dB clipping, Cartosat-2S DOS correction, RISAT speckle
    filtering) as defined in ``training/configs/unified_config.yaml``.

GeoTIFFHandler
    Thin wrapper around rasterio / GDAL for reading, reprojecting, and
    windowed-reading of GeoTIFF files.  Returns numpy arrays with
    attached CRS / transform metadata.

Usage
-----
>>> from backend.preprocessing import CompatibilityChecker, SensorNormalizer, GeoTIFFHandler
>>> handler = GeoTIFFHandler()
>>> arr, meta = handler.read('/path/to/image.tif')
>>> checker = CompatibilityChecker()
>>> checker.check(meta)
>>> normalizer = SensorNormalizer(sensor='sentinel2')
>>> arr_norm = normalizer.normalize(arr)
"""

from .compatibility_checker import CompatibilityChecker
from .sensor_normalizer import SensorNormalizer
from .geotiff_handler import GeoTIFFHandler

__all__ = [
    "CompatibilityChecker",
    "SensorNormalizer",
    "GeoTIFFHandler",
]
