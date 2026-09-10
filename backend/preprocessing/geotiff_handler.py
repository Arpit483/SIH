"""
geotiff_handler.py
==================
SatQuery AI – SIH-26167 (ISRO Remote Sensing VLM)

GeoTIFF / raster file handler.

Provides structured loading, metadata extraction, co-registration checking,
and thumbnail generation for GeoTIFF files.  Uses rasterio as the primary
backend; falls back gracefully to PIL (Pillow) for basic PNG/JPEG support
when rasterio is not available.
"""

from __future__ import annotations

import logging
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False
    warnings.warn("rasterio not installed – geotiff_handler will use PIL fallback.", ImportWarning, stacklevel=2)

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    warnings.warn("Pillow not installed – thumbnail generation will be disabled.", ImportWarning, stacklevel=2)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}

_BAND_RES_HINTS = [
    # (band_counts, min_res, max_res, hint)
    ((1,),   0.0,  1.5, "cartosat2s_pan"),
    ((4,),   1.5,  3.0, "cartosat2s_hrmx"),
    ((13,),  5.0, 20.0, "sentinel2"),
    ((2,),   5.0, 15.0, "sentinel1"),
    ((2,4),  3.0, 10.0, "risat1_cpol"),
    ((4,),   3.0, 10.0, "risat1_hybrid"),
    ((1,2),  0.5,  3.0, "risat2_xband"),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GeoTIFFInfo:
    """Structured metadata for a single raster file."""
    path: str
    bands: int
    width: int
    height: int
    crs: str
    resolution_x: float
    resolution_y: float
    acquisition_date: Optional[datetime]
    sensor_hint: str
    dtype: str
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top)
    nodata: Optional[float]
    extra_tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompatibilityReport:
    """Result of running a compatibility check on one or more raster inputs."""
    valid: bool
    input_type: str   # 'single_optical','single_sar','optical_sar_pair','bitemporal_pair'
    sensor_types: List[str]
    issues: List[str]
    warnings: List[str]
    recommended_tasks: List[str]
    file_infos: List[GeoTIFFInfo]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_sar_sensor(sensor_hint: str) -> bool:
    return sensor_hint in {"sentinel1", "risat1_cpol", "risat1_hybrid", "risat2_xband"}


def _is_optical_sensor(sensor_hint: str) -> bool:
    return sensor_hint in {"sentinel2", "cartosat2s_pan", "cartosat2s_hrmx"}


def _hint_from_tags(tags: dict, band_count: int, res_m: float) -> str:
    """Derive sensor hint from GDAL tags dict or fall back to band/res heuristics."""
    tags_l = {k.lower(): str(v).lower() for k, v in tags.items()}
    sat  = tags_l.get("satellite", tags_l.get("mission", ""))
    sens = tags_l.get("sensor",    tags_l.get("instrument", ""))
    mode = tags_l.get("mode",      tags_l.get("beam_mode", ""))

    if "sentinel-2" in sat or "sentinel2" in sat:
        return "sentinel2"
    if "sentinel-1" in sat or "sentinel1" in sat:
        return "sentinel1"
    if "cartosat" in sat:
        return "cartosat2s_pan" if ("pan" in (sens + mode) or band_count == 1) else "cartosat2s_hrmx"
    if "risat-1" in sat or "risat1" in sat:
        return "risat1_hybrid" if ("hybrid" in mode or band_count == 4) else "risat1_cpol"
    if "risat-2" in sat or "risat2" in sat:
        return "risat2_xband"

    for (band_set, min_r, max_r, hint) in _BAND_RES_HINTS:
        if band_count in band_set and min_r <= res_m <= max_r:
            return hint
    return "unknown"


def _parse_date_from_string(s: str) -> Optional[datetime]:
    """Try to parse a date from an arbitrary string using common ISRO/ESA formats."""
    patterns = [
        r"(\d{4})[_\-](\d{2})[_\-](\d{2})[T ](\d{2}):(\d{2}):(\d{2})",
        r"(\d{4})[_\-](\d{2})[_\-](\d{2})",
        r"(\d{2})[_\-](\d{2})[_\-](\d{4})",
        r"(\d{8})T(\d{6})",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            groups = m.groups()
            try:
                if len(groups) == 6:
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]),
                                    int(groups[3]), int(groups[4]), int(groups[5]))
                elif len(groups) == 3:
                    y, mo, d = int(groups[0]), int(groups[1]), int(groups[2])
                    if y < 100:
                        y, mo, d = int(groups[2]), int(groups[1]), int(groups[0])
                    return datetime(y, mo, d)
                elif len(groups) == 2 and len(groups[0]) == 8:
                    s0, s1 = groups[0], groups[1]
                    return datetime(int(s0[:4]), int(s0[4:6]), int(s0[6:8]),
                                    int(s1[:2]), int(s1[2:4]), int(s1[4:6]))
            except (ValueError, OverflowError):
                continue
    return None


def _metres_per_degree_approx(lat: float = 20.0) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def _resolution_to_metres(res_x: float, crs_str: str) -> float:
    crs_lower = crs_str.lower()
    if "degree" in crs_lower or "geographic" in crs_lower or "4326" in crs_lower:
        return abs(res_x) * _metres_per_degree_approx()
    return abs(res_x)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_geotiff(path: str) -> GeoTIFFInfo:
    """
    Load raster metadata from a GeoTIFF (or PNG/JPEG) and return a GeoTIFFInfo.

    Uses rasterio when available; falls back to PIL for non-GeoTIFF formats.

    Raises
    ------
    FileNotFoundError  – if file does not exist
    ValueError         – if extension is not supported
    RuntimeError       – if neither rasterio nor PIL can open the file
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if _HAS_RASTERIO and ext not in (".png", ".jpg", ".jpeg"):
        return _load_with_rasterio(path)
    if _HAS_PIL:
        return _load_with_pil(path)
    raise RuntimeError("Cannot open raster: neither rasterio nor Pillow is installed.")


def _load_with_rasterio(path: str) -> GeoTIFFInfo:
    with rasterio.open(path) as src:
        bands  = src.count
        width  = src.width
        height = src.height
        dtype  = str(src.dtypes[0]) if src.dtypes else "unknown"
        nodata = src.nodata
        crs_str = src.crs.to_string() if src.crs else "unknown"
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        b = src.bounds
        bounds = (b.left, b.bottom, b.right, b.top)
        tags = src.tags() or {}

    acq_date = _extract_acq_date_from_tags(tags, path)
    res_m = _resolution_to_metres(res_x, crs_str)
    sensor_hint = _hint_from_tags(tags, bands, res_m)

    logger.info("load_geotiff (rasterio) | %s | bands=%d | %dx%d | CRS=%s | sensor=%s",
                os.path.basename(path), bands, width, height, crs_str, sensor_hint)

    return GeoTIFFInfo(
        path=path, bands=bands, width=width, height=height,
        crs=crs_str, resolution_x=res_x, resolution_y=res_y,
        acquisition_date=acq_date, sensor_hint=sensor_hint,
        dtype=dtype, bounds=bounds, nodata=nodata, extra_tags=tags,
    )


def _load_with_pil(path: str) -> GeoTIFFInfo:
    img = PILImage.open(path)
    bands = len(img.getbands())
    width, height = img.size
    dtype = str(np.array(img).dtype)
    sensor_hint = _hint_from_tags({}, bands, 10.0)
    logger.info("load_geotiff (PIL) | %s | bands=%d | %dx%d",
                os.path.basename(path), bands, width, height)
    return GeoTIFFInfo(
        path=path, bands=bands, width=width, height=height,
        crs="unknown", resolution_x=float("nan"), resolution_y=float("nan"),
        acquisition_date=_parse_date_from_string(os.path.basename(path)),
        sensor_hint=sensor_hint, dtype=dtype,
        bounds=(float("nan"),) * 4, nodata=None, extra_tags={},
    )


def extract_acquisition_date(path: str) -> Optional[datetime]:
    """
    Extract the acquisition datetime for a raster file.
    Checks GDAL metadata tags first, then falls back to filename parsing.
    """
    if _HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                tags = src.tags() or {}
            dt = _extract_acq_date_from_tags(tags, path)
            if dt:
                return dt
        except Exception as e:
            logger.debug("extract_acquisition_date rasterio error: %s", e)
    return _parse_date_from_string(os.path.basename(path))


def _extract_acq_date_from_tags(tags: dict, path: str) -> Optional[datetime]:
    tag_keys = [
        "ACQUISITIONDATETIME", "ACQUISITION_DATE", "DATE_TIME",
        "TIFFTAG_DATETIME", "START_TIME", "SCENE_CENTER_TIME",
        "ProductStartTime", "sensingTime",
    ]
    for key in tag_keys:
        val = tags.get(key) or tags.get(key.lower())
        if val:
            dt = _parse_date_from_string(str(val))
            if dt:
                return dt
    return _parse_date_from_string(os.path.basename(path))


def check_coregistration(path1: str, path2: str, tolerance_m: float = 50.0) -> dict:
    """
    Check whether two raster files are co-registered to within tolerance_m metres.

    Checks CRS compatibility, pixel resolution match, bounding-box overlap
    (>= 80% of smaller image), and origin alignment.

    Returns dict with keys:
        'coregistered' (bool), 'overlap_fraction' (float),
        'crs_match' (bool), 'resolution_match' (bool),
        'origin_offset_m' (float), 'issues' (List[str])
    """
    result: dict = {
        "coregistered": False, "overlap_fraction": 0.0,
        "crs_match": False, "resolution_match": False,
        "origin_offset_m": float("inf"), "issues": [],
    }

    try:
        info1 = load_geotiff(path1)
        info2 = load_geotiff(path2)
    except Exception as e:
        result["issues"].append(f"Failed to load files: {e}")
        return result

    # CRS check
    if info1.crs == "unknown" or info2.crs == "unknown":
        result["issues"].append("CRS unknown for at least one file – skipping CRS check.")
        result["crs_match"] = True
    elif info1.crs == info2.crs:
        result["crs_match"] = True
    else:
        result["issues"].append(f"CRS mismatch: '{info1.crs}' vs '{info2.crs}'.")

    # Resolution check
    res_ratio = max(info1.resolution_x, info2.resolution_x) / (min(info1.resolution_x, info2.resolution_x) + 1e-9)
    if res_ratio <= 2.0:
        result["resolution_match"] = True
    else:
        result["issues"].append(
            f"Resolution mismatch: {info1.resolution_x:.2f}m vs {info2.resolution_x:.2f}m (ratio={res_ratio:.2f} > 2.0).")

    # Bounding-box overlap
    l1, b1, r1, t1 = info1.bounds
    l2, b2, r2, t2 = info2.bounds

    if any(math.isnan(v) for v in (l1, b1, r1, t1, l2, b2, r2, t2)):
        result["issues"].append("Bounding box unknown – cannot compute overlap.")
        result["overlap_fraction"] = 1.0
    else:
        inter_l = max(l1, l2); inter_b = max(b1, b2)
        inter_r = min(r1, r2); inter_t = min(t1, t2)
        if inter_r > inter_l and inter_t > inter_b:
            inter_area = (inter_r - inter_l) * (inter_t - inter_b)
            area1 = max((r1 - l1) * (t1 - b1), 1e-9)
            area2 = max((r2 - l2) * (t2 - b2), 1e-9)
            overlap = inter_area / min(area1, area2)
        else:
            overlap = 0.0
        result["overlap_fraction"] = float(overlap)
        if overlap < 0.8:
            result["issues"].append(f"Insufficient overlap: {overlap*100:.1f}% (< 80%).")

    # Origin alignment
    if not (math.isnan(l1) or math.isnan(l2)):
        dx = abs(l1 - l2); dy = abs(t1 - t2)
        res_m = _resolution_to_metres(min(info1.resolution_x, info2.resolution_x), info1.crs)
        offset_m = math.sqrt((dx * res_m) ** 2 + (dy * res_m) ** 2) if res_m else float("inf")
        result["origin_offset_m"] = float(offset_m)
        if offset_m > tolerance_m:
            result["issues"].append(f"Origin misalignment: ~{offset_m:.1f}m > tolerance {tolerance_m}m.")

    result["coregistered"] = len(result["issues"]) == 0
    return result


def check_bitemporal_pair(path1: str, path2: str) -> dict:
    """
    Determine whether two rasters form a valid bitemporal pair.

    Returns dict with keys:
        'is_bitemporal' (bool), 'sensor_match' (bool),
        'date1' (datetime|None), 'date2' (datetime|None),
        'days_apart' (int|None), 'overlap_fraction' (float),
        'issues' (List[str])
    """
    result: dict = {
        "is_bitemporal": False, "sensor_match": False,
        "date1": None, "date2": None, "days_apart": None,
        "overlap_fraction": 0.0, "issues": [],
    }

    try:
        info1 = load_geotiff(path1)
        info2 = load_geotiff(path2)
    except Exception as e:
        result["issues"].append(f"Failed to load files: {e}")
        return result

    result["date1"] = info1.acquisition_date
    result["date2"] = info2.acquisition_date

    if info1.acquisition_date and info2.acquisition_date:
        delta = abs((info2.acquisition_date - info1.acquisition_date).days)
        result["days_apart"] = delta
        if delta == 0:
            result["issues"].append("Both images appear to have the same acquisition date.")
    else:
        result["issues"].append("Could not extract acquisition date for one or both files.")

    s1, s2 = info1.sensor_hint, info2.sensor_hint
    compat = (s1 == s2) or (_is_optical_sensor(s1) and _is_optical_sensor(s2)) or \
             (_is_sar_sensor(s1) and _is_sar_sensor(s2))
    if compat:
        result["sensor_match"] = True
    else:
        result["issues"].append(f"Sensor type mismatch for bitemporal: '{s1}' vs '{s2}'.")

    coreg = check_coregistration(path1, path2, tolerance_m=500.0)
    result["overlap_fraction"] = coreg["overlap_fraction"]
    if coreg["overlap_fraction"] < 0.5:
        result["issues"].append(f"Low spatial overlap for bitemporal pair: {coreg['overlap_fraction']*100:.1f}%.")

    result["is_bitemporal"] = (
        result["sensor_match"]
        and result["overlap_fraction"] >= 0.5
        and (result["days_apart"] is None or result["days_apart"] > 0)
    )
    return result


def run_compatibility_check(paths: List[str]) -> CompatibilityReport:
    """
    Run a full compatibility check on one or more raster file paths.
    Returns a CompatibilityReport.
    """
    issues: List[str] = []
    warnings_: List[str] = []
    file_infos: List[GeoTIFFInfo] = []
    sensor_types: List[str] = []

    for p in paths:
        if not os.path.isfile(p):
            issues.append(f"File not found: {p}")
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            issues.append(f"Unsupported file type '{ext}': {p}")
            continue
        try:
            info = load_geotiff(p)
            file_infos.append(info)
            sensor_types.append(info.sensor_hint)
        except Exception as e:
            issues.append(f"Cannot open '{p}': {e}")

    if not file_infos:
        return CompatibilityReport(
            valid=False, input_type="unknown", sensor_types=sensor_types,
            issues=issues, warnings=warnings_, recommended_tasks=["run_vqa"], file_infos=[],
        )

    for info in file_infos:
        if info.sensor_hint == "unknown":
            warnings_.append(f"Could not determine sensor type for '{os.path.basename(info.path)}'.")
        if math.isnan(info.resolution_x):
            warnings_.append(f"No spatial resolution metadata for '{os.path.basename(info.path)}'.")

    input_type = auto_detect_input_type(paths)

    if len(file_infos) == 2:
        s0, s1 = file_infos[0].sensor_hint, file_infos[1].sensor_hint
        if (_is_optical_sensor(s0) and _is_sar_sensor(s1)) or (_is_sar_sensor(s0) and _is_optical_sensor(s1)):
            coreg = check_coregistration(paths[0], paths[1])
            if not coreg["coregistered"]:
                warnings_.extend(coreg["issues"])
        elif s0 == s1 or (_is_optical_sensor(s0) and _is_optical_sensor(s1)) or \
             (_is_sar_sensor(s0) and _is_sar_sensor(s1)):
            bt = check_bitemporal_pair(paths[0], paths[1])
            if not bt["is_bitemporal"]:
                warnings_.extend(bt["issues"])

    recommended = _build_recommended_tasks(input_type, file_infos)
    return CompatibilityReport(
        valid=len(issues) == 0, input_type=input_type, sensor_types=sensor_types,
        issues=issues, warnings=warnings_, recommended_tasks=recommended, file_infos=file_infos,
    )


def to_thumbnail(path: str, max_size: int = 512) -> Any:
    """
    Generate a thumbnail PIL Image from a raster file.
    Returns PIL.Image.Image or None if PIL is unavailable.
    """
    if not _HAS_PIL:
        logger.warning("Pillow not installed – cannot generate thumbnail.")
        return None

    try:
        if _HAS_RASTERIO and os.path.splitext(path)[1].lower() in (".tif", ".tiff", ".geotiff"):
            with rasterio.open(path) as src:
                n_bands = src.count
                scale = max_size / max(src.width, src.height)
                out_w = max(1, int(src.width * scale))
                out_h = max(1, int(src.height * scale))

                if n_bands >= 3:
                    r = src.read(min(3, n_bands), out_shape=(out_h, out_w), resampling=Resampling.bilinear)
                    g = src.read(min(2, n_bands), out_shape=(out_h, out_w), resampling=Resampling.bilinear)
                    b = src.read(1,               out_shape=(out_h, out_w), resampling=Resampling.bilinear)
                    arr = np.stack([r, g, b], axis=-1)
                else:
                    arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)

                arr = arr.astype(np.float32)
                lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
                if hi > lo:
                    arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
                return PILImage.fromarray(arr)
        else:
            img = PILImage.open(path)
            img.thumbnail((max_size, max_size), PILImage.LANCZOS)
            return img
    except Exception as e:
        logger.error("to_thumbnail failed for %s: %s", path, e)
        return None


def auto_detect_input_type(paths: List[str]) -> str:
    """
    Determine the category of input based on file paths.
    Returns one of: 'single_optical', 'single_sar', 'optical_sar_pair',
                    'bitemporal_pair', 'multi_image', 'unknown'
    """
    if not paths:
        return "unknown"

    infos: List[GeoTIFFInfo] = []
    for p in paths:
        try:
            infos.append(load_geotiff(p))
        except Exception:
            continue

    n = len(infos)
    if n == 0:
        return "unknown"
    if n == 1:
        return "single_sar" if _is_sar_sensor(infos[0].sensor_hint) else "single_optical"
    if n == 2:
        s0, s1 = infos[0].sensor_hint, infos[1].sensor_hint
        if (_is_optical_sensor(s0) and _is_sar_sensor(s1)) or \
           (_is_sar_sensor(s0) and _is_optical_sensor(s1)):
            return "optical_sar_pair"
        return "bitemporal_pair"
    return "multi_image"


def load_as_array(path: str) -> np.ndarray:
    """
    Load a raster file as a float32 NumPy array in channels-first format [C, H, W].
    Normalises raw values by dtype range.
    """
    if _HAS_RASTERIO and os.path.splitext(path)[1].lower() not in (".png", ".jpg", ".jpeg"):
        return _load_array_rasterio(path)
    if _HAS_PIL:
        return _load_array_pil(path)
    raise RuntimeError("Cannot load array: neither rasterio nor Pillow is installed.")


def _load_array_rasterio(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    dtype_str = str(data.dtype)
    if "uint8" in dtype_str:
        data /= 255.0
    elif "uint16" in dtype_str or "int16" in dtype_str:
        data /= 65535.0
    elif "uint32" in dtype_str or "int32" in dtype_str:
        data /= 4_294_967_295.0
    return data


def _load_array_pil(path: str) -> np.ndarray:
    img = PILImage.open(path).convert("RGB")
    return np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))


def _build_recommended_tasks(input_type: str, file_infos: List[GeoTIFFInfo]) -> List[str]:
    tasks: List[str] = []
    if input_type == "single_optical":
        tasks += ["run_vqa", "run_grounding"]
    elif input_type == "single_sar":
        tasks += ["run_vqa"]
    elif input_type == "optical_sar_pair":
        tasks += ["run_fusion", "run_vqa"]
    elif input_type == "bitemporal_pair":
        tasks += ["run_change", "run_vqa"]
    elif input_type == "multi_image":
        tasks += ["run_vqa", "run_change"]
    else:
        tasks += ["run_vqa"]
    if "run_vqa" not in tasks:
        tasks.append("run_vqa")
    return tasks


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("geotiff_handler -- Smoke Test")
    print("=" * 60)

    for ts in [
        "S2A_MSIL2A_20220115T052321_N0400_R005_T44QLK.SAFE",
        "RISAT1_2021-08-15_orbit0123.tif",
        "cartosat_20230301T094512.tiff",
        "no_date_here.tif",
    ]:
        dt = _parse_date_from_string(ts)
        print(f"  parse_date('{ts[:55]}') -> {dt}")

    assert _hint_from_tags({"satellite": "Sentinel-2"}, 13, 10.0) == "sentinel2"
    assert _hint_from_tags({"satellite": "RISAT-1"}, 4, 6.0) == "risat1_hybrid"
    assert _hint_from_tags({"satellite": "Cartosat"}, 1, 0.65) == "cartosat2s_pan"
    print("[PASS] _hint_from_tags()")

    assert _is_sar_sensor("risat2_xband")
    assert _is_optical_sensor("cartosat2s_hrmx")
    assert not _is_sar_sensor("sentinel2")
    print("[PASS] _is_sar_sensor / _is_optical_sensor")

    tasks = _build_recommended_tasks("optical_sar_pair", [])
    assert "run_fusion" in tasks and "run_vqa" in tasks
    tasks2 = _build_recommended_tasks("bitemporal_pair", [])
    assert "run_change" in tasks2
    print("[PASS] _build_recommended_tasks()")

    print("=" * 60)
    print("All non-IO smoke tests PASSED.")
    sys.exit(0)
