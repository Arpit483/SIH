"""
sensor_normalizer.py
====================
SatQuery AI – SIH-26167 (ISRO Remote Sensing VLM)

5-stage sensor normalization pipeline that bridges the domain gap between
training sensors (Sentinel-1/2) and ISRO evaluation sensors
(Cartosat-2S, RISAT-1, RISAT-2).

Sensor specs quick-reference
-----------------------------
Cartosat-2S PAN  : 0.65 m,  mono,          11-bit DN [0-2047]
Cartosat-2S HRMX : 2.0 m,   4-band VNIR,   11-bit DN [0-2047]  (B,G,R,NIR)
RISAT-1          : C-band (5.35 GHz), HH/HV/VV/VH + Hybrid Pol (S0,S1,S2,S3)
RISAT-2          : X-band (9.6 GHz),  HH or HH+HV
Sentinel-2       : 13 bands, 10-60 m, 12-bit, L2A BOA reflectance = (DN-1000)/10000
Sentinel-1       : C-band, VV+VH, dB;  VV [-25,0]->,[0,1], VH [-32,-5]->[0,1]

Pipeline stages
---------------
1. Channel Harmonization
2. Radiometric Calibration
3. Spatial Scale Alignment
4. Statistical Normalization
5. Final clip to [0, 1]
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Optional

import numpy as np

# optional heavy dependencies
try:
    import rasterio
    from rasterio.enums import Resampling
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False
    warnings.warn("rasterio not installed – detect_sensor_type() will be limited.", ImportWarning)

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not installed – falling back to NumPy-only implementations.", ImportWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SENTINEL1_RANGES = {
    "VV": (-25.0, 0.0),
    "VH": (-32.0, -5.0),
}

CARTOSAT_KKAL = 10.0

RISAT_KCAL: dict = {
    "risat1_cpol":   0.0,
    "risat1_hybrid": 0.0,
    "risat2_xband":  0.0,
}

SUPPORTED_SENSORS = {
    "sentinel2",
    "sentinel1",
    "cartosat2s_pan",
    "cartosat2s_hrmx",
    "risat1_cpol",
    "risat1_hybrid",
    "risat2_xband",
}

TARGET_GSD_M = 10.0


# ---------------------------------------------------------------------------
# Helper: Refined Lee Speckle Filter
# ---------------------------------------------------------------------------

def refined_lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply the Refined Lee speckle filter to a SAR amplitude/intensity image.

    Computes local mean and variance in a NxN window, then applies a
    local Wiener-style weighting: W = 1 - Cv^2 / (Cv^2 + ENL^-1)
    where ENL = window_size^2 / 2 (equivalent number of looks).

    Parameters
    ----------
    image : np.ndarray
        Input 2-D intensity image (linear scale, not dB). Also accepts 3-D
        [C, H, W] in which case each channel is filtered independently.
    window_size : int
        Size of the sliding window (must be odd). Default 5.

    Returns
    -------
    np.ndarray
        Speckle-filtered image with same shape and dtype as input.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    orig_shape = image.shape
    if image.ndim == 2:
        image = image[np.newaxis, ...]

    C, H, W = image.shape
    filtered = np.empty_like(image, dtype=np.float32)
    enl = (window_size ** 2) / 2.0

    for c in range(C):
        band = image[c].astype(np.float32)

        if _HAS_SCIPY:
            local_mean = uniform_filter(band, size=window_size, mode="reflect")
            local_sq_mean = uniform_filter(band ** 2, size=window_size, mode="reflect")
        else:
            local_mean = _numpy_uniform_filter(band, window_size)
            local_sq_mean = _numpy_uniform_filter(band ** 2, window_size)

        local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)
        eps = 1e-8
        cv_sq = local_var / (local_mean ** 2 + eps)
        weight = np.clip(1.0 - (1.0 / enl) / (cv_sq + 1.0 / enl + eps), 0.0, 1.0)
        filtered[c] = local_mean + weight * (band - local_mean)

    return filtered.reshape(orig_shape)


def _numpy_uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Pure-NumPy sliding-window mean via cumsum."""
    arr = arr.astype(np.float64)
    pad = size // 2
    arr_pad = np.pad(arr, pad, mode="reflect")
    H, W = arr.shape
    integral = arr_pad.cumsum(axis=0).cumsum(axis=1)
    out = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / (size * size)
    return out[:H, :W].astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: detect sensor type from GeoTIFF metadata
# ---------------------------------------------------------------------------

def detect_sensor_type(geotiff_path: str) -> str:
    """
    Attempt to auto-detect the sensor type by reading GeoTIFF metadata tags.

    Returns one of SUPPORTED_SENSORS or 'unknown'.
    Uses rasterio when available; falls back to band-count/resolution heuristics.
    """
    if not _HAS_RASTERIO:
        logger.warning("rasterio not available – sensor detection is limited.")
        return "unknown"

    try:
        with rasterio.open(geotiff_path) as src:
            tags = src.tags()
            profile = src.profile
            band_count = src.count
            res_x = abs(src.transform.a)
            dtype = str(profile.get("dtype", ""))

        tags_lower = {k.lower(): v.lower() for k, v in tags.items()}
        satellite = tags_lower.get("satellite", tags_lower.get("mission", ""))
        sensor    = tags_lower.get("sensor",    tags_lower.get("instrument", ""))
        mode      = tags_lower.get("mode",      tags_lower.get("beam_mode", ""))

        if "sentinel-2" in satellite or "sentinel2" in satellite or satellite == "s2":
            return "sentinel2"
        if "sentinel-1" in satellite or "sentinel1" in satellite or satellite == "s1":
            return "sentinel1"
        if "cartosat" in satellite and "pan" in (sensor + mode):
            return "cartosat2s_pan"
        if "cartosat" in satellite:
            return "cartosat2s_hrmx"
        if "risat-1" in satellite or "risat1" in satellite:
            return "risat1_hybrid" if ("hybrid" in mode or band_count == 4) else "risat1_cpol"
        if "risat-2" in satellite or "risat2" in satellite:
            return "risat2_xband"

        # Band-count/resolution heuristic fallback
        if band_count == 1 and res_x < 1.5:
            return "cartosat2s_pan"
        if band_count in (2, 4) and res_x <= 2.5:
            return "cartosat2s_hrmx"
        if band_count == 13:
            return "sentinel2"
        if band_count == 2 and dtype.startswith("float"):
            return "sentinel1"

    except Exception as exc:
        logger.error("detect_sensor_type failed for %s: %s", geotiff_path, exc)

    return "unknown"


# ---------------------------------------------------------------------------
# FDA (Fourier Domain Adaptation) helper
# ---------------------------------------------------------------------------

def _fda_style_transfer(source: np.ndarray, target: np.ndarray, beta: float = 0.05) -> np.ndarray:
    """
    Fourier Domain Adaptation (FDA) – swap low-frequency amplitude spectrum
    of `source` with that of `target` to reduce domain gap.

    Both inputs should be single-channel 2-D arrays in [0, 1].
    Reference: Yang and Soatto, "FDA: Fourier Domain Adaptation for Semantic
    Segmentation", CVPR 2020.
    """
    H, W = source.shape
    fft_src = np.fft.fft2(source)
    fft_tgt = np.fft.fft2(target) if target.shape == source.shape else \
              np.fft.fft2(_resize_array_nearest(target, H, W))

    b = int(min(H, W) * beta)
    mask = np.zeros((H, W), dtype=bool)
    mask[:b, :b]  = True
    mask[:b, -b:] = True
    mask[-b:, :b] = True
    mask[-b:, -b:] = True

    amp_src = np.abs(fft_src)
    amp_tgt = np.abs(fft_tgt)
    phase_src = np.angle(fft_src)

    amp_new = amp_src.copy()
    amp_new[mask] = amp_tgt[mask]

    fft_new = amp_new * np.exp(1j * phase_src)
    return np.real(np.fft.ifft2(fft_new)).astype(np.float32)


def _resize_array_nearest(arr: np.ndarray, new_H: int, new_W: int) -> np.ndarray:
    """Nearest-neighbour resize using index arithmetic."""
    H, W = arr.shape
    row_idx = (np.arange(new_H) * H / new_H).astype(int)
    col_idx = (np.arange(new_W) * W / new_W).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


# ---------------------------------------------------------------------------
# Main class: SensorNormalizer
# ---------------------------------------------------------------------------

class SensorNormalizer:
    """
    5-stage normalization pipeline to harmonize ISRO sensor imagery to the
    same statistical distribution as Sentinel-1/2 training data.

    Canonical output layouts
    -------------------------
    Optical : (C=4, H, W)  →  [B, G, R, NIR]
    SAR     : (C=2, H, W)  →  [co_pol, cross_pol]

    Usage
    -----
    >>> normalizer = SensorNormalizer()
    >>> out = normalizer.normalize(image_array, sensor_type='cartosat2s_hrmx',
    ...                            source_gsd=2.0)
    """

    S2_BAND_IDX = {"B": 1, "G": 2, "R": 3, "NIR": 7}  # 0-indexed within 13-band stack

    def __init__(
        self,
        apply_speckle_filter: bool = True,
        apply_histogram_matching: bool = False,
        apply_fda: bool = False,
        fda_beta: float = 0.05,
        reference_image: Optional[np.ndarray] = None,
    ):
        self.apply_speckle_filter     = apply_speckle_filter
        self.apply_histogram_matching = apply_histogram_matching
        self.apply_fda                = apply_fda
        self.fda_beta                 = fda_beta
        self.reference_image          = reference_image

        if (apply_histogram_matching or apply_fda) and reference_image is None:
            logger.warning(
                "apply_histogram_matching/apply_fda enabled but no reference_image "
                "supplied – these stages will be skipped."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        image: np.ndarray,
        sensor_type: str,
        metadata: Optional[dict] = None,
        source_gsd: Optional[float] = None,
    ) -> np.ndarray:
        """
        Run the full 5-stage normalization pipeline.

        Parameters
        ----------
        image : np.ndarray
            Raw input. Accepted layouts: (H,W), (H,W,C) channels-last,
            or (C,H,W) channels-first.
        sensor_type : str
            One of SUPPORTED_SENSORS.
        metadata : dict, optional
            Keys used (sensor-dependent):
              'incidence_angle'   – degrees (SAR)
              'center_incidence'  – degrees (SAR)
              'k_cal'             – calibration constant dB (SAR)
              'solar_zenith'      – degrees (optical DOS)
        source_gsd : float, optional
            Ground Sampling Distance in metres.

        Returns
        -------
        np.ndarray  shape (C, H', W'), dtype float32, values in [0, 1]
        """
        if sensor_type not in SUPPORTED_SENSORS:
            raise ValueError(
                f"Unknown sensor_type '{sensor_type}'. Supported: {sorted(SUPPORTED_SENSORS)}"
            )

        metadata = metadata or {}
        image = self._ensure_channels_first(image)
        logger.info("normalize() | sensor=%s | raw shape=%s | dtype=%s",
                    sensor_type, image.shape, image.dtype)

        # Stage 1: Channel Harmonization
        image = self._stage1_channel_harmonization(image, sensor_type)
        logger.debug("Stage 1 done: shape=%s", image.shape)

        # Stage 2: Radiometric Calibration
        image = self._stage2_radiometric_calibration(image, sensor_type, metadata)
        logger.debug("Stage 2 done: shape=%s", image.shape)

        # Stage 3: Spatial Scale Alignment
        gsd = source_gsd or _default_gsd(sensor_type)
        image = self._stage3_spatial_alignment(image, sensor_type, gsd)
        logger.debug("Stage 3 done: shape=%s", image.shape)

        # Stage 4: Statistical Normalization
        image = self._stage4_statistical_normalization(image, sensor_type)
        logger.debug("Stage 4 done: shape=%s", image.shape)

        # Stage 5: Final clip to [0, 1]
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        logger.info("normalize() complete | output shape=%s", image.shape)
        return image

    # ------------------------------------------------------------------
    # Stage 1 – Channel Harmonization
    # ------------------------------------------------------------------

    def _stage1_channel_harmonization(self, image: np.ndarray, sensor_type: str) -> np.ndarray:
        """
        Map sensor-specific bands to canonical layout:
          Optical -> [B, G, R, NIR]
          SAR     -> [co_pol, cross_pol]
        """
        C = image.shape[0]

        if sensor_type == "sentinel2":
            if C == 13:
                image = np.stack([
                    image[self.S2_BAND_IDX["B"]],
                    image[self.S2_BAND_IDX["G"]],
                    image[self.S2_BAND_IDX["R"]],
                    image[self.S2_BAND_IDX["NIR"]],
                ], axis=0)
            elif C == 4:
                pass
            elif C == 3:
                logger.warning("Sentinel-2 with 3 bands – synthesizing NIR from red channel.")
                image = np.concatenate([image, image[[2]]], axis=0)
            else:
                raise ValueError(f"Unexpected Sentinel-2 band count: {C}")

        elif sensor_type == "sentinel1":
            if C < 2:
                raise ValueError("Sentinel-1 requires at least 2 channels (VV, VH).")
            image = image[:2]

        elif sensor_type == "cartosat2s_pan":
            if C != 1:
                image = image[[0]]
            # Replicate to 4-channel [B, G, R, NIR]
            image = np.repeat(image, 4, axis=0)

        elif sensor_type == "cartosat2s_hrmx":
            if C == 4:
                pass
            elif C == 3:
                logger.warning("Cartosat-2S HRMX with 3 bands – synthesizing NIR.")
                image = np.concatenate([image, image[[2]]], axis=0)
            else:
                raise ValueError(f"Cartosat-2S HRMX expects 3 or 4 bands, got {C}.")

        elif sensor_type == "risat1_cpol":
            if C < 2:
                raise ValueError("RISAT-1 C-pol requires >= 2 channels.")
            image = image[:2]

        elif sensor_type == "risat1_hybrid":
            # Input: Stokes parameters [S0, S1, S2, S3]
            if C < 2:
                raise ValueError("RISAT-1 Hybrid Pol requires >= 2 Stokes channels.")
            if C >= 4:
                S0, S1, S2, S3 = image[0], image[1], image[2], image[3]
            elif C == 2:
                S0, S1 = image[0], image[1]
                S2 = S3 = np.zeros_like(S0)
            else:
                S0, S1, S2 = image[0], image[1], image[2]
                S3 = np.zeros_like(S0)

            eps = 1e-8
            # Degree of circular polarisation
            m = np.clip(S3 / (S0 + eps), -1.0, 1.0)
            # Pseudo-VV / Pseudo-VH (Mandal et al. 2019 IEEE GRSL)
            pseudo_VV = 0.5 * (S0 + S1)
            pseudo_VH = 0.5 * (S0 - S1) * (1.0 - m) / (1.0 + m + eps)
            image = np.stack([pseudo_VV, pseudo_VH], axis=0)

        elif sensor_type == "risat2_xband":
            if C >= 2:
                image = image[:2]
            else:
                # Single HH – pad cross-pol with zeros
                image = np.concatenate([image, np.zeros_like(image[[0]])], axis=0)

        return image.astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 2 – Radiometric Calibration
    # ------------------------------------------------------------------

    def _stage2_radiometric_calibration(
        self, image: np.ndarray, sensor_type: str, metadata: dict
    ) -> np.ndarray:
        if sensor_type == "sentinel2":
            return self._calib_sentinel2(image)
        elif sensor_type == "sentinel1":
            return self._calib_sentinel1(image)
        elif sensor_type in ("cartosat2s_pan", "cartosat2s_hrmx"):
            return self._calib_cartosat(image, metadata)
        elif sensor_type in ("risat1_cpol", "risat1_hybrid", "risat2_xband"):
            return self._calib_risat(image, sensor_type, metadata)
        return image

    def _calib_sentinel2(self, image: np.ndarray) -> np.ndarray:
        """L2A BOA reflectance: (DN - 1000) / 10000."""
        return (image - 1000.0) / 10000.0

    def _calib_sentinel1(self, image: np.ndarray) -> np.ndarray:
        """Normalize dB values channel-wise: VV [-25,0]->[0,1], VH [-32,-5]->[0,1]."""
        vv_min, vv_max = SENTINEL1_RANGES["VV"]
        vh_min, vh_max = SENTINEL1_RANGES["VH"]
        out = image.copy()
        out[0] = (image[0] - vv_min) / (vv_max - vv_min)
        out[1] = (image[1] - vh_min) / (vh_max - vh_min)
        return out

    def _calib_cartosat(self, image: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Cartosat-2S DN -> BOA reflectance via Dark Object Subtraction (DOS).

        Steps:
        1. Compute dark-object value (1st percentile per band) as path radiance proxy.
        2. Subtract: DN_corrected = DN - DOS_value.
        3. Normalise by 11-bit saturation value (2047).
        4. Solar-zenith correction: reflectance /= cos(theta_solar).
        """
        solar_zenith_deg = metadata.get("solar_zenith", 30.0)
        cos_theta_s = math.cos(math.radians(solar_zenith_deg))
        out = np.empty_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            band = image[c].astype(np.float32)
            dos_val = float(np.percentile(band[band > 0], 1.0)) if np.any(band > 0) else 0.0
            corrected = np.clip(band - dos_val, 0.0, None)
            reflectance = corrected / 2047.0
            reflectance = reflectance / (cos_theta_s + 1e-8)
            out[c] = reflectance
        return out

    def _calib_risat(self, image: np.ndarray, sensor_type: str, metadata: dict) -> np.ndarray:
        """
        RISAT amplitude DN -> sigma-nought (linear).

        Formula:
          sigma0_dB = 20*log10(DN) - K_cal + 10*log10(sin(i_p) / sin(i_center))
          sigma0_linear = 10^(sigma0_dB / 10)

        Then applies Refined Lee 5x5 speckle filter.
        """
        k_cal         = metadata.get("k_cal", RISAT_KCAL.get(sensor_type, 0.0))
        incidence_deg = metadata.get("incidence_angle", 35.0)
        center_deg    = metadata.get("center_incidence", 35.0)

        theta_p      = math.radians(incidence_deg)
        theta_center = math.radians(center_deg)
        inc_corr_dB  = 10.0 * math.log10(
            math.sin(theta_p) / (math.sin(theta_center) + 1e-12) + 1e-12
        )

        out = np.empty_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            band = np.clip(image[c].astype(np.float32), 1e-6, None)
            sigma_db = 20.0 * np.log10(band) - k_cal + inc_corr_dB
            out[c] = np.power(10.0, sigma_db / 10.0)

        if self.apply_speckle_filter:
            out = refined_lee_filter(out, window_size=5)

        return out

    # ------------------------------------------------------------------
    # Stage 3 – Spatial Scale Alignment
    # ------------------------------------------------------------------

    def _stage3_spatial_alignment(
        self, image: np.ndarray, sensor_type: str, source_gsd: float
    ) -> np.ndarray:
        """
        Downsample finer-than-10m imagery to ~10m using Gaussian anti-aliasing
        pre-filter followed by bilinear or nearest-neighbour decimation.
        Coarser-than-10m data is returned unchanged.
        """
        if source_gsd >= TARGET_GSD_M:
            logger.debug("Stage 3: GSD=%.2fm >= 10m – no downsampling.", source_gsd)
            return image

        scale_factor = source_gsd / TARGET_GSD_M
        C, H, W = image.shape
        new_H = max(1, int(round(H * scale_factor)))
        new_W = max(1, int(round(W * scale_factor)))

        logger.info("Stage 3: %s %.2fm -> (%d,%d) -> (%d,%d)",
                    sensor_type, source_gsd, H, W, new_H, new_W)

        sigma = 0.5 * (1.0 / scale_factor)
        out = np.empty((C, new_H, new_W), dtype=np.float32)

        for c in range(C):
            band = image[c].astype(np.float32)
            if _HAS_SCIPY:
                blurred = gaussian_filter(band, sigma=sigma)
                from scipy.ndimage import zoom
                out[c] = zoom(blurred, (new_H / H, new_W / W), order=1)
            else:
                blurred = _numpy_gaussian_blur(band, sigma)
                out[c] = _resize_array_nearest(blurred, new_H, new_W)

        return out

    # ------------------------------------------------------------------
    # Stage 4 – Statistical Normalization
    # ------------------------------------------------------------------

    def _stage4_statistical_normalization(
        self, image: np.ndarray, sensor_type: str
    ) -> np.ndarray:
        """
        Per-channel robust normalization:
          1. Clip at [P1, P99] percentiles.
          2. Rescale to [0, 1].
          3. Optional histogram matching to reference image.
          4. Optional Fourier Domain Adaptation (FDA, beta=fda_beta).
        """
        C = image.shape[0]
        out = np.empty_like(image, dtype=np.float32)

        for c in range(C):
            band = image[c].astype(np.float32)
            p1  = float(np.percentile(band, 1.0))
            p99 = float(np.percentile(band, 99.0))
            if abs(p99 - p1) < 1e-8:
                out[c] = np.zeros_like(band)
                logger.warning("Stage 4: channel %d degenerate (p1~p99=%.4f).", c, p1)
                continue
            out[c] = (np.clip(band, p1, p99) - p1) / (p99 - p1)

        if self.apply_histogram_matching and self.reference_image is not None:
            ref = self._ensure_channels_first(self.reference_image).astype(np.float32)
            for c in range(min(C, ref.shape[0])):
                out[c] = _histogram_match(out[c], ref[c])

        if self.apply_fda and self.reference_image is not None:
            ref = self._ensure_channels_first(self.reference_image).astype(np.float32)
            for c in range(min(C, ref.shape[0])):
                out[c] = _fda_style_transfer(out[c], ref[c], beta=self.fda_beta)

        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_channels_first(image: np.ndarray) -> np.ndarray:
        """Convert (H, W) or (H, W, C) -> (C, H, W)."""
        if image.ndim == 2:
            return image[np.newaxis, ...]
        if image.ndim == 3:
            if image.shape[2] <= 13 and image.shape[0] > 13:
                return np.transpose(image, (2, 0, 1))
        return image


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _default_gsd(sensor_type: str) -> float:
    """Return a sensible default GSD (metres) per sensor."""
    defaults = {
        "sentinel2":       10.0,
        "sentinel1":       10.0,
        "cartosat2s_pan":   0.65,
        "cartosat2s_hrmx":  2.0,
        "risat1_cpol":      3.0,
        "risat1_hybrid":    6.0,
        "risat2_xband":     1.0,
    }
    return defaults.get(sensor_type, 10.0)


def _histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Adjust `source` 2-D array so its histogram matches `reference`.
    Both inputs should be in [0, 1]. Returns float32.
    """
    src_flat = source.ravel()
    ref_flat = reference.ravel()
    src_hist, bins = np.histogram(src_flat, bins=256, range=(0.0, 1.0))
    ref_hist, _    = np.histogram(ref_flat, bins=256, range=(0.0, 1.0))
    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)
    src_cdf /= (src_cdf[-1] + 1e-8)
    ref_cdf /= (ref_cdf[-1] + 1e-8)
    lut = np.interp(src_cdf, ref_cdf, np.linspace(0, 1, 256))
    bin_idx = np.clip(np.searchsorted(bins[:-1], src_flat, side="right") - 1, 0, 255)
    return lut[bin_idx].reshape(source.shape).astype(np.float32)


def _numpy_gaussian_blur(arr: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Pure-NumPy separable Gaussian blur."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, arr)
    blurred = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, blurred)
    return blurred.astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("SensorNormalizer -- Smoke Test")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    normalizer = SensorNormalizer(apply_speckle_filter=True)

    # Sentinel-2 (13-band)
    s2 = rng.integers(0, 10000, size=(256, 256, 13), dtype=np.int16).astype(np.float32) + 1000
    out = normalizer.normalize(s2, "sentinel2", source_gsd=10.0)
    assert out.shape[0] == 4 and out.min() >= 0.0 and out.max() <= 1.0
    print(f"[PASS] Sentinel-2  | shape={out.shape} | range=[{out.min():.4f},{out.max():.4f}]")

    # Sentinel-1 (2-band VV+VH)
    s1 = rng.uniform(-30, 0, size=(2, 256, 256)).astype(np.float32)
    out = normalizer.normalize(s1, "sentinel1", source_gsd=10.0)
    assert out.shape[0] == 2 and out.min() >= 0.0 and out.max() <= 1.0
    print(f"[PASS] Sentinel-1  | shape={out.shape} | range=[{out.min():.4f},{out.max():.4f}]")

    # Cartosat-2S PAN (0.65m)
    pan = rng.integers(100, 2047, size=(256, 256), dtype=np.uint16).astype(np.float32)
    out = normalizer.normalize(pan, "cartosat2s_pan", source_gsd=0.65)
    assert out.shape[0] == 4 and out.shape[1] < 50 and out.min() >= 0.0 and out.max() <= 1.0
    print(f"[PASS] Cartosat PAN| shape={out.shape} | range=[{out.min():.4f},{out.max():.4f}]")

    # Cartosat-2S HRMX (4-band, 2m)
    hrmx = rng.integers(50, 2000, size=(4, 128, 128), dtype=np.uint16).astype(np.float32)
    out = normalizer.normalize(hrmx, "cartosat2s_hrmx", source_gsd=2.0)
    assert out.shape[0] == 4 and out.min() >= 0.0 and out.max() <= 1.0
    print(f"[PASS] Cartosat HRX| shape={out.shape} | range=[{out.min():.4f},{out.max():.4f}]")

    # RISAT-1 C-pol (HH+HV)
    meta = {"incidence_angle": 35.0, "center_incidence": 35.0, "k_cal": 0.0}
    r1c = rng.uniform(100, 5000, size=(2, 128, 128)).astype(np.float32)
    out = normalizer.normalize(r1c, "risat1_cpol", meta, source_gsd=3.0)
    assert out.shape[0] == 2 and out.min() >= 0.0 and out.max() <= 1.0
    print(f"[PASS] RISAT-1 Cpol| shape={out.shape} | range=[{out.min():.4f},{out.max():.4f}]")

    # RISAT-1 Hybrid Pol (4 Stokes parameters)
    r1h = rng.uniform(50, 3000, size=(4, 64, 64)).astype(np.float32)
    r1h[0] = r1h[0] + r1h[1]  # S0 >= S1
    out = normalizer.normalize(r1h, "risat1_hybrid", meta, source_gsd=6.0)
    assert out.shape[0] == 2 and out.min() >= 0.0 and out.max() <= 1.0
    print(f"[PASS] RISAT-1 Hybr| shape={out.shape} | range=[{out.min():.4f},{out.max():.4f}]")

    # RISAT-2 X-band (single HH)
    r2 = rng.uniform(200, 8000, size=(1, 64, 64)).astype(np.float32)
    out = normalizer.normalize(r2, "risat2_xband", meta, source_gsd=1.0)
    assert out.shape[0] == 2 and out.min() >= 0.0 and out.max() <= 1.0
    print(f"[PASS] RISAT-2 Xbnd| shape={out.shape} | range=[{out.min():.4f},{out.max():.4f}]")

    # Standalone refined_lee_filter
    sar_band = rng.uniform(0, 100, size=(64, 64)).astype(np.float32)
    filtered = refined_lee_filter(sar_band, window_size=5)
    assert filtered.shape == sar_band.shape
    print(f"[PASS] Lee Filter   | in={sar_band.shape} -> out={filtered.shape}")

    print("=" * 60)
    print("All smoke tests PASSED.")
    sys.exit(0)
