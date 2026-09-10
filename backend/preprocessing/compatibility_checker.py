"""
compatibility_checker.py
========================
SatQuery AI – SIH-26167 (ISRO Remote Sensing VLM)

CompatibilityChecker – orchestrates full compatibility checks on one or more
raster inputs and determines which processing tasks are applicable.

Imports from:
  - geotiff_handler   : GeoTIFFInfo, CompatibilityReport, and helper functions
  - sensor_normalizer : SensorNormalizer, detect_sensor_type, SUPPORTED_SENSORS
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

from geotiff_handler import (
    GeoTIFFInfo,
    CompatibilityReport,
    load_geotiff,
    auto_detect_input_type,
    check_coregistration,
    check_bitemporal_pair,
    _is_sar_sensor,
    _is_optical_sensor,
    SUPPORTED_EXTENSIONS,
)
from sensor_normalizer import SensorNormalizer, detect_sensor_type, SUPPORTED_SENSORS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Task identifiers
TASK_VQA          = "run_vqa"
TASK_GROUNDING    = "run_grounding"
TASK_FUSION       = "run_fusion"
TASK_CHANGE       = "run_change"
TASK_SEGMENTATION = "run_segmentation"

FUSION_COREG_TOLERANCE_M = 100.0
MIN_OVERLAP_FRACTION     = 0.5


# ---------------------------------------------------------------------------
# GeoTIFFHandler wrapper
# ---------------------------------------------------------------------------

class GeoTIFFHandler:
    """
    Thin object-oriented wrapper around free functions in geotiff_handler.py.
    Provides the interface expected by CompatibilityChecker.
    """

    def load(self, path: str) -> GeoTIFFInfo:
        return load_geotiff(path)

    def check_coreg(self, path1: str, path2: str, tolerance_m: float = 50.0) -> dict:
        return check_coregistration(path1, path2, tolerance_m=tolerance_m)

    def check_bitemporal(self, path1: str, path2: str) -> dict:
        return check_bitemporal_pair(path1, path2)

    def detect_input_type(self, paths: List[str]) -> str:
        return auto_detect_input_type(paths)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CompatibilityChecker:
    """
    Full compatibility check pipeline for SatQuery AI inputs.

    Pipeline stages
    ---------------
    1. File format validation (extension + existence + size)
    2. Load GeoTIFFInfo metadata for each file
    3. Auto-detect / refine sensor types
    4. Auto-detect input type (single/pair/bitemporal/multi)
    5. Co-registration / bitemporal checks for pairs
    6. Build recommended_tasks list for the Claude orchestrator
    7. Assemble and return CompatibilityReport

    Task routing
    ------------
    1 optical image            -> [run_vqa, run_grounding]
    1 SAR image                -> [run_vqa]
    1 optical + 1 SAR (pair)   -> [run_fusion, run_vqa]
    2 images (same modality)   -> [run_change, run_vqa]
    Any input (fallback)       -> run_vqa always included

    Usage
    -----
    >>> checker = CompatibilityChecker()
    >>> report = checker.check(["/data/cartosat_hrmx.tif"])
    >>> print(report.recommended_tasks)
    ['run_vqa', 'run_grounding']
    """

    def __init__(self) -> None:
        self.geotiff_handler   = GeoTIFFHandler()
        self.sensor_normalizer = SensorNormalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, file_paths: List[str]) -> CompatibilityReport:
        """
        Run the full compatibility check pipeline.

        Parameters
        ----------
        file_paths : List[str]
            One or more file system paths to raster images.

        Returns
        -------
        CompatibilityReport
        """
        issues:   List[str] = []
        warnings: List[str] = []
        file_infos: List[GeoTIFFInfo] = []

        # Stage 1: File format validation
        valid_paths = self._validate_file_formats(file_paths, issues, warnings)
        if not valid_paths:
            return CompatibilityReport(
                valid=False, input_type="unknown", sensor_types=[],
                issues=issues, warnings=warnings,
                recommended_tasks=[TASK_VQA], file_infos=[],
            )

        # Stage 2: Load metadata
        file_infos = self._load_metadata(valid_paths, issues, warnings)
        if not file_infos:
            return CompatibilityReport(
                valid=False, input_type="unknown", sensor_types=[],
                issues=issues, warnings=warnings,
                recommended_tasks=[TASK_VQA], file_infos=[],
            )

        # Stage 3: Refine sensor hints
        self._refine_sensor_hints(file_infos, warnings)
        sensor_types = [info.sensor_hint for info in file_infos]

        # Stage 4: Detect input type
        input_type = self._detect_input_type(file_infos)
        logger.info("Input type detected: %s", input_type)

        # Stage 5: Pair-specific compatibility checks
        if len(file_infos) == 2:
            self._check_pair_compatibility(file_infos, input_type, issues, warnings)

        # Stage 6: Build recommended task list
        recommended_tasks = self._build_recommended_tasks(input_type, file_infos, issues)
        logger.info("Recommended tasks: %s", recommended_tasks)

        # Stage 7: Assemble report
        return CompatibilityReport(
            valid=len(issues) == 0,
            input_type=input_type,
            sensor_types=sensor_types,
            issues=issues,
            warnings=warnings,
            recommended_tasks=recommended_tasks,
            file_infos=file_infos,
        )

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _validate_file_formats(
        self,
        file_paths: List[str],
        issues: List[str],
        warnings: List[str],
    ) -> List[str]:
        """
        Stage 1: Validate file existence, extension, and basic sanity.
        Returns the subset of paths that passed all checks.
        """
        valid: List[str] = []

        if not file_paths:
            issues.append("No file paths provided.")
            return valid

        if len(file_paths) > 10:
            warnings.append(
                f"{len(file_paths)} files provided – only the first 10 will be processed."
            )
            file_paths = file_paths[:10]

        for path in file_paths:
            if not isinstance(path, str):
                issues.append(f"Invalid path (not a string): {path!r}")
                continue
            if not os.path.exists(path):
                issues.append(f"File not found: {path}")
                continue
            if not os.path.isfile(path):
                issues.append(f"Path is not a regular file: {path}")
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                issues.append(
                    f"Unsupported file extension '{ext}' for '{os.path.basename(path)}'. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )
                continue
            file_size_mb = os.path.getsize(path) / (1024 ** 2)
            if file_size_mb < 0.001:
                warnings.append(
                    f"File is very small ({file_size_mb*1024:.1f} KB): "
                    f"'{os.path.basename(path)}' – may be empty or corrupt."
                )
            valid.append(path)

        logger.info("Format validation: %d/%d paths valid.", len(valid), len(file_paths))
        return valid

    def _load_metadata(
        self,
        paths: List[str],
        issues: List[str],
        warnings: List[str],
    ) -> List[GeoTIFFInfo]:
        """Stage 2: Load GeoTIFFInfo for each validated path."""
        infos: List[GeoTIFFInfo] = []
        for path in paths:
            try:
                info = self.geotiff_handler.load(path)
                infos.append(info)
                logger.debug("Metadata loaded: %s | bands=%d | CRS=%s", path, info.bands, info.crs)
            except Exception as exc:
                issues.append(f"Failed to read metadata for '{os.path.basename(path)}': {exc}")
        return infos

    def _refine_sensor_hints(
        self, file_infos: List[GeoTIFFInfo], warnings: List[str]
    ) -> None:
        """
        Stage 3: Attempt to refine sensor hints using detect_sensor_type()
        which reads full GeoTIFF metadata including GDAL tags.
        Falls back to hint already set by load_geotiff() if detection fails.
        """
        for info in file_infos:
            refined = "unknown"
            try:
                refined = detect_sensor_type(info.path)
            except Exception as exc:
                logger.debug("detect_sensor_type failed for %s: %s", info.path, exc)

            if refined != "unknown" and info.sensor_hint != refined:
                logger.info("Refined sensor hint for '%s': '%s' -> '%s'",
                            os.path.basename(info.path), info.sensor_hint, refined)
                try:
                    object.__setattr__(info, "sensor_hint", refined)
                except Exception:
                    pass

            if info.sensor_hint == "unknown":
                warnings.append(
                    f"Sensor type unknown for '{os.path.basename(info.path)}'. "
                    "Using generic optical processing."
                )

    def _detect_input_type(self, file_infos: List[GeoTIFFInfo]) -> str:
        """Stage 4: Classify input type from loaded metadata."""
        n = len(file_infos)
        if n == 0:
            return "unknown"
        if n == 1:
            return "single_sar" if _is_sar_sensor(file_infos[0].sensor_hint) else "single_optical"
        if n == 2:
            s0, s1 = file_infos[0].sensor_hint, file_infos[1].sensor_hint
            if (_is_optical_sensor(s0) and _is_sar_sensor(s1)) or \
               (_is_sar_sensor(s0) and _is_optical_sensor(s1)):
                return "optical_sar_pair"
            return "bitemporal_pair"
        return "multi_image"

    def _check_pair_compatibility(
        self,
        file_infos: List[GeoTIFFInfo],
        input_type: str,
        issues: List[str],
        warnings: List[str],
    ) -> None:
        """Stage 5: Co-registration / bitemporal checks for two-file inputs."""
        p0, p1 = file_infos[0].path, file_infos[1].path

        if input_type == "optical_sar_pair":
            coreg = self.geotiff_handler.check_coreg(p0, p1, tolerance_m=FUSION_COREG_TOLERANCE_M)
            if not coreg.get("coregistered", False):
                for issue in coreg.get("issues", []):
                    warnings.append(f"[Co-reg] {issue}")
                logger.warning("Optical+SAR pair: co-registration issues detected.")
            else:
                logger.info("Optical+SAR pair: co-registration OK.")

        elif input_type == "bitemporal_pair":
            bt = self.geotiff_handler.check_bitemporal(p0, p1)
            if not bt.get("is_bitemporal", False):
                for issue in bt.get("issues", []):
                    warnings.append(f"[Bitemporal] {issue}")
            else:
                d = bt.get("days_apart")
                logger.info("Bitemporal pair: valid. %s",
                            f"{d} day(s) apart" if d is not None else "unknown interval")

    def _build_recommended_tasks(
        self,
        input_type: str,
        file_infos: List[GeoTIFFInfo],
        issues: List[str],
    ) -> List[str]:
        """
        Stage 6: Build the ordered list of recommended Claude tool calls.

        Task routing logic
        ------------------
        single_optical   -> [run_vqa, run_grounding]  (+run_segmentation if >= 4 bands)
        single_sar       -> [run_vqa]
        optical_sar_pair -> [run_fusion, run_vqa]
        bitemporal_pair  -> [run_change, run_vqa]
        multi_image      -> [run_vqa, run_change]
        issues present   -> [run_vqa]  (safe fallback only)
        run_vqa is ALWAYS included as fallback in all cases.
        """
        if issues:
            logger.warning("Blocking issues found – limiting to run_vqa only.")
            return [TASK_VQA]

        tasks: List[str] = []

        if input_type == "single_optical":
            tasks += [TASK_VQA, TASK_GROUNDING]
            if file_infos and file_infos[0].bands >= 4:
                tasks.append(TASK_SEGMENTATION)
        elif input_type == "single_sar":
            tasks += [TASK_VQA]
        elif input_type == "optical_sar_pair":
            tasks += [TASK_FUSION, TASK_VQA]
        elif input_type == "bitemporal_pair":
            tasks += [TASK_CHANGE, TASK_VQA]
        elif input_type == "multi_image":
            tasks += [TASK_VQA, TASK_CHANGE]
        else:
            tasks += [TASK_VQA]

        # Guarantee run_vqa as fallback
        if TASK_VQA not in tasks:
            tasks.append(TASK_VQA)

        # Deduplicate preserving order
        seen: set = set()
        deduped: List[str] = []
        for t in tasks:
            if t not in seen:
                deduped.append(t)
                seen.add(t)
        return deduped


# ---------------------------------------------------------------------------
# Module-level convenience wrapper
# ---------------------------------------------------------------------------

def run_compatibility_check(paths: List[str]) -> CompatibilityReport:
    """
    Module-level convenience wrapper around CompatibilityChecker.check().

    Parameters
    ----------
    paths : List[str]
        File paths to check.

    Returns
    -------
    CompatibilityReport
    """
    return CompatibilityChecker().check(paths)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    print("=" * 60)
    print("CompatibilityChecker -- Smoke Test")
    print("=" * 60)

    checker = CompatibilityChecker()

    # Test 1: empty input
    report = checker.check([])
    assert not report.valid, "Empty input should be invalid"
    assert "No file paths provided" in report.issues[0]
    assert TASK_VQA in report.recommended_tasks
    print("[PASS] Empty input correctly flagged as invalid.")

    # Test 2: non-existent file
    report = checker.check(["/nonexistent/path/image.tif"])
    assert not report.valid
    assert any("not found" in i.lower() for i in report.issues)
    print("[PASS] Non-existent file correctly flagged.")

    # Test 3: unsupported extension
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    try:
        report = checker.check([tmp.name])
        assert not report.valid
        assert any("unsupported" in i.lower() for i in report.issues)
        print("[PASS] Unsupported file extension correctly flagged.")
    finally:
        os.unlink(tmp.name)

    # Test 4: non-string path
    issues: List[str] = []
    warn:   List[str] = []
    result = checker._validate_file_formats([123], issues, warn)
    assert len(result) == 0 and any("not a string" in i for i in issues)
    print("[PASS] Non-string path rejected.")

    # Test 5: directory path
    issues, warn = [], []
    result = checker._validate_file_formats([os.getcwd()], issues, warn)
    assert len(result) == 0 and any("not a regular file" in i for i in issues)
    print("[PASS] Directory path rejected.")

    # Test 6: _detect_input_type with mock GeoTIFFInfo objects
    from geotiff_handler import GeoTIFFInfo

    def _mock(sensor, bands=4):
        return GeoTIFFInfo(
            path="/mock.tif", bands=bands, width=256, height=256,
            crs="EPSG:32644", resolution_x=10.0, resolution_y=10.0,
            acquisition_date=None, sensor_hint=sensor,
            dtype="uint16", bounds=(0., 0., 2560., 2560.), nodata=None,
        )

    assert checker._detect_input_type([_mock("sentinel2")]) == "single_optical"
    assert checker._detect_input_type([_mock("sentinel1", bands=2)]) == "single_sar"
    assert checker._detect_input_type([_mock("sentinel2"), _mock("sentinel1", bands=2)]) == "optical_sar_pair"
    assert checker._detect_input_type([_mock("cartosat2s_hrmx"), _mock("cartosat2s_hrmx")]) == "bitemporal_pair"
    print("[PASS] _detect_input_type() all cases.")

    # Test 7: _build_recommended_tasks routing
    tasks = checker._build_recommended_tasks("single_optical", [_mock("sentinel2")], [])
    assert TASK_VQA in tasks and TASK_GROUNDING in tasks
    print("[PASS] single_optical -> run_vqa + run_grounding")

    tasks = checker._build_recommended_tasks("optical_sar_pair", [], [])
    assert TASK_FUSION in tasks and TASK_VQA in tasks
    print("[PASS] optical_sar_pair -> run_fusion + run_vqa")

    tasks = checker._build_recommended_tasks("bitemporal_pair", [], [])
    assert TASK_CHANGE in tasks and TASK_VQA in tasks
    print("[PASS] bitemporal_pair -> run_change + run_vqa")

    tasks = checker._build_recommended_tasks("optical_sar_pair", [], ["some blocking error"])
    assert tasks == [TASK_VQA]
    print("[PASS] Issues present -> only run_vqa.")

    print("=" * 60)
    print("All smoke tests PASSED.")
    sys.exit(0)
