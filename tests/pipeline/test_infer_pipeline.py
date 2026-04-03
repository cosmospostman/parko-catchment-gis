"""Pipeline tests: infer.py orchestrator failure modes.

Tests
-----
IP7  Mismatched feature_names raises a clear error, not a silent wrong prediction
IP8  Missing model file raises immediately with actionable message

Design
------
Both tests exercise cmd_run() directly (no subprocess) with monkeypatched
network calls so they run offline.  They reuse the chip-writing infrastructure
from tests/integration/test_infer_pipeline.py.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import SCL_BAND
from pipelines.infer import (
    INFER_BANDS,
    _build_pixel_grid,
    cmd_run,
)

# ---------------------------------------------------------------------------
# Grid / STAC parameters (small grid so tests run fast)
# ---------------------------------------------------------------------------

_MINLON, _MINLAT, _MAXLON, _MAXLAT = 141.0, -18.003, 141.002, -18.001
_RESOLUTION = 0.001   # ~2×2 pixel grid
_STAC_START  = "2022-07-01"
_STAC_END    = "2022-10-31"

_YEARS = [2021, 2022, 2023]
_DOYS  = [210, 250, 290]

# Nominal training feature names (from feature_names_fixture.json)
_CORRECT_FEATURE_NAMES = [
    "peak_value", "peak_doy", "spike_duration",
    "peak_doy_mean", "peak_doy_sd", "years_detected",
    "HAND", "dist_to_water", "mean_quality",
]

# Mismatched names: includes a name that assemble_infer_feature_stack cannot populate
_MISMATCHED_FEATURE_NAMES = [
    "peak_value", "peak_doy", "spike_duration",
    "peak_doy_mean", "peak_doy_sd", "years_detected",
    "HAND", "dist_to_water", "mean_quality",
    "wrong_feature_from_different_run",   # <-- unknown name
]

_PRESENCE_BANDS: dict[str, float] = {
    "B05": 0.10, "B07": 0.40, "B08": 0.50, "B11": 0.08,
}


# ---------------------------------------------------------------------------
# Chip-writing helpers
# ---------------------------------------------------------------------------

def _write_chip_1x1(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(0, 0, 1, 1, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=1, width=1, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.array([[value]], dtype=np.float32), 1)


def _make_item(item_id: str, year: int, doy: int) -> SimpleNamespace:
    import datetime as _dt
    dt = _dt.datetime(year, 1, 1) + _dt.timedelta(days=doy - 1)
    return SimpleNamespace(
        id=item_id,
        datetime=dt,
        properties={"s2:mgrs_tile": "55HBU"},
    )


def _write_chips_for_pixel(
    inputs_dir: Path,
    pixel_id: str,
    band_values: dict[str, float],
) -> list[SimpleNamespace]:
    """Write chips for all (year, doy) combos and return item list."""
    items = []
    for year in _YEARS:
        for doy in _DOYS:
            item_id = f"S2_{pixel_id}_{year}_doy{doy}"
            item = _make_item(item_id, year, doy)
            items.append(item)
            _write_chip_1x1(
                inputs_dir / item_id / f"{SCL_BAND}_{pixel_id}.tif", 4.0,
            )
            for band, val in band_values.items():
                _write_chip_1x1(
                    inputs_dir / item_id / f"{band}_{pixel_id}.tif", val,
                )
    return items


# ---------------------------------------------------------------------------
# Shared environment setup
# ---------------------------------------------------------------------------

def _make_infer_env(
    tmp_path: Path,
    monkeypatch,
    feature_names: list[str],
    model_run_id: str = "test_run_ip",
):
    """Build a minimal infer environment with configurable feature_names.

    Returns SimpleNamespace(artefact_dir, output_dir, inputs_dir,
    model_run_id, unique_items, pixels).
    """
    artefact_dir = tmp_path / "artefacts"
    output_dir   = tmp_path / "outputs"
    inputs_dir   = tmp_path / "inputs"

    for d in (artefact_dir, output_dir, inputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Build pixel grid
    pixels, nrows, ncols, lons, lats = _build_pixel_grid(
        _MINLON, _MINLAT, _MAXLON, _MAXLAT, _RESOLUTION,
    )

    # Write chips for each pixel
    all_items = []
    for pid, lon, lat in pixels:
        items_for_pixel = _write_chips_for_pixel(inputs_dir, pid, _PRESENCE_BANDS)
        all_items.extend(items_for_pixel)

    seen: set[str] = set()
    unique_items = []
    for it in all_items:
        if it.id not in seen:
            seen.add(it.id)
            unique_items.append(it)

    # Training artefacts
    stats_path = artefact_dir / f"archive_stats_{model_run_id}.json"
    stats_path.write_text(json.dumps({"mean": 0.35, "std": 0.10}))

    fnames_path = artefact_dir / f"feature_names_{model_run_id}.json"
    fnames_path.write_text(json.dumps(feature_names))

    # Train a minimal RF using the CORRECT feature names so the model is
    # compatible regardless of what feature_names_*.json says.
    from analysis.timeseries.infer_features import assemble_infer_feature_stack
    from analysis.primitives.quality import ArchiveStats

    shape = (1, 1)

    def _fv(band_vals: dict[str, float]) -> np.ndarray:
        composite = {b: np.full(shape, v, dtype=np.float32) for b, v in band_vals.items()}
        return assemble_infer_feature_stack(
            composite_bands=composite,
            hand_raster=np.zeros(shape),
            dist_to_water_raster=np.zeros(shape),
            quality_weights=[0.9, 0.8, 0.75],
            feature_names=_CORRECT_FEATURE_NAMES,
        )[0]

    X_train = [_fv(_PRESENCE_BANDS)] * 5 + [_fv({
        "B05": 0.30, "B07": 0.20, "B08": 0.15, "B11": 0.35,
    })] * 5
    y_train = [1] * 5 + [0] * 5
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)

    model_path = artefact_dir / f"model_{model_run_id}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(rf, fh)

    # Monkeypatch network calls
    import pipelines.infer as _infer_module

    monkeypatch.setattr(_infer_module, "search_sentinel2", lambda **kwargs: unique_items)
    monkeypatch.setattr(_infer_module, "fetch_chips", AsyncMock(return_value=None))
    monkeypatch.setattr(_infer_module, "INPUTS_DIR", inputs_dir)

    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("CODE_DIR", str(tmp_path))
    monkeypatch.setenv("YEAR", "2022")

    return SimpleNamespace(
        artefact_dir=artefact_dir,
        output_dir=output_dir,
        inputs_dir=inputs_dir,
        model_run_id=model_run_id,
        pixels=pixels,
        unique_items=unique_items,
        feature_names=feature_names,
    )


def _make_run_args(env: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_run_id=env.model_run_id,
        run_id=env.model_run_id,
        artefact_dir=str(env.artefact_dir),
        output_dir=str(env.output_dir),
        bbox_minlon=_MINLON,
        bbox_minlat=_MINLAT,
        bbox_maxlon=_MAXLON,
        bbox_maxlat=_MAXLAT,
        resolution=_RESOLUTION,
        stac_start=_STAC_START,
        stac_end=_STAC_END,
        cloud_max=30,
        workers=1,
    )


# ---------------------------------------------------------------------------
# Needs AsyncMock import
# ---------------------------------------------------------------------------

try:
    from unittest.mock import AsyncMock
except ImportError:
    # Python < 3.8 fallback (should not be needed)
    AsyncMock = None  # type: ignore


# ---------------------------------------------------------------------------
# IP7: Mismatched feature_names raises a clear error
# ---------------------------------------------------------------------------

def test_IP7_mismatched_feature_names_raises_clear_error(tmp_path, monkeypatch):
    """IP7: Mismatched feature_names (unknown name) causes clear failure, not silent wrong prediction.

    The feature_names JSON contains a name that assemble_infer_feature_stack
    cannot populate.  This should surface as a SystemExit(1) with a descriptive
    log message — not a silent wrong prediction.

    Rationale: if the model and feature_names come from different runs (wrong
    run ID), the pipeline must fail loudly rather than produce garbage outputs.
    """
    env = _make_infer_env(tmp_path, monkeypatch, feature_names=_MISMATCHED_FEATURE_NAMES)
    args = _make_run_args(env)

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(args)

    assert exc_info.value.code == 1, (
        f"Expected sys.exit(1) for mismatched feature_names, "
        f"got exit code {exc_info.value.code}"
    )


def test_IP7b_correct_feature_names_does_not_exit(tmp_path, monkeypatch):
    """IP7b: Correct feature_names allows the pipeline to complete (sanity check).

    Ensures the test environment is valid: with the correct names the pipeline
    should not exit with code 1 due to feature mismatch.
    """
    env = _make_infer_env(tmp_path, monkeypatch, feature_names=_CORRECT_FEATURE_NAMES)
    args = _make_run_args(env)

    # Should complete without SystemExit (probability and confidence rasters written)
    cmd_run(args)

    prob_path = env.output_dir / f"probability_{env.model_run_id}.tif"
    assert prob_path.exists(), (
        f"probability raster not found at {prob_path} — pipeline did not complete"
    )


# ---------------------------------------------------------------------------
# IP8: Missing model file raises immediately with actionable message
# ---------------------------------------------------------------------------

def test_IP8_missing_model_file_raises_immediately(tmp_path, monkeypatch):
    """IP8: Pointing infer.py at a non-existent model run ID fails immediately.

    The pipeline should detect the missing artefact at startup (before any
    expensive STAC search or chip fetch) and exit with code 1.
    """
    import pipelines.infer as _infer_module

    artefact_dir = tmp_path / "artefacts"
    output_dir   = tmp_path / "outputs"
    artefact_dir.mkdir()
    output_dir.mkdir()

    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("CODE_DIR", str(tmp_path))
    monkeypatch.setenv("YEAR", "2022")

    # Patch network so we'd know if it was called (it shouldn't be)
    fetch_calls: list[str] = []

    async def _spy_fetch(**kwargs):
        fetch_calls.append("called")

    monkeypatch.setattr(_infer_module, "search_sentinel2", lambda **kwargs: (_ for _ in ()).throw(AssertionError("search_sentinel2 should not be called")))
    monkeypatch.setattr(_infer_module, "fetch_chips", _spy_fetch)

    args = SimpleNamespace(
        model_run_id="nonexistent_run_id_xyz",
        run_id="nonexistent_run_id_xyz",
        artefact_dir=str(artefact_dir),
        output_dir=str(output_dir),
        bbox_minlon=_MINLON,
        bbox_minlat=_MINLAT,
        bbox_maxlon=_MAXLON,
        bbox_maxlat=_MAXLAT,
        resolution=_RESOLUTION,
        stac_start=_STAC_START,
        stac_end=_STAC_END,
        cloud_max=30,
        workers=1,
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(args)

    assert exc_info.value.code == 1, (
        f"Expected sys.exit(1) for missing model file, "
        f"got exit code {exc_info.value.code}"
    )

    # Network fetch should NOT have been called — pipeline should exit before Stage 0
    assert len(fetch_calls) == 0, (
        "fetch_chips was called even though model artefacts are missing. "
        "The pipeline should fail immediately at artefact validation."
    )


def test_IP8b_missing_feature_names_file_raises_immediately(tmp_path, monkeypatch):
    """IP8b: Missing feature_names artefact also causes immediate failure."""
    import pipelines.infer as _infer_module

    artefact_dir = tmp_path / "artefacts"
    output_dir   = tmp_path / "outputs"
    artefact_dir.mkdir()
    output_dir.mkdir()

    model_run_id = "partial_run"

    # Write model and archive_stats but NOT feature_names
    stats_path = artefact_dir / f"archive_stats_{model_run_id}.json"
    stats_path.write_text(json.dumps({"mean": 0.35, "std": 0.10}))

    # Write a minimal dummy model
    rf = RandomForestClassifier(n_estimators=2, random_state=42)
    rf.fit([[0.1, 0.2]] * 5, [0, 1, 0, 1, 0])
    model_path = artefact_dir / f"model_{model_run_id}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(rf, fh)

    # feature_names_<run_id>.json intentionally NOT written

    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("CODE_DIR", str(tmp_path))
    monkeypatch.setenv("YEAR", "2022")

    fetch_calls: list[str] = []

    async def _spy_fetch(**kwargs):
        fetch_calls.append("called")

    monkeypatch.setattr(_infer_module, "fetch_chips", _spy_fetch)

    args = SimpleNamespace(
        model_run_id=model_run_id,
        run_id=model_run_id,
        artefact_dir=str(artefact_dir),
        output_dir=str(output_dir),
        bbox_minlon=_MINLON,
        bbox_minlat=_MINLAT,
        bbox_maxlon=_MAXLON,
        bbox_maxlat=_MAXLAT,
        resolution=_RESOLUTION,
        stac_start=_STAC_START,
        stac_end=_STAC_END,
        cloud_max=30,
        workers=1,
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(args)

    assert exc_info.value.code == 1, (
        f"Expected sys.exit(1) for missing feature_names file, "
        f"got exit code {exc_info.value.code}"
    )

    assert len(fetch_calls) == 0, (
        "fetch_chips was called even though feature_names artefact is missing"
    )
