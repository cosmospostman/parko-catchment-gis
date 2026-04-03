"""Integration test: infer.py run — smoke test against synthetic fixture data.

Session 14 exit condition: infer.py run completes; probability raster and
confidence raster are written; raster is not degenerate (value distribution
is not flat).

Design
------
Rather than hitting the network, this test directly exercises the pipeline
internals using synthetic chips written to a tmp_path DiskChipStore, a
synthetic trained model, and synthetic training artefacts.

The test:
1. Writes a tiny 2×3 pixel grid worth of synthetic chips (presence-like
   spectral signature at some pixels, absence-like at others).
2. Writes a minimal set of training artefacts (archive_stats, feature_names,
   a trained RF) to a tmp artefact directory.
3. Calls cmd_run() directly (bypassing the STAC/fetch layers by monkeypatching
   search_sentinel2 and fetch_chips to no-ops, and pre-writing the chips
   manually so the DiskChipStore finds them).
4. Asserts:
   IP1. probability_{run_id}.tif exists
   IP2. confidence_{run_id}.tif exists
   IP3. Both rasters are non-degenerate (std > 0) — value distribution is not flat
   IP4. Probability values are in [0, 1]
   IP5. Confidence values are in [0, 1]
   IP6. Raster shape matches the grid dimensions
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

# ---------------------------------------------------------------------------
# Imports from pipeline
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import Q_FULL, SCL_BAND
from pipelines.infer import (
    INFER_BANDS,
    _build_pixel_grid,
    cmd_run,
)

# ---------------------------------------------------------------------------
# Synthetic band values (same signatures as feature pipeline tests)
# ---------------------------------------------------------------------------

_PRESENCE_BANDS: dict[str, float] = {
    "B05": 0.10, "B07": 0.40, "B08": 0.50, "B11": 0.08,
}

_ABSENCE_BANDS: dict[str, float] = {
    "B05": 0.30, "B07": 0.20, "B08": 0.15, "B11": 0.35,
}

# Grid parameters for test
_MINLON, _MINLAT, _MAXLON, _MAXLAT = 141.0, -18.003, 141.002, -18.001
# Use coarse resolution so we get a small deterministic grid
_RESOLUTION = 0.001  # ~100m — produces ~2×2 pixels

# STAC search window
_STAC_START = "2022-07-01"
_STAC_END = "2022-10-31"

# Years/DOYs for synthetic chips
_YEARS = [2021, 2022, 2023]
_DOYS = [210, 250, 290]


# ---------------------------------------------------------------------------
# Chip helpers
# ---------------------------------------------------------------------------

def _write_chip_1x1(path: Path, value: float) -> None:
    """Write a 1×1 float32 GeoTIFF chip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(0, 0, 1, 1, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=1, width=1,
        count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.array([[value]], dtype=np.float32), 1)


def _make_item(item_id: str, year: int, doy: int) -> SimpleNamespace:
    """Duck-typed pystac.Item."""
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
            # SCL chip: class 4 = vegetation (clear)
            _write_chip_1x1(
                inputs_dir / item_id / f"{SCL_BAND}_{pixel_id}.tif",
                4.0,
            )
            for band, val in band_values.items():
                _write_chip_1x1(
                    inputs_dir / item_id / f"{band}_{pixel_id}.tif",
                    val,
                )
    return items


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def infer_env(tmp_path, monkeypatch):
    """Set up everything needed to run cmd_run() without network access.

    Returns a namespace with:
        artefact_dir, output_dir, inputs_dir, model_run_id,
        items (the synthetic item list), pixels (the grid point list)
    """
    artefact_dir = tmp_path / "artefacts"
    output_dir = tmp_path / "outputs"
    inputs_dir = tmp_path / "inputs"

    for d in (artefact_dir, output_dir, inputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Build pixel grid
    pixels, nrows, ncols, lons, lats = _build_pixel_grid(
        _MINLON, _MINLAT, _MAXLON, _MAXLAT, _RESOLUTION,
    )

    # Write synthetic chips for each pixel
    all_items = []
    for i, (pid, lon, lat) in enumerate(pixels):
        # Alternate presence / absence
        band_vals = _PRESENCE_BANDS if i % 2 == 0 else _ABSENCE_BANDS
        items_for_pixel = _write_chips_for_pixel(inputs_dir, pid, band_vals)
        all_items.extend(items_for_pixel)

    # Deduplicate items by id (same item_id appears for multiple pixels)
    seen = set()
    unique_items = []
    for it in all_items:
        if it.id not in seen:
            seen.add(it.id)
            unique_items.append(it)

    # Write archive_stats artefact
    model_run_id = "test_run_001"
    stats_path = artefact_dir / f"archive_stats_{model_run_id}.json"
    stats_path.write_text(json.dumps({"mean": 0.35, "std": 0.10}))

    # Write feature_names artefact (must match fixture)
    fnames_fixture = PROJECT_ROOT / "tests" / "fixtures" / "feature_names_fixture.json"
    feature_names = json.loads(fnames_fixture.read_text())
    fnames_path = artefact_dir / f"feature_names_{model_run_id}.json"
    fnames_path.write_text(json.dumps(feature_names))

    # Train a minimal RF on synthetic data and write model artefact
    from sklearn.ensemble import RandomForestClassifier
    from analysis.timeseries.infer_features import assemble_infer_feature_stack
    from analysis.primitives.quality import ArchiveStats

    archive_stats = ArchiveStats(mean=0.35, std=0.10)

    def _make_features(band_vals: dict[str, float]) -> np.ndarray:
        shape = (1, 1)
        composite = {b: np.full(shape, v, dtype=np.float32) for b, v in band_vals.items()}
        hand = np.zeros(shape, dtype=np.float64)
        dtw = np.zeros(shape, dtype=np.float64)
        return assemble_infer_feature_stack(
            composite_bands=composite,
            hand_raster=hand,
            dist_to_water_raster=dtw,
            quality_weights=[0.9, 0.8, 0.75],
            feature_names=feature_names,
        )[0]

    X_train = [_make_features(_PRESENCE_BANDS)] * 5 + [_make_features(_ABSENCE_BANDS)] * 5
    y_train = [1] * 5 + [0] * 5
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)

    model_path = artefact_dir / f"model_{model_run_id}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(rf, fh)

    # Monkeypatch search_sentinel2 and fetch_chips so cmd_run doesn't hit network
    import pipelines.infer as _infer_module

    def _fake_search(**kwargs):
        return unique_items

    async def _fake_fetch(**kwargs):
        pass  # chips already written

    monkeypatch.setattr(_infer_module, "search_sentinel2", _fake_search)
    monkeypatch.setattr(_infer_module, "fetch_chips", _fake_fetch)

    # Monkeypatch INPUTS_DIR in infer module to point to tmp inputs
    monkeypatch.setattr(_infer_module, "INPUTS_DIR", inputs_dir)

    # Monkeypatch config so cmd_run can import it
    import os
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("CODE_DIR", str(tmp_path))
    monkeypatch.setenv("YEAR", "2022")

    return SimpleNamespace(
        artefact_dir=artefact_dir,
        output_dir=output_dir,
        inputs_dir=inputs_dir,
        model_run_id=model_run_id,
        pixels=pixels,
        nrows=nrows,
        ncols=ncols,
        unique_items=unique_items,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_IP1_probability_raster_written(infer_env, tmp_path):
    """IP1: probability_{run_id}.tif is written by cmd_run."""
    env = infer_env
    args = SimpleNamespace(
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
    cmd_run(args)

    prob_path = env.output_dir / f"probability_{env.model_run_id}.tif"
    assert prob_path.exists(), f"Probability raster not found: {prob_path}"


def test_IP2_confidence_raster_written(infer_env, tmp_path):
    """IP2: confidence_{run_id}.tif is written by cmd_run."""
    env = infer_env
    args = SimpleNamespace(
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
    cmd_run(args)

    conf_path = env.output_dir / f"confidence_{env.model_run_id}.tif"
    assert conf_path.exists(), f"Confidence raster not found: {conf_path}"


def _run_and_read(infer_env) -> tuple[np.ndarray, np.ndarray]:
    """Helper: run cmd_run and return (prob_array, conf_array), nodata masked to nan."""
    env = infer_env
    args = SimpleNamespace(
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
    cmd_run(args)

    def _read(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nd = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
        return arr

    prob = _read(env.output_dir / f"probability_{env.model_run_id}.tif")
    conf = _read(env.output_dir / f"confidence_{env.model_run_id}.tif")
    return prob, conf


def test_IP3_rasters_not_degenerate(infer_env):
    """IP3: probability raster is non-degenerate (std > 0) across valid pixels."""
    prob, conf = _run_and_read(infer_env)

    valid_probs = prob[~np.isnan(prob)]
    assert len(valid_probs) > 0, "No valid (non-NaN) pixels in probability raster"

    prob_std = float(valid_probs.std())
    assert prob_std > 0.0, (
        f"Probability raster is flat (std={prob_std:.6f}). "
        "This indicates degenerate predictions — check feature assembly and model."
    )


def test_IP4_probability_values_in_range(infer_env):
    """IP4: probability values are in [0, 1]."""
    prob, _ = _run_and_read(infer_env)
    valid = prob[~np.isnan(prob)]
    assert len(valid) > 0
    assert float(valid.min()) >= 0.0, f"Probability below 0: {valid.min()}"
    assert float(valid.max()) <= 1.0, f"Probability above 1: {valid.max()}"


def test_IP5_confidence_values_in_range(infer_env):
    """IP5: confidence values are in [0, 1]."""
    _, conf = _run_and_read(infer_env)
    valid = conf[~np.isnan(conf)]
    assert len(valid) > 0
    assert float(valid.min()) >= 0.0, f"Confidence below 0: {valid.min()}"
    assert float(valid.max()) <= 1.0, f"Confidence above 1: {valid.max()}"


def test_IP6_raster_shape_matches_grid(infer_env):
    """IP6: raster shape matches the expected grid dimensions."""
    env = infer_env
    _, nrows, ncols, _, _ = _build_pixel_grid(_MINLON, _MINLAT, _MAXLON, _MAXLAT, _RESOLUTION)

    args = SimpleNamespace(
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
    cmd_run(args)

    with rasterio.open(env.output_dir / f"probability_{env.model_run_id}.tif") as src:
        assert src.height == nrows, f"Expected {nrows} rows, got {src.height}"
        assert src.width == ncols, f"Expected {ncols} cols, got {src.width}"
