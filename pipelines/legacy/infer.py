"""pipelines/infer.py — Inference pipeline orchestrator.

Subcommands
-----------
load-testdata
    Fetch and stage a small set of fixture chips from the STAC archive for use
    by pytest. Writes a .infer_fixture_commit sentinel to tests/fixtures/ so
    conftest can detect stale test data.

    Prerequisites: train.py load-testdata must have already run (shares the
    same inputs/ directory).

run
    Full inference pipeline: read training artefacts → Stage 0 fetch for a
    regular pixel grid → quality-weighted composite → feature stack assembly
    → RF.predict_proba() → write probability raster and confidence raster.

    Required:
        --model-run-id ID   Run ID of the trained model (e.g. 20240101T120000Z).
                            Reads model_{run_id}.pkl, feature_names_{run_id}.json,
                            archive_stats_{run_id}.json from --artefact-dir.
        --bbox MINLON MINLAT MAXLON MAXLAT
                            Bounding box (EPSG:4326) for inference region.

    Optional:
        --artefact-dir DIR  Directory containing training artefacts. Default: outputs/
        --output-dir DIR    Directory for output rasters. Default: outputs/
        --resolution DEG    Grid resolution in decimal degrees. Default: 0.0001 (~10m)
        --stac-start DATE   STAC search start date (YYYY-MM-DD). Default: 2022-07-01
        --stac-end DATE     STAC search end date (YYYY-MM-DD). Default: 2022-10-31
        --cloud-max PCT     Max cloud cover % for STAC filter. Default: 30
        --run-id ID         Identifier for output raster filenames.
                            Defaults to model-run-id.
        --workers N         ProcessPool workers. Default: auto (_pool_size)

drop-checkpoint
    Delete all output rasters for a given run ID.
    Requires --run-id and --yes to guard against accidental deletion.

Output rasters (written to --output-dir)
-----------------------------------------
probability_{run_id}.tif    float32, values [0, 1], EPSG:4326
confidence_{run_id}.tif     float32, values [0, 1], EPSG:4326 — max class probability

Both rasters share the same CRS, transform, and extent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SCL_BAND
from analysis.primitives.quality import ArchiveStats, score_observation
from analysis.timeseries.composite import quality_weighted_composite
from analysis.timeseries.extraction import extract_observations
from analysis.timeseries.infer_features import assemble_infer_feature_stack
from pipelines.legacy.chip_store import DiskChipStore
from pipelines.legacy.fetch import fetch_chips
from utils.pipeline import _pool_size
from utils.stac import search_sentinel2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level worker state (populated before fork, inherited by children)
# ---------------------------------------------------------------------------

_WORKER_STATE: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Fixture parameters — load-testdata only
# ---------------------------------------------------------------------------

# Reuse the same fixture region as train.py load-testdata; chips are shared.
FIXTURE_POINTS: list[tuple[str, float, float]] = [
    ("infer_px_001", 141.24363, -18.35002),
    ("infer_px_002", 142.86670, -18.20000),
    ("infer_px_003", 141.75000, -17.16670),
]

FIXTURE_BANDS: list[str] = ["B05", "B07", "B08", "B11", SCL_BAND]
FIXTURE_BBOX: list[float] = [141.0, -19.0, 143.5, -17.0]
FIXTURE_START: str = "2022-07-01"
FIXTURE_END: str = "2022-10-31"
FIXTURE_CLOUD_MAX: int = 30

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
INFER_SENTINEL_FILE = FIXTURE_DIR / ".infer_fixture_commit"
INPUTS_DIR = PROJECT_ROOT / "inputs"

# Bands required for inference (flowering_index needs B05, B07, B08, B11)
INFER_BANDS: list[str] = ["B05", "B07", "B08", "B11"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _run_id_from_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_pixel_grid(
    minlon: float,
    minlat: float,
    maxlon: float,
    maxlat: float,
    resolution: float,
) -> tuple[list[tuple[str, float, float]], int, int, np.ndarray, np.ndarray]:
    """Build a regular grid of pixel centre points.

    Returns
    -------
    points : list of (pixel_id, lon, lat)
    nrows  : number of rows
    ncols  : number of columns
    lons   : 1-D array of column centre longitudes
    lats   : 1-D array of row centre latitudes (descending — north at top)
    """
    lons = np.arange(minlon + resolution / 2, maxlon, resolution)
    lats = np.arange(maxlat - resolution / 2, minlat, -resolution)  # north→south
    ncols = len(lons)
    nrows = len(lats)

    points: list[tuple[str, float, float]] = []
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            pid = f"px_{r:04d}_{c:04d}"
            points.append((pid, float(lon), float(lat)))

    return points, nrows, ncols, lons, lats


def _write_raster(
    path: Path,
    data: np.ndarray,
    transform,
    crs: str = "EPSG:4326",
    nodata: float = -9999.0,
) -> None:
    """Write a 2-D float32 array as a single-band GeoTIFF."""
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = data.shape
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        arr = data.astype(np.float32)
        arr = np.where(np.isnan(arr), nodata, arr)
        dst.write(arr, 1)

    logger.info("Written raster: %s  shape=(%d, %d)", path, nrows, ncols)


# ---------------------------------------------------------------------------
# Worker: per-pixel composite + feature extraction
# ---------------------------------------------------------------------------

def _worker_init_infer(archive_stats_dict: dict, feature_names: list[str]) -> None:
    """Pool initializer: deserialise ArchiveStats and feature_names."""
    _WORKER_STATE["archive_stats"] = ArchiveStats(
        mean=archive_stats_dict["mean"],
        std=archive_stats_dict["std"],
    )
    _WORKER_STATE["feature_names"] = feature_names


def _extract_pixel_features(
    pixel: tuple[str, float, float],
    items: list,
    inputs_dir: Path,
    composite_doy: float,
) -> tuple[str, np.ndarray | None]:
    """Worker: build composite → feature vector for one pixel.

    Returns (pixel_id, feature_vector_1d) or (pixel_id, None) on failure.
    """
    pixel_id, lon, lat = pixel
    archive_stats: ArchiveStats = _WORKER_STATE["archive_stats"]
    feature_names: list[str] = _WORKER_STATE["feature_names"]

    store = DiskChipStore(inputs_dir=inputs_dir)
    points_arg = [(pixel_id, lon, lat)]

    # Extract observations for this pixel.
    # window_px=1 chips have shape (1,1), so center_px=0 (not the default 2).
    raw_obs = extract_observations(items, points_arg, store, bands=INFER_BANDS, center_px=0)
    if not raw_obs:
        return (pixel_id, None)

    # Quality score each observation
    scored_obs = [score_observation(obs, archive_stats) for obs in raw_obs]

    # Build quality weights (Q_FULL score for each observation)
    from analysis.constants import Q_FULL
    quality_weights = [obs.quality.score(Q_FULL) for obs in scored_obs]

    # Build band stacks: {band: [chip_for_obs_0, chip_for_obs_1, ...]}
    band_stacks: dict[str, list[np.ndarray]] = {band: [] for band in INFER_BANDS}
    for obs in scored_obs:
        for band in INFER_BANDS:
            val = obs.bands.get(band, 0.0)
            # Each "chip" is a 1×1 pixel since window_px=1
            band_stacks[band].append(np.array([[val]], dtype=np.float64))

    # Quality-weighted composite (result: {band: 1×1 array})
    composite = quality_weighted_composite(band_stacks, quality_weights)

    # Structural features: HAND and dist_to_water unavailable at inference time
    # without GIS rasters — use 0.0 sentinel (same as train.py placeholder)
    shape_1x1 = (1, 1)
    hand_raster = np.zeros(shape_1x1, dtype=np.float64)
    dtw_raster = np.zeros(shape_1x1, dtype=np.float64)

    # Assemble feature stack (1 pixel → 1-row array)
    composite_2d = {band: arr for band, arr in composite.items()}
    try:
        stack = assemble_infer_feature_stack(
            composite_bands=composite_2d,
            hand_raster=hand_raster,
            dist_to_water_raster=dtw_raster,
            quality_weights=quality_weights,
            feature_names=feature_names,
            composite_doy=composite_doy,
        )
    except ValueError as exc:
        logger.debug("Feature assembly failed for pixel %s: %s", pixel_id, exc)
        return (pixel_id, None)

    # stack shape: (1, n_features)
    return (pixel_id, stack[0])  # 1-D array


# ---------------------------------------------------------------------------
# Subcommand: load-testdata
# ---------------------------------------------------------------------------

def cmd_load_testdata(args: argparse.Namespace) -> None:
    """Fetch inference fixture chips and write the staleness sentinel."""
    import config as _config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    logger.info("load-testdata: searching STAC for inference fixture items")
    items = search_sentinel2(
        bbox=FIXTURE_BBOX,
        start=FIXTURE_START,
        end=FIXTURE_END,
        cloud_cover_max=FIXTURE_CLOUD_MAX,
        endpoint=_config.STAC_ENDPOINT_ELEMENT84,
        collection=_config.S2_COLLECTION,
    )
    if not items:
        logger.error("No STAC items found for fixture search parameters.")
        sys.exit(1)

    logger.info(
        "load-testdata: fetching chips for %d items × %d pixels × %d bands",
        len(items), len(FIXTURE_POINTS), len(FIXTURE_BANDS),
    )

    asyncio.run(fetch_chips(
        points=FIXTURE_POINTS,
        items=items,
        bands=FIXTURE_BANDS,
        window_px=1,
        inputs_dir=INPUTS_DIR,
        scl_filter=True,
        max_concurrent=32,
    ))

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    commit = _current_git_commit()
    INFER_SENTINEL_FILE.write_text(commit + "\n")
    logger.info(
        "load-testdata complete — sentinel written: %s (commit %s)",
        INFER_SENTINEL_FILE, commit[:12],
    )


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Full inference pipeline."""
    import config as _config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    model_run_id: str = args.model_run_id
    run_id: str = args.run_id or model_run_id
    artefact_dir = Path(args.artefact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = INPUTS_DIR

    logger.info(
        "run: model_run_id=%s  run_id=%s  output_dir=%s",
        model_run_id, run_id, output_dir,
    )

    # ------------------------------------------------------------------
    # Load training artefacts
    # ------------------------------------------------------------------
    model_path = artefact_dir / f"model_{model_run_id}.pkl"
    fnames_path = artefact_dir / f"feature_names_{model_run_id}.json"
    stats_path = artefact_dir / f"archive_stats_{model_run_id}.json"

    for p in (model_path, fnames_path, stats_path):
        if not p.exists():
            logger.error("run: required artefact not found: %s", p)
            sys.exit(1)

    with open(model_path, "rb") as fh:
        rf = pickle.load(fh)
    logger.info("run: loaded model from %s", model_path)

    feature_names: list[str] = json.loads(fnames_path.read_text())
    logger.info("run: feature_names: %s", feature_names)

    s = json.loads(stats_path.read_text())
    archive_stats = ArchiveStats(mean=s["mean"], std=s["std"])
    logger.info(
        "run: ArchiveStats  mean=%.4f  std=%.4f",
        archive_stats.mean, archive_stats.std,
    )

    # ------------------------------------------------------------------
    # Build inference pixel grid
    # ------------------------------------------------------------------
    minlon, minlat, maxlon, maxlat = (
        args.bbox_minlon, args.bbox_minlat, args.bbox_maxlon, args.bbox_maxlat,
    )
    resolution = args.resolution

    logger.info(
        "run: building pixel grid  bbox=[%.4f, %.4f, %.4f, %.4f]  res=%.6f°",
        minlon, minlat, maxlon, maxlat, resolution,
    )

    pixels, nrows, ncols, lons, lats = _build_pixel_grid(
        minlon, minlat, maxlon, maxlat, resolution,
    )
    n_pixels = len(pixels)
    logger.info("run: pixel grid  %d rows × %d cols = %d pixels", nrows, ncols, n_pixels)

    if n_pixels > 100_000:
        logger.warning(
            "run: large grid (%d pixels) — consider increasing --resolution to reduce cost",
            n_pixels,
        )

    # ------------------------------------------------------------------
    # Stage 0: STAC search + fetch chips for pixel grid
    # ------------------------------------------------------------------
    logger.info("run: Stage 0 — searching STAC")
    bbox = [minlon - 0.01, minlat - 0.01, maxlon + 0.01, maxlat + 0.01]
    items = search_sentinel2(
        bbox=bbox,
        start=args.stac_start,
        end=args.stac_end,
        cloud_cover_max=args.cloud_max,
        endpoint=_config.STAC_ENDPOINT_ELEMENT84,
        collection=_config.S2_COLLECTION,
    )
    if not items:
        logger.error("run: No STAC items found for search parameters.")
        sys.exit(1)
    logger.info("run: found %d STAC items", len(items))

    bands_to_fetch = INFER_BANDS + [SCL_BAND]
    logger.info(
        "run: fetching chips (%d items × %d pixels × %d bands)",
        len(items), n_pixels, len(bands_to_fetch),
    )
    asyncio.run(fetch_chips(
        points=pixels,
        items=items,
        bands=bands_to_fetch,
        window_px=1,
        inputs_dir=inputs_dir,
        scl_filter=True,
        max_concurrent=32,
    ))
    logger.info("run: Stage 0 fetch complete")

    # ------------------------------------------------------------------
    # Composite DOY: midpoint of the STAC search window
    # ------------------------------------------------------------------
    try:
        from datetime import date as _date
        d0 = _date.fromisoformat(args.stac_start)
        d1 = _date.fromisoformat(args.stac_end)
        mid = d0 + (d1 - d0) / 2
        composite_doy = mid.timetuple().tm_yday
    except Exception:
        composite_doy = 270.0
    logger.info("run: composite_doy=%d", composite_doy)

    # ------------------------------------------------------------------
    # Parallel per-pixel feature extraction
    # ------------------------------------------------------------------
    n_workers = args.workers if args.workers else _pool_size(n_pixels)
    logger.info("run: ProcessPoolExecutor workers=%d", n_workers)

    archive_stats_dict = {"mean": archive_stats.mean, "std": archive_stats.std}

    # pixel_id → feature vector (1-D array)
    pixel_features: dict[str, np.ndarray] = {}

    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_ctx,
        initializer=_worker_init_infer,
        initargs=(archive_stats_dict, feature_names),
    ) as pool:
        futures = {
            pool.submit(
                _extract_pixel_features,
                px, items, inputs_dir, float(composite_doy),
            ): px
            for px in pixels
        }
        n_done = 0
        for future in as_completed(futures):
            px = futures[future]
            try:
                pixel_id, fv = future.result()
            except Exception as exc:
                logger.warning("run: worker failed for pixel %s: %s", px[0], exc)
                continue

            if fv is not None:
                pixel_features[pixel_id] = fv

            n_done += 1
            if n_done % max(1, n_pixels // 10) == 0:
                logger.info("run: %d/%d pixels processed (%d with features)",
                            n_done, n_pixels, len(pixel_features))

    logger.info(
        "run: feature extraction complete — %d/%d pixels produced features",
        len(pixel_features), n_pixels,
    )

    if not pixel_features:
        logger.error(
            "run: no feature vectors produced — cannot run inference. "
            "Check chip data, STAC search parameters, and INFER_BANDS."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # RF.predict_proba() across all pixels with features
    # ------------------------------------------------------------------
    # Build feature matrix in pixel order
    pixel_ids_ordered = [px[0] for px in pixels]
    n_features = len(feature_names)

    X = np.full((n_pixels, n_features), np.nan, dtype=np.float64)
    for i, pid in enumerate(pixel_ids_ordered):
        if pid in pixel_features:
            X[i] = pixel_features[pid]

    # Mask for pixels that have features
    valid_mask = ~np.any(np.isnan(X), axis=1)
    n_valid = int(valid_mask.sum())
    logger.info("run: running RF.predict_proba() on %d valid pixels", n_valid)

    prob_flat = np.full(n_pixels, np.nan, dtype=np.float64)
    conf_flat = np.full(n_pixels, np.nan, dtype=np.float64)

    if n_valid > 0:
        X_valid = X[valid_mask]
        proba = rf.predict_proba(X_valid)  # shape: (n_valid, n_classes)

        # Probability of class 1 (presence)
        class_labels = list(rf.classes_)
        if 1 in class_labels:
            pos_idx = class_labels.index(1)
            prob_flat[valid_mask] = proba[:, pos_idx]
        else:
            # Fallback: use last column
            prob_flat[valid_mask] = proba[:, -1]

        # Confidence: max class probability (proxy for prediction confidence)
        conf_flat[valid_mask] = proba.max(axis=1)

    # Reshape to (nrows, ncols)
    prob_raster = prob_flat.reshape(nrows, ncols).astype(np.float32)
    conf_raster = conf_flat.reshape(nrows, ncols).astype(np.float32)

    # Log value distribution
    valid_probs = prob_flat[valid_mask]
    if len(valid_probs) > 0:
        logger.info(
            "run: probability distribution  min=%.3f  max=%.3f  mean=%.3f  std=%.3f",
            float(valid_probs.min()), float(valid_probs.max()),
            float(valid_probs.mean()), float(valid_probs.std()),
        )
        if valid_probs.std() < 1e-6:
            logger.warning(
                "run: probability raster is FLAT (std < 1e-6). "
                "This may indicate degenerate features — check INFER_BANDS and chip data."
            )

    # ------------------------------------------------------------------
    # Build rasterio affine transform
    # ------------------------------------------------------------------
    from rasterio.transform import from_origin
    # from_origin: (west, north, xsize, ysize) — pixel size in CRS units
    transform = from_origin(
        west=float(lons[0]) - resolution / 2,
        north=float(lats[0]) + resolution / 2,
        xsize=resolution,
        ysize=resolution,
    )

    # ------------------------------------------------------------------
    # Write output rasters
    # ------------------------------------------------------------------
    prob_path = output_dir / f"probability_{run_id}.tif"
    conf_path = output_dir / f"confidence_{run_id}.tif"

    _write_raster(prob_path, prob_raster, transform, crs="EPSG:4326")
    _write_raster(conf_path, conf_raster, transform, crs="EPSG:4326")

    logger.info(
        "run: COMPLETE  run_id=%s  probability=%s  confidence=%s",
        run_id, prob_path, conf_path,
    )


# ---------------------------------------------------------------------------
# Subcommand: drop-checkpoint
# ---------------------------------------------------------------------------

def cmd_drop_checkpoint(args: argparse.Namespace) -> None:
    """Delete output rasters for a given run ID."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    if not args.yes:
        logger.error(
            "drop-checkpoint: pass --yes to confirm deletion of run '%s'",
            args.run_id,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    patterns = [
        f"probability_{args.run_id}.tif",
        f"confidence_{args.run_id}.tif",
    ]

    deleted = []
    for name in patterns:
        p = output_dir / name
        if p.exists():
            p.unlink()
            deleted.append(name)
            logger.info("drop-checkpoint: deleted %s", p)

    if not deleted:
        logger.info(
            "drop-checkpoint: nothing to delete for run_id=%s in %s",
            args.run_id, output_dir,
        )
    else:
        logger.info(
            "drop-checkpoint: deleted %d file(s) for run_id=%s",
            len(deleted), args.run_id,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Spectral time series inference pipeline",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- load-testdata ----
    p_ltd = sub.add_parser(
        "load-testdata",
        help="Fetch inference fixture chips from STAC and write the pytest sentinel.",
    )
    p_ltd.set_defaults(func=cmd_load_testdata)

    # ---- run ----
    p_run = sub.add_parser(
        "run",
        help="Full inference pipeline: fetch → composite → features → predict → rasters.",
    )
    p_run.add_argument(
        "--model-run-id", required=True, metavar="ID",
        dest="model_run_id",
        help="Run ID of the trained model artefacts (e.g. 20240101T120000Z).",
    )
    p_run.add_argument(
        "--bbox", required=True, nargs=4,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        type=float,
        dest="bbox_raw",
        help="Bounding box (EPSG:4326) for inference region.",
    )
    p_run.add_argument(
        "--artefact-dir", default=str(PROJECT_ROOT / "outputs"),
        metavar="DIR", dest="artefact_dir",
        help="Directory containing training artefacts. Default: outputs/",
    )
    p_run.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs"),
        metavar="DIR", dest="output_dir",
        help="Directory for output rasters. Default: outputs/",
    )
    p_run.add_argument(
        "--resolution", type=float, default=0.0001, metavar="DEG",
        help="Grid resolution in decimal degrees. Default: 0.0001 (~10m at equator)",
    )
    p_run.add_argument(
        "--stac-start", default="2022-07-01", metavar="DATE", dest="stac_start",
        help="STAC search start date (YYYY-MM-DD). Default: 2022-07-01",
    )
    p_run.add_argument(
        "--stac-end", default="2022-10-31", metavar="DATE", dest="stac_end",
        help="STAC search end date (YYYY-MM-DD). Default: 2022-10-31",
    )
    p_run.add_argument(
        "--cloud-max", type=int, default=30, metavar="PCT", dest="cloud_max",
        help="Max cloud cover %% for STAC filter. Default: 30",
    )
    p_run.add_argument(
        "--run-id", default=None, metavar="ID", dest="run_id",
        help="Run ID for output raster filenames. Default: same as --model-run-id.",
    )
    p_run.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="ProcessPool worker count. Default: auto",
    )
    p_run.set_defaults(func=_cmd_run_trampoline)

    # ---- drop-checkpoint ----
    p_drop = sub.add_parser(
        "drop-checkpoint",
        help="Delete output rasters for a run ID. Requires --yes.",
    )
    p_drop.add_argument("--run-id", required=True, metavar="ID",
                        help="Run ID whose rasters to delete.")
    p_drop.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs"),
        metavar="DIR", dest="output_dir",
        help="Directory containing rasters. Default: outputs/",
    )
    p_drop.add_argument("--yes", action="store_true",
                        help="Confirm deletion (required).")
    p_drop.set_defaults(func=cmd_drop_checkpoint)

    return parser


def _cmd_run_trampoline(args: argparse.Namespace) -> None:
    """Unpack --bbox list into named attributes before calling cmd_run."""
    args.bbox_minlon, args.bbox_minlat, args.bbox_maxlon, args.bbox_maxlat = args.bbox_raw
    cmd_run(args)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
