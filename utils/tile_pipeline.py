"""utils/tile_pipeline.py — Local (non-proxy) tile fetch pipeline.

fetch_tile_local() runs the same pipeline as the proxy VM — tile-first,
COG-block-aligned 2D chunks — but writes chunk parquets directly to
disk instead of streaming them over HTTP.

Output layout:
  out_dir/<year>/<tile_id>/<tile_id>_r{row:02d}_c{col:02d}.parquet  — one file per chunk
  out_dir/<tile_id>/_grid.json                                       — COG origin cache (tile-wide)

Chunk-level resume: existing <tile_id>_r??_c??.parquet files in the tile dir are
skipped individually; the pipeline fetches only chunks whose output file does not
yet exist.  This handles non-contiguous gaps (e.g. a middle row that failed while
later rows completed).

Grid stability: the COG origin (UTM CRS, y_top, x_left, block size) is cached in
_grid.json after the first successful fetch so that resumed runs use the same chunk
numbering as the original run.

The fundamental unit of work is (chunk, year): all (chunk, year) pairs are queued
together so the fetch stage is kept busy across year boundaries.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK_RE = re.compile(r"_r(\d{2})_c(\d{2})$")

# Loggers whose output would tear up the Rich live display.  Raised to WARNING
# during a live run so only genuine errors surface in the terminal.
_NOISY_LOGGERS = [
    "proxy._pipeline",
    "proxy.pipeline",
    "utils.tile_pipeline",
    "utils.fetch",
    "utils.fetch_spec",
    "utils.pipeline_types",
    "utils.s1_collector",
    "utils.pixel_collector",
    "utils.granule_angles",
    "utils.stac",
    "rasterio",
    "urllib3",
    "botocore",
]


def _chunk_key(stem: str) -> tuple[int, int]:
    """Return (chunk_row, chunk_col) from a parquet file stem, or (9999,9999) if not a chunk file."""
    m = _CHUNK_RE.search(stem)
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9999)


def fetch_tile_local(
    tile_id: str,
    years: list[int] | int,
    polygon_geometry,
    out_dir: Path,
    cloud_max: int = 80,
    apply_nbar: bool = True,
    chunk_height_px: int = 1024,
    chunk_width_px: int = 1024,
    max_concurrent: int = 128,
    n_workers: int | None = None,
    items=None,
    calibration_out: Path | None = None,
    point_id_prefix: str = "px",
    work_dir: Path | None = None,
) -> list[Path] | None:
    """Fetch one tile × one-or-more years locally using the same pipeline as the proxy VM.

    All (chunk, year) pairs are queued into a single pipeline so the fetch stage
    is kept busy while extract processes a previous chunk.

    Writes chunk parquets to *out_dir*/<year>/<tile_id>/<tile_id>_r{row:02d}_c{col:02d}.parquet
    with no merge step — callers consume chunks directly.
    No merge step — callers consume chunks directly.

    Parameters
    ----------
    tile_id:
        MGRS tile ID, e.g. "55HBU".
    years:
        Calendar year or list of years to fetch.
    polygon_geometry:
        Shapely geometry defining the area of interest.
    out_dir:
        Root output directory.  Chunks land at out_dir/year/tile_id/<tile_id>_r{row:02d}_c{col:02d}.parquet.
    work_dir:
        Root directory for temporary working data (_work, .tmp files).  Defaults
        to out_dir so behaviour is unchanged when not specified.
    items:
        Optional pre-fetched STAC item list (training pipeline, single-year only).
    calibration_out:
        Optional path for NBAR calibration output.

    Returns
    -------
    list[Path] | None
        Sorted list of written chunk paths across all years (year then row-major
        order within each year), or None if no chunks produced data.
    """
    from proxy._pipeline import run_tile_pipeline_v2 as run_tile_pipeline
    from utils.pipeline_progress import TileProgress

    _years: list[int] = [years] if isinstance(years, int) else list(years)

    _work_root = (work_dir / tile_id) if work_dir is not None else None

    # --- Per-year setup: collect existing chunks, clean .tmp ------------------
    pending_years: list[int] = []
    all_received: list[Path] = []
    skip_keys: set[tuple[int, int, int]] = set()  # (row, col, year)

    for yr in _years:
        tile_dir = out_dir / str(yr) / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)
        yr_work_root = (_work_root / str(yr)) if _work_root is not None else tile_dir

        # Clean up incomplete writes from a prior interrupted run.
        for p in yr_work_root.glob(f"{tile_id}_r??_c??.tmp"):
            p.unlink(missing_ok=True)

        existing = sorted(tile_dir.glob(f"{tile_id}_r??_c??.parquet"),
                          key=lambda p: _chunk_key(p.stem))
        if existing:
            _ids = " ".join(f"r{r:02d}_c{c:02d}" for r, c in (_chunk_key(p.stem) for p in existing))
            logger.info("[%s %d] %d chunks already present (skipping): %s",
                        tile_id, yr, len(existing), _ids)
            all_received.extend(existing)
            for p in existing:
                row, col = _chunk_key(p.stem)
                skip_keys.add((row, col, yr))

        pending_years.append(yr)

    if not pending_years:
        return all_received or None

    pipeline_tmp_root = (_work_root or out_dir / tile_id) / "_work"
    pipeline_tmp_root.mkdir(parents=True, exist_ok=True)

    grid_cache = out_dir / tile_id / "_grid.json"

    chunks_written_by_year: dict[int, list[Path]] = {yr: [] for yr in pending_years}

    # --- Live progress display -----------------------------------------------
    tile_progress = TileProgress(tile_id, pending_years)

    # Suppress verbose loggers so their output doesn't tear up the live display.
    _saved_levels: dict[str, int] = {}
    for _name in _NOISY_LOGGERS:
        _lg = logging.getLogger(_name)
        _saved_levels[_name] = _lg.level
        _lg.setLevel(logging.WARNING)

    # Silence the root logger during the live run so any unlisted logger doesn't
    # escape to stderr and corrupt Rich's cursor tracking.
    _root_lg = logging.getLogger()
    _saved_root_handlers = list(_root_lg.handlers)
    _saved_root_level = _root_lg.level
    _null_handler = logging.NullHandler()
    for _h in _saved_root_handlers:
        _root_lg.removeHandler(_h)
    _root_lg.addHandler(_null_handler)
    _root_lg.setLevel(logging.CRITICAL)

    import queue as _queue
    import threading as _threading

    _copy_q: _queue.Queue = _queue.Queue()
    _copy_errors: list[BaseException] = []

    def _copy_worker() -> None:
        tile_progress.copy_update("waiting")
        while True:
            item = _copy_q.get()
            if item is None:
                break
            chunk_path, dest_tmp, dest, yr, chunk_row, chunk_col = item
            chunk_id = f"r{chunk_row:02d}_c{chunk_col:02d}"
            tile_progress.copy_update("copy", tile_id, yr, chunk_id)
            try:
                shutil.copy2(chunk_path, dest_tmp)
                dest_tmp.replace(dest)
                chunk_path.unlink(missing_ok=True)
            except BaseException as exc:
                _copy_errors.append(exc)
            finally:
                tile_progress.copy_update("waiting")
        tile_progress.copy_update("done")

    _copy_thread = _threading.Thread(target=_copy_worker, daemon=True, name=f"copy_{tile_id}")
    _copy_thread.start()

    try:
        with tile_progress:
            for chunk_row, chunk_col, yr, chunk_path in run_tile_pipeline(
                tile_id=tile_id,
                years=pending_years,
                polygon_geometry=polygon_geometry,
                tmp=pipeline_tmp_root,
                cloud_max=cloud_max,
                apply_nbar=apply_nbar,
                chunk_height_px=chunk_height_px,
                chunk_width_px=chunk_width_px,
                max_concurrent=max_concurrent,
                n_workers=n_workers,
                resume_from_chunk=(0, 0),
                skip_chunks=skip_keys,
                items=items,
                calibration_out=calibration_out,
                point_id_prefix=point_id_prefix,
                log_dir=out_dir,
                progress=tile_progress,
                grid_cache=grid_cache,
            ):
                if _copy_errors:
                    raise _copy_errors[0]
                tile_dir = out_dir / str(yr) / tile_id
                tile_dir.mkdir(parents=True, exist_ok=True)
                dest     = tile_dir / f"{tile_id}_r{chunk_row:02d}_c{chunk_col:02d}.parquet"
                dest_tmp = dest.with_suffix(".tmp")
                chunks_written_by_year[yr].append(dest)
                all_received.append(dest)
                _copy_q.put((chunk_path, dest_tmp, dest, yr, chunk_row, chunk_col))
    finally:
        _copy_q.put(None)
        _copy_thread.join()
        if _copy_errors and not any(isinstance(e, KeyboardInterrupt) for e in _copy_errors):
            raise _copy_errors[0]
        # Restore logger levels regardless of success or failure.
        for _name, _level in _saved_levels.items():
            logging.getLogger(_name).setLevel(_level)
        # Restore root logger handlers.
        _root_lg.removeHandler(_null_handler)
        for _h in _saved_root_handlers:
            _root_lg.addHandler(_h)
        _root_lg.setLevel(_saved_root_level)

    try:
        shutil.rmtree(pipeline_tmp_root)
    except Exception:
        pass

    if not all_received:
        return None

    # Sort: year-major, then row-major within year.
    all_received.sort(key=lambda p: (int(p.parent.parent.name), _chunk_key(p.stem)))
    return all_received
