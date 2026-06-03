"""utils/tile_pipeline.py — Local (non-proxy) tile fetch pipeline.

fetch_tile_local() runs the same pipeline as the proxy VM — tile-first,
COG-block-aligned 2D chunks — but writes chunk parquets directly to
disk instead of streaming them over HTTP.

Output layout:
  out_dir/<year>/<tile_id>/<tile_id>_r{row:02d}_c{col:02d}.parquet  — one file per chunk
  out_dir/<year>/<tile_id>/.done                                     — sentinel when all chunks complete

Chunk-level resume: existing <tile_id>_r??_c??.parquet files in the tile dir are
skipped individually; the pipeline fetches only chunks whose output file does not
yet exist.  This handles non-contiguous gaps (e.g. a middle row that failed while
later rows completed).

Tile-level resume: if the .done sentinel exists the function returns immediately.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK_RE = re.compile(r"_r(\d{2})_c(\d{2})$")


def _chunk_key(stem: str) -> tuple[int, int]:
    """Return (chunk_row, chunk_col) from a parquet file stem, or (9999,9999) if not a chunk file."""
    m = _CHUNK_RE.search(stem)
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9999)


def fetch_tile_local(
    tile_id: str,
    year: int,
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
    """Fetch one tile×year locally using the same pipeline as the proxy VM.

    Writes chunk parquets to *out_dir*/<year>/<tile_id>/<tile_id>_r{row:02d}_c{col:02d}.parquet
    and touches a .done sentinel on completion.  No merge step — callers consume
    chunks directly.

    Parameters
    ----------
    tile_id:
        MGRS tile ID, e.g. "55HBU".
    year:
        Calendar year to fetch.
    polygon_geometry:
        Shapely geometry defining the area of interest.
    out_dir:
        Root output directory.  Chunks land at out_dir/year/tile_id/<tile_id>_r{row:02d}_c{col:02d}.parquet.
    work_dir:
        Root directory for temporary working data (_work, .tmp files).  Defaults
        to out_dir so behaviour is unchanged when not specified.
    items:
        Optional pre-fetched STAC item list (training pipeline).
    calibration_out:
        Optional path for NBAR calibration output.

    Returns
    -------
    list[Path] | None
        Sorted list of written chunk paths (row-major order), or None if no chunks produced data.
    """
    from proxy._pipeline import run_tile_pipeline_v2 as run_tile_pipeline

    tile_dir      = out_dir / str(year) / tile_id
    done_sentinel = tile_dir / ".done"

    if done_sentinel.exists():
        chunks = sorted(tile_dir.glob(f"{tile_id}_r??_c??.parquet"),
                        key=lambda p: _chunk_key(p.stem))
        logger.info("[%s %d] already done (%d chunks) — skipping", tile_id, year, len(chunks))
        return chunks or None

    tile_dir.mkdir(parents=True, exist_ok=True)

    _work_root = (work_dir / str(year) / tile_id) if work_dir is not None else tile_dir

    # Clean up incomplete writes from a prior interrupted run.
    for p in _work_root.glob(f"{tile_id}_r??_c??.tmp"):
        p.unlink(missing_ok=True)

    # Collect existing complete chunks; the pipeline will skip any whose dest
    # already exists, allowing non-contiguous gaps to be filled without re-fetching
    # chunks that completed in a prior run.
    complete_chunks = sorted(tile_dir.glob(f"{tile_id}_r??_c??.parquet"),
                             key=lambda p: _chunk_key(p.stem))
    existing_keys: set[tuple[int, int]] = {_chunk_key(p.stem) for p in complete_chunks}
    if complete_chunks:
        logger.info("[%s %d] %d chunks already present — will skip existing, fetch missing",
                    tile_id, year, len(complete_chunks))

    received_chunks: list[Path] = list(complete_chunks)

    pipeline_tmp = _work_root / "_work"
    pipeline_tmp.mkdir(parents=True, exist_ok=True)

    for chunk_row, chunk_col, chunk_path in run_tile_pipeline(
        tile_id=tile_id,
        year=year,
        polygon_geometry=polygon_geometry,
        tmp=pipeline_tmp,
        cloud_max=cloud_max,
        apply_nbar=apply_nbar,
        chunk_height_px=chunk_height_px,
        chunk_width_px=chunk_width_px,
        max_concurrent=max_concurrent,
        n_workers=n_workers,
        resume_from_chunk=(0, 0),
        skip_chunks=existing_keys,
        items=items,
        calibration_out=calibration_out,
        point_id_prefix=point_id_prefix,
    ):
        dest     = tile_dir / f"{tile_id}_r{chunk_row:02d}_c{chunk_col:02d}.parquet"
        dest_tmp = dest.with_suffix(".tmp")
        shutil.copy2(chunk_path, dest_tmp)
        dest_tmp.replace(dest)
        chunk_path.unlink(missing_ok=True)
        received_chunks.append(dest)
        logger.info("[%s %d] chunk (%d,%d) written (%.1f MB)",
                    tile_id, year, chunk_row, chunk_col, dest.stat().st_size / 1e6)

    try:
        shutil.rmtree(pipeline_tmp)
    except Exception:
        pass

    if not received_chunks:
        logger.warning("[%s %d] no chunks — nothing written", tile_id, year)
        return None

    received_chunks.sort(key=lambda p: _chunk_key(p.stem))
    done_sentinel.touch()
    return received_chunks
