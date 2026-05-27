"""utils/tile_pipeline.py — Local (non-proxy) tile fetch pipeline.

fetch_tile_local() runs the same pipeline as the proxy VM — tile-first,
1024-px COG-block-aligned strips — but writes strip parquets directly to
disk instead of streaming them over HTTP.

The mechanics are identical to proxy/client.py's receive-and-write loop:
  - done-sentinel check (skip if already complete)
  - contiguous-strip resume scan
  - atomic .tmp → rename write per strip
  - merge_strips() to assemble the final tile parquet
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_tile_local(
    tile_id: str,
    year: int,
    polygon_geometry,
    out_dir: Path,
    tmp_dir: Path,
    cloud_max: int = 20,
    apply_nbar: bool = True,
    strip_height_px: int = 1024,
    max_concurrent: int = 32,
    n_workers: int | None = None,
    items=None,
    calibration_out: Path | None = None,
) -> Path | None:
    """Fetch one tile×year locally using the same pipeline as the proxy VM.

    Writes strip shards to *tmp_dir*/<tile_id>/<year>/ during processing,
    merges them into *out_dir*/<year>/<tile_id>.parquet on completion, and
    writes a .done sentinel to mark the tile as complete.

    Strip-level resume: existing strip_NNNN.parquet files in the tmp dir are
    included in the final merge; processing resumes from the first gap.

    Tile-level resume: if <out_dir>/<year>/<tile_id>.parquet and its .done
    sentinel both exist the function returns immediately without any I/O.

    Parameters
    ----------
    tile_id:
        MGRS tile ID, e.g. "55HBU".
    year:
        Calendar year to fetch.
    polygon_geometry:
        Shapely geometry defining the area of interest.
    out_dir:
        Root output directory.  Final parquet lands at out_dir/year/tile_id.parquet.
    tmp_dir:
        Scratch root.  Strip shards land at tmp_dir/tile_id/year/strip_NNNN.parquet.
    items:
        Optional pre-fetched STAC item list (training pipeline).
    calibration_out:
        Optional path for NBAR calibration output.

    Returns
    -------
    Path | None
        The written output path, or None if no strips produced data.
    """
    from proxy._pipeline import run_tile_pipeline
    from utils.parquet_utils import merge_strips

    out_path      = out_dir / str(year) / f"{tile_id}.parquet"
    done_sentinel = out_path.with_suffix(".done")

    if done_sentinel.exists() and out_path.exists():
        logger.info("[%s %d] already done — skipping", tile_id, year)
        return out_path

    tile_tmp = tmp_dir / tile_id / str(year)
    tile_tmp.mkdir(parents=True, exist_ok=True)

    # Clean up incomplete writes from a prior interrupted run.
    for p in tile_tmp.glob("strip_????.tmp"):
        p.unlink(missing_ok=True)

    # Determine resume point from contiguous existing strips.
    complete_strips = sorted(tile_tmp.glob("strip_????.parquet"))
    resume_from = 0
    if complete_strips:
        expected = 0
        for p in complete_strips:
            if int(p.stem.split("_")[1]) == expected:
                expected += 1
            else:
                break
        resume_from = expected
        logger.info("[%s %d] resuming from strip %d (%d already complete)",
                    tile_id, year, resume_from, resume_from)

    received_strips: list[Path] = list(complete_strips)

    pipeline_tmp = tile_tmp / "_work"
    pipeline_tmp.mkdir(parents=True, exist_ok=True)

    for strip_idx, strip_path in run_tile_pipeline(
        tile_id=tile_id,
        year=year,
        polygon_geometry=polygon_geometry,
        tmp=pipeline_tmp,
        cloud_max=cloud_max,
        apply_nbar=apply_nbar,
        strip_height_px=strip_height_px,
        max_concurrent=max_concurrent,
        n_workers=n_workers,
        resume_from_strip=resume_from,
        items=items,
        calibration_out=calibration_out,
    ):
        dest     = tile_tmp / f"strip_{strip_idx:04d}.parquet"
        dest_tmp = tile_tmp / f"strip_{strip_idx:04d}.tmp"
        shutil.copy2(strip_path, dest_tmp)
        dest_tmp.replace(dest)
        strip_path.unlink(missing_ok=True)
        received_strips.append(dest)
        logger.info("[%s %d] strip %04d written (%.1f MB)",
                    tile_id, year, strip_idx, dest.stat().st_size / 1e6)

    try:
        shutil.rmtree(pipeline_tmp)
    except Exception:
        pass

    if not received_strips:
        logger.warning("[%s %d] no strips — skipping merge", tile_id, year)
        return None

    received_strips.sort(key=lambda p: int(p.stem.split("_")[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[%s %d] merging %d strips → %s", tile_id, year, len(received_strips), out_path)
    merge_strips(received_strips, out_path)
    done_sentinel.touch()

    for p in received_strips:
        p.unlink(missing_ok=True)
    try:
        tile_tmp.rmdir()
    except OSError:
        pass

    return out_path
