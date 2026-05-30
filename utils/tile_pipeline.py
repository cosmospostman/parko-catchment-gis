"""utils/tile_pipeline.py — Local (non-proxy) tile fetch pipeline.

fetch_tile_local() runs the same pipeline as the proxy VM — tile-first,
1024-px COG-block-aligned strips — but writes strip parquets directly to
disk instead of streaming them over HTTP.

Output layout:
  out_dir/<year>/<tile_id>/<tile_id>_strip_NN.parquet   — one file per strip
  out_dir/<year>/<tile_id>/.done                        — sentinel when all strips complete

Strip-level resume: existing <tile_id>_strip_NN.parquet files in the tile dir are
included automatically; processing resumes from the first gap.

Tile-level resume: if the .done sentinel exists the function returns immediately.
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
    cloud_max: int = 20,
    apply_nbar: bool = True,
    strip_height_px: int = 1024,
    max_concurrent: int = 128,
    n_workers: int | None = None,
    items=None,
    calibration_out: Path | None = None,
    point_id_prefix: str = "px",
    work_dir: Path | None = None,
) -> list[Path] | None:
    """Fetch one tile×year locally using the same pipeline as the proxy VM.

    Writes strip parquets to *out_dir*/<year>/<tile_id>/<tile_id>_strip_NN.parquet
    and touches a .done sentinel on completion.  No merge step — callers consume
    strips directly.

    Parameters
    ----------
    tile_id:
        MGRS tile ID, e.g. "55HBU".
    year:
        Calendar year to fetch.
    polygon_geometry:
        Shapely geometry defining the area of interest.
    out_dir:
        Root output directory.  Strips land at out_dir/year/tile_id/<tile_id>_strip_NN.parquet.
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
        Sorted list of written strip paths, or None if no strips produced data.
    """
    from proxy._pipeline import run_tile_pipeline_v2 as run_tile_pipeline

    tile_dir      = out_dir / str(year) / tile_id
    done_sentinel = tile_dir / ".done"

    if done_sentinel.exists():
        strips = sorted(tile_dir.glob(f"{tile_id}_strip_??.parquet"))
        logger.info("[%s %d] already done (%d strips) — skipping", tile_id, year, len(strips))
        return strips or None

    tile_dir.mkdir(parents=True, exist_ok=True)

    _work_root = (work_dir / str(year) / tile_id) if work_dir is not None else tile_dir

    # Clean up incomplete writes from a prior interrupted run.
    for p in _work_root.glob(f"{tile_id}_strip_??.tmp"):
        p.unlink(missing_ok=True)

    # Determine resume point from contiguous existing strips.
    # Strip indices are absolute COG block rows (north-to-south), so the
    # first strip on disk may not be strip_00.  Find the lowest index present
    # and walk forward from there to find the first gap.
    complete_strips = sorted(tile_dir.glob(f"{tile_id}_strip_??.parquet"),
                             key=lambda p: int(p.stem.split("_")[-1]))
    resume_from = 0
    if complete_strips:
        indices = [int(p.stem.split("_")[-1]) for p in complete_strips]
        expected = indices[0]
        for idx in indices:
            if idx == expected:
                expected += 1
            else:
                break
        resume_from = expected
        logger.info("[%s %d] resuming from strip %d (%d already complete)",
                    tile_id, year, resume_from, len(complete_strips))

    received_strips: list[Path] = list(complete_strips)

    pipeline_tmp = _work_root / "_work"
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
        point_id_prefix=point_id_prefix,
    ):
        dest     = tile_dir / f"{tile_id}_strip_{strip_idx:02d}.parquet"
        dest_tmp = _work_root / f"{tile_id}_strip_{strip_idx:02d}.tmp"
        shutil.copy2(strip_path, dest_tmp)
        dest_tmp.replace(dest)
        strip_path.unlink(missing_ok=True)
        received_strips.append(dest)
        logger.info("[%s %d] strip %02d written (%.1f MB)",
                    tile_id, year, strip_idx, dest.stat().st_size / 1e6)

    try:
        shutil.rmtree(pipeline_tmp)
    except Exception:
        pass

    if not received_strips:
        logger.warning("[%s %d] no strips — nothing written", tile_id, year)
        return None

    received_strips.sort(key=lambda p: int(p.stem.split("_")[-1]))
    done_sentinel.touch()
    return received_strips
