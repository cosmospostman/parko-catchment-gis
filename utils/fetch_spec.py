"""utils/fetch_spec.py — Unified fetch pipeline for S2+S1 pixel observations.

Both the location pipeline and training pipeline share the same three-stage
fetch: bbox+years → collect S2 → collect S1 → merge into one parquet per tile
per year.  FetchSpec + fetch_spec() unify that logic so there is one
implementation used by both callers.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchSpec:
    id: str                         # "longreach" or "lake_mueller_presence"
    bbox: list[float]               # [lon_min, lat_min, lon_max, lat_max]
    years: list[int]
    point_id_prefix: str            # "px" for locations, region.id for training
    geometry: object | None = None  # polygon mask (locations); None for training regions
    label: str | None = None        # "presence"/"absence" (training); None for locations
    out_dir: Path | None = None     # caller controls output root
    cache_dir: Path | None = None   # explicit override; training uses per-region subdir


def fetch_spec(
    spec: FetchSpec,
    cloud_max: int = 30,
    apply_nbar: bool = True,
    max_concurrent: int = 32,
    items=None,
    n_workers: int | None = None,
    n_s1_workers: int = 4,
    calibration_out: Path | None = None,
    max_extract_years: int | None = None,
) -> dict[int, list[Path]]:
    """Run the three-stage fetch pipeline for a FetchSpec.

    Decouples network I/O from memory-intensive extraction:

      Phase A (all years in parallel, network-bound):
        collect_s2(phases={"fetch"}) — populate .npz patch cache, low memory
        collect_s1 fetch_patches     — populate S1 .npz patch cache

      Phase B (at most max_extract_years concurrent, memory-bound):
        collect_s2(phases={"extract"}) → <out_dir>/<year>/<tile_id>.s2.parquet
        collect_s1 extraction          → <out_dir>/<year>/<tile_id>.s1.parquet
        merge_tile                     → <out_dir>/<year>/<tile_id>.parquet

    ``max_extract_years`` caps how many years run Phase B concurrently.
    Defaults to 2 to keep peak RAM bounded for large multi-tile locations;
    set to len(years) to restore the old fully-parallel behaviour.

    ``items`` may be a pre-fetched, deduplicated STAC item list (training
    pipeline passes the tile-level search result here to avoid redundant
    searches across regions on the same tile).

    Returns {year: [merged_parquet_paths]}.
    """
    from utils.pixel_collector import collect
    from utils.s1_collector import collect_s1_for_tile, _DEFAULT_CACHE_DIR as _S1_CACHE_DIR
    from utils.parquet_utils import merge_tile

    if spec.out_dir is None:
        raise ValueError(f"FetchSpec {spec.id!r}: out_dir must be set before calling fetch_spec()")

    results: dict[int, list[Path]] = {}
    years = sorted(spec.years)

    _max_extract = max_extract_years if max_extract_years is not None else 2

    _collect_kwargs = dict(
        bbox_wgs84=spec.bbox,
        cloud_max=cloud_max,
        cache_dir=spec.cache_dir,
        apply_nbar=apply_nbar,
        max_concurrent=max_concurrent,
        items=items,
        point_id_prefix=spec.point_id_prefix,
        calibration_out=calibration_out,
        geometry=spec.geometry,
        n_workers=n_workers,
    )

    # -------------------------------------------------------------------------
    # Phase A → Phase B pipeline: years are processed in two stages but start
    # Phase B as soon as their own Phase A completes, not when all years finish.
    #
    # Two pools run concurrently:
    #   fetch_pool  — unbounded, network I/O only, low memory per worker
    #   extract_pool — capped at max_extract_years, memory-intensive
    #
    # As each Phase A future completes, its year is immediately submitted to
    # extract_pool.  extract_pool's cap is the only thing bounding peak RAM.
    # -------------------------------------------------------------------------
    logger.info(
        "fetch_spec %s: %d years — Phase A (all parallel) → Phase B (%d concurrent)",
        spec.id, len(years), _max_extract,
    )

    def _fetch_patches_year(year: int) -> int:
        year_dir = spec.out_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        collect(
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            out_dir=year_dir,
            phases={"fetch"},
            **_collect_kwargs,
        )
        return year

    def _extract_year(year: int) -> tuple[int, list[Path]]:
        year_dir = spec.out_dir / str(year)

        # Stage 1: S2 extract (reads from .npz cache, writes .s2.parquet)
        s2_paths = collect(
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            out_dir=year_dir,
            phases={"extract"},
            **_collect_kwargs,
        )

        if not s2_paths:
            s2_paths = sorted(
                p for p in year_dir.glob("*.s2.parquet")
                if not p.stem.startswith("_") and "_tmp" not in p.stem
            )

        if not s2_paths:
            logger.warning("fetch_spec %s year %d: no S2 data", spec.id, year)
            return year, []

        merged_paths: list[Path] = []
        for s2_path in s2_paths:
            tile_id = s2_path.name.replace(".s2.parquet", "")
            s1_path = year_dir / f"{tile_id}.s1.parquet"
            out_path = year_dir / f"{tile_id}.parquet"

            # Stage 2: S1 (idempotent — collect_s1_for_tile skips if already done)
            s1_cache = spec.cache_dir if spec.cache_dir is not None else _S1_CACHE_DIR
            collect_s1_for_tile(
                s2_path=s2_path,
                bbox_wgs84=spec.bbox,
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                out_path=s1_path,
                cache_dir=s1_cache,
                n_workers=n_s1_workers,
            )

            # Stage 3: merge (idempotent — merge_tile skips if row count matches)
            merge_tile(
                s2_path=s2_path,
                s1_path=s1_path if s1_path.exists() else None,
                out_path=out_path,
            )
            merged_paths.append(out_path)

        return year, merged_paths

    extract_futs: dict = {}
    with ThreadPoolExecutor(max_workers=len(years)) as fetch_pool, \
         ThreadPoolExecutor(max_workers=_max_extract) as extract_pool:

        fetch_futs = {fetch_pool.submit(_fetch_patches_year, yr): yr for yr in years}

        for fut in as_completed(fetch_futs):
            yr = fetch_futs[fut]
            try:
                fut.result()
                logger.info("fetch_spec %s year %d: patches ready, queuing extract", spec.id, yr)
            except Exception as exc:
                logger.error("fetch_spec %s year %d fetch failed: %s", spec.id, yr, exc, exc_info=exc)
                results[yr] = []
                continue
            extract_futs[extract_pool.submit(_extract_year, yr)] = yr

    for fut in as_completed(extract_futs):
        yr = extract_futs[fut]
        try:
            _, paths = fut.result()
            results[yr] = paths
        except Exception as exc:
            logger.error("fetch_spec %s year %d extract failed: %s", spec.id, yr, exc, exc_info=exc)
            results[yr] = []

    return results
