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
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Memory-budget sizing
# ---------------------------------------------------------------------------

def _system_memory_gb() -> float:
    """Return total physical RAM in GB, or 8.0 if detection fails."""
    try:
        import psutil
        gb = psutil.virtual_memory().total / (1024 ** 3)
        logger.info("System memory detected via psutil: %.1f GB", gb)
        return gb
    except Exception:
        pass
    # Fallback: /proc/meminfo on Linux
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 ** 2)
                    logger.info("System memory detected via /proc/meminfo: %.1f GB", gb)
                    return gb
    except Exception:
        pass
    logger.warning("System memory detection failed — defaulting to 8.0 GB budget")
    return 8.0


def _budget_params(memory_budget_gb: float) -> dict:
    """Derive pipeline sizing parameters from a memory budget in GB.

    Returns a dict with keys:
      strip_height_px       — UTM pixel rows per strip (None = no strip decomposition)
      max_extract_years     — concurrent years in Phase B extraction
      max_concurrent_strips — concurrent strips within one year's extraction
      target_extraction_gb  — target_gb passed to _auto_n_workers in collect()
      n_s1_workers          — S1 extraction thread count

    Invariant: max_concurrent_strips × target_extraction_gb + ~2 GB baseline ≤ budget.

    Strip height controls the Polars sort peak (the floor that can't be driven
    lower without changing the data model).  At ≥32 GB, strips are disabled and
    the full bbox is processed as before.
    """
    gb = memory_budget_gb
    if gb >= 32:
        return dict(strip_height_px=None, max_extract_years=2,  max_concurrent_strips=1, target_extraction_gb=8.0, n_s1_workers=4)
    if gb >= 16:
        return dict(strip_height_px=2000, max_extract_years=1,  max_concurrent_strips=2, target_extraction_gb=4.0, n_s1_workers=4)
    if gb >= 8:
        return dict(strip_height_px=1000, max_extract_years=1,  max_concurrent_strips=1, target_extraction_gb=2.0, n_s1_workers=2)
    # ≥ 4 GB (minimum supported)
    return dict(strip_height_px=500,  max_extract_years=1,  max_concurrent_strips=1, target_extraction_gb=1.0, n_s1_workers=1)



def fetch_spec(
    spec: FetchSpec,
    cloud_max: int = 30,
    apply_nbar: bool = True,
    max_concurrent: int = 32,
    items=None,
    n_workers: int | None = None,
    n_s1_workers: int | None = None,
    calibration_out: Path | None = None,
    max_extract_years: int | None = None,
    memory_budget_gb: float | None = None,
) -> dict[int, list[Path]]:
    """Run the three-stage fetch pipeline for a FetchSpec.

    Decouples network I/O from memory-intensive extraction:

      Phase A (all years in parallel, network-bound):
        collect_s2(phases={"fetch"}) — populate .npz patch cache, low memory
        collect_s1 fetch_patches     — populate S1 .npz patch cache

      Phase B (at most max_extract_years concurrent, memory-bound):
        collect_s2(phases={"extract"}) → <out_dir>/<year>/<tile_id>.s2.parquet
        collect_s1 extraction          → <out_dir>/<year>/<tile_id>.s1.parquet
        merge_tile / merge_strips      → <out_dir>/<year>/<tile_id>.parquet

    ``memory_budget_gb`` is the primary sizing knob.  Defaults to total system
    RAM (auto-detected).  From it, _budget_params() derives max_extract_years,
    target_extraction_gb, and n_s1_workers.  All of these can still be
    overridden individually via the explicit keyword arguments below.

    ``items`` may be a pre-fetched, deduplicated STAC item list (training
    pipeline passes the tile-level search result here to avoid redundant
    searches across regions on the same tile).

    Returns {year: [merged_parquet_paths]}.
    """
    from utils.pixel_collector import collect
    from utils.s1_collector import collect_s1_for_tile
    from utils.parquet_utils import merge_tile, merge_strips

    if spec.out_dir is None:
        raise ValueError(f"FetchSpec {spec.id!r}: out_dir must be set before calling fetch_spec()")

    # --- Resolve memory budget and derived params ----------------------------
    _budget_gb = memory_budget_gb if memory_budget_gb is not None else _system_memory_gb()
    _params = _budget_params(_budget_gb)

    _max_extract: int = max_extract_years if max_extract_years is not None else _params["max_extract_years"]
    _target_gb: float = _params["target_extraction_gb"]
    _n_s1: int = n_s1_workers if n_s1_workers is not None else _params["n_s1_workers"]

    _budget_source = "explicit" if memory_budget_gb is not None else "auto-detected"
    logger.info(
        "fetch_spec %s: memory budget %.0f GB (%s) → "
        "max_extract_years=%d, target_extraction_gb=%.0f, n_s1_workers=%d",
        spec.id, _budget_gb, _budget_source,
        _max_extract, _target_gb, _n_s1,
    )

    results: dict[int, list[Path]] = {}
    years = sorted(spec.years)

    # n_workers=None lets collect()/_auto_n_workers() scale from _target_gb;
    # if the caller explicitly passes n_workers we honour it unchanged.
    _resolved_n_workers = n_workers  # None → auto-scaled inside collect()

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
        n_workers=_resolved_n_workers,
        target_extraction_gb=_target_gb,
    )

    # -------------------------------------------------------------------------
    # Full-bbox path (no strips)
    #
    # Phase A → Phase B pipeline: years start Phase B as soon as their own
    # Phase A completes (not when all years finish).
    #
    # Two pools run concurrently:
    #   fetch_pool   — unbounded, network I/O only, low memory per worker
    #   extract_pool — capped at max_extract_years, memory-intensive
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
        n_tiles = len(s2_paths)
        for tile_idx, s2_path in enumerate(s2_paths, 1):
            tile_id = s2_path.name.replace(".s2.parquet", "")
            s1_path = year_dir / f"{tile_id}.s1.parquet"
            out_path = year_dir / f"{tile_id}.parquet"

            logger.info(
                "fetch_spec %s year %d: tile %d/%d — %s",
                spec.id, year, tile_idx, n_tiles, tile_id,
            )

            # Use per-tile bbox so S1 COG reads only cover this tile's pixel extent,
            # not the full catchment.  Derived from actual pixel coords in the parquet.
            import pyarrow.compute as _pc
            import pyarrow.parquet as _pq
            _coords = _pq.ParquetFile(s2_path).read(columns=["lon", "lat"])
            _lon_col = _coords.column("lon").combine_chunks()
            _lat_col = _coords.column("lat").combine_chunks()
            _tile_bbox = [
                float(_pc.min(_lon_col).as_py()),
                float(_pc.min(_lat_col).as_py()),
                float(_pc.max(_lon_col).as_py()),
                float(_pc.max(_lat_col).as_py()),
            ]
            del _coords, _lon_col, _lat_col

            s1_cache = spec.cache_dir if spec.cache_dir is not None else _S1_CACHE_DIR
            collect_s1_for_tile(
                s2_path=s2_path,
                bbox_wgs84=_tile_bbox,
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                out_path=s1_path,
                cache_dir=s1_cache,
                max_concurrent=_n_s1,
            )

            merge_tile(
                s2_path=s2_path,
                s1_path=s1_path if s1_path.exists() else None,
                out_path=out_path,
            )
            merged_paths.append(out_path)

        return year, merged_paths

    _phase_a_done: set[int] = set()
    extract_futs: dict = {}
    with ThreadPoolExecutor(max_workers=len(years)) as fetch_pool, \
         ThreadPoolExecutor(max_workers=_max_extract) as extract_pool:

        fetch_futs = {fetch_pool.submit(_fetch_patches_year, yr): yr for yr in years}

        for fut in as_completed(fetch_futs):
            yr = fetch_futs[fut]
            try:
                fut.result()
                _phase_a_done.add(yr)
                if len(_phase_a_done) == len(years):
                    logger.info("fetch_spec %s: Phase A complete — starting extraction", spec.id)
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
