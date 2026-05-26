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


def _strip_bboxes(bbox_wgs84: list[float], strip_height_px: int, resolution_m: float = 10.0) -> list[list[float]]:
    """Divide bbox_wgs84 into horizontal strips of strip_height_px UTM rows.

    Strips share the full longitude extent of the original bbox and are ordered
    south-to-north (ascending northing), matching the pixel sort order so the
    N-way merge emits them in sequence with minimal heap churn.

    Returns a list of [lon_min, lat_min, lon_max, lat_max] bboxes in WGS84.
    Each strip's lat_max equals the next strip's lat_min (no overlap, no gap).
    """
    import numpy as np
    from pyproj import Transformer

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    # Derive UTM CRS from bbox centre (same logic as pixel_collector)
    lon_c = (lon_min + lon_max) / 2
    lat_c = (lat_min + lat_max) / 2
    zone = min(int((lon_c + 180) / 6) + 1, 60)
    epsg = 32600 + zone if lat_c >= 0 else 32700 + zone
    utm_crs = f"EPSG:{epsg}"

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    # Use centre longitude for the UTM transforms to minimise round-trip distortion.
    # Edge-of-zone distortion causes y → WGS84 lat to overshoot by ~0.1° at lon_min/max.
    _, y_min = to_utm.transform(lon_c, lat_min)
    _, y_max = to_utm.transform(lon_c, lat_max)

    strip_m = strip_height_px * resolution_m
    y_edges = np.arange(y_min, y_max, strip_m).tolist()
    y_edges.append(y_max)

    strips: list[list[float]] = []
    for i in range(len(y_edges) - 1):
        y0, y1 = y_edges[i], y_edges[i + 1]
        _, slat_min = to_wgs.transform(lon_c, y0)
        _, slat_max = to_wgs.transform(lon_c, y1)
        strips.append([lon_min, slat_min, lon_max, slat_max])

    # Clamp boundary edges so coverage exactly matches the original bbox.
    # UTM round-trip distortion (edge-of-zone) can shift outer strip lats by ~0.1°;
    # clamping ensures no coverage is dropped at bbox_wgs84's southern/northern edges.
    # Strips whose clamped extent collapses (lat_min ≥ lat_max) are dropped — this
    # only happens when the second-to-last strip's WGS84 lat already overshoots lat_max,
    # meaning coverage is already complete.
    if strips:
        strips[0][1] = lat_min
        strips[-1][3] = lat_max
    strips = [s for s in strips if s[1] < s[3]]

    return strips


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
    RAM (auto-detected).  From it, _budget_params() derives strip_height_px,
    max_extract_years, max_concurrent_strips, target_extraction_gb, and
    n_s1_workers.  All of these can still be overridden individually via the
    explicit keyword arguments below.

    When strip_height_px > 0 the bbox is divided into horizontal UTM strips of
    that many pixel rows.  Each strip runs the full three-stage pipeline
    independently (with a shared .npz patch cache so COG fetches are not
    repeated).  After all strips complete, merge_strips() assembles them into
    the same final output path that the non-strip path produces.  Downstream
    consumers see no difference.

    ``items`` may be a pre-fetched, deduplicated STAC item list (training
    pipeline passes the tile-level search result here to avoid redundant
    searches across regions on the same tile).  When stripping is active the
    same item list is passed to every strip — COG reads are skipped for strips
    whose patches are already cached.

    Returns {year: [merged_parquet_paths]}.
    """
    from utils.pixel_collector import collect
    from utils.s1_collector import collect_s1_for_tile, _DEFAULT_CACHE_DIR as _S1_CACHE_DIR
    from utils.parquet_utils import merge_tile, merge_strips

    if spec.out_dir is None:
        raise ValueError(f"FetchSpec {spec.id!r}: out_dir must be set before calling fetch_spec()")

    # --- Resolve memory budget and derived params ----------------------------
    _budget_gb = memory_budget_gb if memory_budget_gb is not None else _system_memory_gb()
    _params = _budget_params(_budget_gb)

    _strip_height: int | None = _params["strip_height_px"]
    _max_extract: int = max_extract_years if max_extract_years is not None else _params["max_extract_years"]
    _max_strips: int = _params["max_concurrent_strips"]
    _target_gb: float = _params["target_extraction_gb"]
    _n_s1: int = n_s1_workers if n_s1_workers is not None else _params["n_s1_workers"]

    _budget_source = "explicit" if memory_budget_gb is not None else "auto-detected"
    logger.info(
        "fetch_spec %s: memory budget %.0f GB (%s) → "
        "strip_height=%s, max_extract_years=%d, max_concurrent_strips=%d, "
        "target_extraction_gb=%.0f, n_s1_workers=%d",
        spec.id, _budget_gb, _budget_source,
        f"{_strip_height} px" if _strip_height else "none (full bbox)",
        _max_extract, _max_strips, _target_gb, _n_s1,
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
    # Strip decomposition
    # When _strip_height is set, split the bbox into horizontal strips and run
    # each through the full three-stage pipeline independently.  After all
    # strips complete, merge_strips() reassembles them.
    #
    # Key properties:
    #   - Same cache_dir for all strips → COG patches fetched once, shared
    #   - Same items list for all strips → no redundant STAC searches
    #   - Strips are south-to-north (ascending northing) → the N-way merge
    #     emits whole blocks with minimal heap churn
    #   - max_concurrent_strips bounds how many strips run Phase B at once
    # -------------------------------------------------------------------------
    if _strip_height is not None:
        strips = _strip_bboxes(spec.bbox, _strip_height)
        logger.info("fetch_spec %s: %d strips of ≤%d px height", spec.id, len(strips), _strip_height)
        return _fetch_spec_strips(
            spec=spec,
            strips=strips,
            years=years,
            collect_kwargs_base=_collect_kwargs,
            max_extract_years=_max_extract,
            max_concurrent_strips=_max_strips,
            n_s1_workers=_n_s1,
            s1_cache_dir=spec.cache_dir if spec.cache_dir is not None else _S1_CACHE_DIR,
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


def _fetch_spec_strips(
    spec: FetchSpec,
    strips: list[list[float]],
    years: list[int],
    collect_kwargs_base: dict,
    max_extract_years: int,
    max_concurrent_strips: int,
    n_s1_workers: int,
    s1_cache_dir: Path,
) -> dict[int, list[Path]]:
    """Strip-decomposed fetch pipeline.

    For each year:
      Phase A: fetch patches for all strips in parallel (network-bound, shared cache)
      Phase B: extract strips sequentially (capped at max_concurrent_strips),
               then merge_strips() reassembles them into one parquet per tile.

    Strip outputs land in:
      <out_dir>/<year>/strips/<strip_NNNN>/<tile_id>.parquet

    Final merged outputs:
      <out_dir>/<year>/<tile_id>.parquet   ← same as the no-strip path
    """
    from utils.pixel_collector import collect
    from utils.s1_collector import collect_s1_for_tile
    from utils.parquet_utils import merge_tile, merge_strips

    results: dict[int, list[Path]] = {}
    n_strips = len(strips)

    def _strip_dir(year: int, strip_idx: int) -> Path:
        return spec.out_dir / str(year) / "strips" / f"strip_{strip_idx:04d}"

    def _fetch_patches_strip(year: int, strip_idx: int, strip_bbox: list[float]) -> None:
        sdir = _strip_dir(year, strip_idx)
        sdir.mkdir(parents=True, exist_ok=True)
        kwargs = {**collect_kwargs_base, "bbox_wgs84": strip_bbox}
        collect(
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            out_dir=sdir,
            phases={"fetch"},
            **kwargs,
        )

    def _extract_strip(year: int, strip_idx: int, strip_bbox: list[float]) -> list[Path]:
        sdir = _strip_dir(year, strip_idx)
        kwargs = {**collect_kwargs_base, "bbox_wgs84": strip_bbox}
        s2_paths = collect(
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            out_dir=sdir,
            phases={"extract"},
            **kwargs,
        )
        if not s2_paths:
            s2_paths = sorted(
                p for p in sdir.glob("*.s2.parquet")
                if not p.stem.startswith("_") and "_tmp" not in p.stem
            )
        if not s2_paths:
            logger.warning("fetch_spec %s year %d strip %d: no S2 data", spec.id, year, strip_idx)
            return []

        strip_merged: list[Path] = []
        for s2_path in s2_paths:
            tile_id = s2_path.name.replace(".s2.parquet", "")
            s1_path = sdir / f"{tile_id}.s1.parquet"
            out_path = sdir / f"{tile_id}.parquet"

            collect_s1_for_tile(
                s2_path=s2_path,
                bbox_wgs84=strip_bbox,
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                out_path=s1_path,
                cache_dir=s1_cache_dir,
                max_concurrent=n_s1_workers,
            )

            merge_tile(
                s2_path=s2_path,
                s1_path=s1_path if s1_path.exists() else None,
                out_path=out_path,
            )
            strip_merged.append(out_path)

        return strip_merged

    def _process_year(year: int) -> tuple[int, list[Path]]:
        year_dir = spec.out_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        _phase_a_workers = min(n_strips, 8)
        logger.info(
            "fetch_spec %s year %d: Phase A — fetching %d strips (%d concurrent threads)",
            spec.id, year, n_strips, _phase_a_workers,
        )

        # Phase A: fetch all strips in parallel (shared cache_dir — COG reads happen once).
        # Cap at 8 threads: each thread runs asyncio.run(fetch_patches(..., max_concurrent=32)),
        # so 8 threads already saturates a typical link (8 × 32 = 256 concurrent requests).
        with ThreadPoolExecutor(max_workers=_phase_a_workers) as fetch_pool:
            fetch_futs = [
                fetch_pool.submit(_fetch_patches_strip, year, i, bbox)
                for i, bbox in enumerate(strips)
            ]
            for fut in as_completed(fetch_futs):
                fut.result()  # re-raise network errors

        logger.info("fetch_spec %s year %d: Phase A complete — starting extraction", spec.id, year)
        logger.info(
            "fetch_spec %s year %d: Phase B — extracting %d strips (%d concurrent)",
            spec.id, year, n_strips, max_concurrent_strips,
        )

        # Phase B: extract strips, bounded concurrency
        strip_results: dict[int, list[Path]] = {}
        with ThreadPoolExecutor(max_workers=max_concurrent_strips) as extract_pool:
            ext_futs = {
                extract_pool.submit(_extract_strip, year, i, bbox): i
                for i, bbox in enumerate(strips)
            }
            for fut in as_completed(ext_futs):
                i = ext_futs[fut]
                try:
                    strip_results[i] = fut.result()
                except Exception as exc:
                    logger.error(
                        "fetch_spec %s year %d strip %d extract failed: %s",
                        spec.id, year, i, exc, exc_info=exc,
                    )
                    strip_results[i] = []

        # Collect per-tile strip parquets (strips are already S2+S1 merged)
        # Group by tile_id: {tile_id: [strip_0_path, strip_1_path, ...]}
        tile_strip_paths: dict[str, list[Path]] = {}
        for i in range(n_strips):
            for strip_path in strip_results.get(i, []):
                tile_id = strip_path.name.replace(".parquet", "")
                tile_strip_paths.setdefault(tile_id, []).append(strip_path)

        # Final merge: N-way merge of sorted strips → one parquet per tile
        merged_paths: list[Path] = []
        n_tiles = len(tile_strip_paths)
        for tile_idx, (tile_id, strip_parquets) in enumerate(sorted(tile_strip_paths.items()), 1):
            out_path = year_dir / f"{tile_id}.parquet"
            logger.info(
                "fetch_spec %s year %d: tile %d/%d — %s",
                spec.id, year, tile_idx, n_tiles, tile_id,
            )
            if len(strip_parquets) == 1:
                # Only one strip produced data for this tile — rename, no merge needed
                import shutil
                shutil.copy2(strip_parquets[0], out_path)
            else:
                logger.info(
                    "fetch_spec %s year %d: merging %d strips → %s",
                    spec.id, year, len(strip_parquets), out_path.name,
                )
                merge_strips(strip_parquets, out_path)
            merged_paths.append(out_path)

        return year, merged_paths

    # Years themselves are still subject to max_extract_years concurrency cap
    year_results: dict[int, list[Path]] = {}
    with ThreadPoolExecutor(max_workers=max_extract_years) as year_pool:
        year_futs = {year_pool.submit(_process_year, yr): yr for yr in years}
        for fut in as_completed(year_futs):
            yr = year_futs[fut]
            try:
                _, paths = fut.result()
                year_results[yr] = paths
            except Exception as exc:
                logger.error("fetch_spec %s year %d failed: %s", spec.id, yr, exc, exc_info=exc)
                year_results[yr] = []

    return year_results
