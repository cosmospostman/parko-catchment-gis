"""proxy/_pipeline.py — Pure pipeline logic shared by the local fetch path and tests."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

from typing import Iterator

logger = logging.getLogger("proxy.pipeline")


from utils.fetch_spec import _system_memory_gb  # noqa: E402

# ---------------------------------------------------------------------------
# merge_scenes — streaming k-way merge of pre-sorted per-scene parquets
# ---------------------------------------------------------------------------

def merge_scenes(
    scene_paths: list[Path],
    s1_path: Path | None,
    out_path: Path,
    chunk_metadata: dict[str, str] | None = None,
) -> None:
    """K-way merge of pre-northing-sorted per-scene parquets (+ optional S1).

    Each scene parquet is written with row_group_size = strip_height_px (1024),
    so each row-group covers exactly one northing row of the strip grid.  Because
    every file is sorted by northing and the row-groups cover non-overlapping
    northing bands within each file, a min-heap keyed on each file's current
    row-group's minimum northing produces a correctly sorted output with no sort
    step and no I/O amplification.

    Peak RAM = n_files × strip_height_px × bytes_per_row ≈ 4 MB for 82 scenes.
    No sort, no DuckDB, no spill — pure sequential PyArrow reads and writes.

    S1 rows have many dates per northing; its row-groups may span multiple
    northing values.  S1 is appended after the S2 merge and the combined output
    is then re-sorted by (northing, date) only for the S1 rows — but in practice
    S1 is a small fraction of total rows (~1%) so this sort is cheap.  Actually,
    S1 is handled by appending S1 row-groups in northing order after S2 (S1 is
    also northing-sorted by _sort_s1_shards) and the combined output is already
    ordered: all S2 rows come first sorted by northing, then S1 rows sorted by
    northing, which is NOT the desired (northing, date) interleaving.

    Correct approach: treat S1 as another set of sorted runs in the heap.
    S1 row-groups cover many dates per northing band, so their min-northing keys
    interleave correctly with the S2 scene row-groups.
    """
    import heapq
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.parquet_utils import COMBINED_PIXEL_SCHEMA, _conform_table, sort_parquet_by_pixel

    all_paths: list[Path] = list(scene_paths)
    if s1_path is not None and s1_path.exists():
        all_paths.append(s1_path)

    if not all_paths:
        return

    n_files = len(all_paths)
    logger.info("merge_scenes: merging %d files → %s", n_files, out_path.name)

    tmp_path = out_path.with_suffix(".merge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)

    schema = COMBINED_PIXEL_SCHEMA
    if chunk_metadata:
        schema = schema.with_metadata({k.encode(): v.encode() for k, v in chunk_metadata.items()})
    writer = pq.ParquetWriter(
        str(tmp_path), schema,
        compression="zstd",
        write_statistics=False,
    )

    readers = [pq.ParquetFile(str(p)) for p in all_paths]
    cursors = [0] * n_files

    def _northing_of_first_row(tbl: pa.Table) -> int:
        # Rows are northing-sorted; the first row has the minimum northing.
        # Split only the first point_id — O(1), no full-column Arrow kernel call.
        return int(tbl.column("point_id")[0].as_py().rsplit("_", 1)[-1])

    def _read_next(file_idx: int) -> tuple[int, int, pa.Table] | None:
        pf  = readers[file_idx]
        rg  = cursors[file_idx]
        if rg >= pf.metadata.num_row_groups:
            return None
        tbl = _conform_table(pf.read_row_group(rg), schema)
        cursors[file_idx] = rg + 1
        return (_northing_of_first_row(tbl), file_idx, tbl)

    heap: list[tuple[int, int, pa.Table]] = []
    for idx in range(n_files):
        entry = _read_next(idx)
        if entry is not None:
            heapq.heappush(heap, entry)

    all_pids: set[str] = set()
    try:
        while heap:
            _key, file_idx, tbl = heapq.heappop(heap)
            writer.write_table(tbl)
            all_pids.update(tbl.column("point_id").to_pylist())
            nxt = _read_next(file_idx)
            if nxt is not None:
                heapq.heappush(heap, nxt)
    finally:
        writer.close()

    tmp_path.replace(out_path)

    # Pixel-sort while still on NVMe (out_path is in pipeline_tmp, before any
    # copy to external HDD).  Polars uses all available CPU threads.
    logger.info("merge_scenes: pixel-sorting %s ...", out_path.name)
    sorting_tmp = out_path.with_suffix(".sorting.parquet")
    sort_parquet_by_pixel(out_path, sorting_tmp, row_group_size=5_000_000)
    out_path.unlink()
    sorting_tmp.replace(out_path)

    # Write pixel count sidecar atomically so num_pixels() is instant on re-read.
    _sidecar_tmp = out_path.with_suffix(".pixel_count.tmp")
    _sidecar_tmp.write_text(str(len(all_pids)))
    _sidecar_tmp.replace(out_path.with_suffix(".pixel_count"))

    logger.info("merge_scenes done: %d input files → %s (%d pixels)", n_files, out_path.name, len(all_pids))


# ---------------------------------------------------------------------------
# Strip geometry helpers
# ---------------------------------------------------------------------------

def read_cog_transform(href: str) -> tuple[str, float, float, int, int]:
    """Read the UTM CRS, top-edge northing, left-edge easting, block height, and block width from a COG href.

    Opens only the GeoTIFF header via a range request (~2 kB).  No pixel data
    is read.  Returns (crs_epsg_str, y_top, x_left, block_height_px, block_width_px) where:
      y_top  = src.transform.f — UTM northing of pixel row 0 (top edge of raster)
      x_left = src.transform.c — UTM easting of pixel col 0 (left edge of raster)

    Used by compute_chunks to snap chunk boundaries to the COG block grid.
    """
    import rasterio
    with rasterio.open(href) as src:
        crs_str = src.crs.to_epsg()
        epsg = f"EPSG:{crs_str}" if crs_str else src.crs.to_string()
        y_top  = src.transform.f
        x_left = src.transform.c
        profile = src.profile
        block_height = profile.get("blockysize", 1024)
        block_width  = profile.get("blockxsize", 1024)
    return epsg, y_top, x_left, block_height, block_width


def compute_chunks(
    bbox_wgs84: list[float],
    chunk_height_px: int,
    chunk_width_px: int,
    polygon_geometry,
    cog_utm_crs: str | None = None,
    cog_y_top: float | None = None,
    cog_x_left: float | None = None,
) -> tuple[list[dict], dict]:
    """Divide bbox into 2D chunks of chunk_height_px × chunk_width_px pixels.

    Generates chunk metadata without materialising point coordinates — points
    are generated on demand via make_chunk_points() so only one chunk's worth
    of points is in memory at a time.

    When cog_utm_crs, cog_y_top, and cog_x_left are supplied, both X and Y
    boundaries are snapped to the COG's block grid (zero block over-fetch).

    Returns (chunks, chunks_meta) where each chunk dict has keys:
      chunk_row, chunk_col, bbox, y_lower, x_left_chunk
    and chunks are in row-major order: (0,0), (0,1), ..., (1,0), ...
    """
    import numpy as np
    from utils.pixel_collector import _utm_crs_for_bbox
    from pyproj import Transformer

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    utm_crs = cog_utm_crs or _utm_crs_for_bbox(bbox_wgs84)

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    r = 10.0  # S2 pixel spacing in metres
    x0, y0 = to_utm.transform(lon_min, lat_min)
    x1, y1 = to_utm.transform(lon_max, lat_max)

    x0_snap = math.floor(x0 / r) * r
    y0_snap = math.floor(y0 / r) * r

    block_h_m = chunk_height_px * r
    block_w_m = chunk_width_px  * r

    # --- Y boundaries (same snapping logic as the old compute_strips) ---
    if cog_utm_crs is not None and cog_y_top is not None:
        k_first_y = math.ceil((cog_y_top - y0_snap) / block_h_m) - 1
        first_y_lower = cog_y_top - (k_first_y + 1) * block_h_m
    else:
        first_y_lower = y0_snap

    y_lowers = []
    cur = first_y_lower
    y_top_snap = math.ceil((y1 - first_y_lower) / block_h_m) * block_h_m + first_y_lower
    while cur < y_top_snap:
        y_lowers.append(cur)
        cur += block_h_m

    cog_y_top_eff = cog_y_top if (cog_utm_crs is not None and cog_y_top is not None) else first_y_lower + len(y_lowers) * block_h_m

    # --- X boundaries (mirror of Y snapping, new dimension) ---
    if cog_utm_crs is not None and cog_x_left is not None:
        k_first_x = math.floor((x0_snap - cog_x_left) / block_w_m)
        first_x_left = cog_x_left + k_first_x * block_w_m
    else:
        first_x_left = x0_snap

    x_lefts = []
    cur = first_x_left
    x_right_snap = math.ceil((x1 - first_x_left) / block_w_m) * block_w_m + first_x_left
    while cur < x_right_snap:
        x_lefts.append(cur)
        cur += block_w_m

    cog_x_left_eff = cog_x_left if (cog_utm_crs is not None and cog_x_left is not None) else first_x_left

    chunks = []
    # Outer: reversed y_lowers → northernmost row first (chunk_row ascending)
    # Inner: x_lefts in order → westernmost col first (chunk_col ascending)
    # Produces row-major ordering: (0,0),(0,1),...,(1,0),...
    for y_lower in reversed(y_lowers):
        chunk_row = round((cog_y_top_eff - y_lower - block_h_m) / block_h_m)
        y_upper = y_lower + block_h_m
        ys = np.arange(y_lower, y_upper, r)
        ys = ys[(ys >= y0_snap) & (ys < y1)]
        if len(ys) == 0:
            continue

        for x_left_chunk in x_lefts:
            chunk_col = round((x_left_chunk - cog_x_left_eff) / block_w_m)
            x_right_chunk = x_left_chunk + block_w_m
            xs = np.arange(x_left_chunk, x_right_chunk, r)
            xs = xs[(xs >= x0_snap) & (xs < x1)]
            if len(xs) == 0:
                continue

            # Compute WGS84 bbox from 4 UTM corners (UTM→WGS84 is monotone over
            # a 10 km chunk, so corners give exact extrema — no full meshgrid needed).
            x_lo = float(xs[0]); x_hi = float(xs[-1])
            y_lo = float(ys[0]); y_hi = float(ys[-1])
            corners_lon, corners_lat = to_wgs.transform(
                [x_lo, x_hi, x_lo, x_hi],
                [y_lo, y_lo, y_hi, y_hi],
            )
            chunk_bbox = [
                float(min(corners_lon)), float(min(corners_lat)),
                float(max(corners_lon)), float(max(corners_lat)),
            ]

            if polygon_geometry is not None:
                from shapely.geometry import box as _box
                if not polygon_geometry.intersects(_box(*chunk_bbox)):
                    continue

            chunks.append({
                "chunk_row":    chunk_row,
                "chunk_col":    chunk_col,
                "bbox":         chunk_bbox,
                "y_lower":      y_lower,
                "x_left_chunk": x_left_chunk,
            })

    chunks_meta = {
        "utm_crs": utm_crs,
        "y0_snap": y0_snap, "y1": y1,
        "x0_snap": x0_snap, "x1": x1,
        "block_h_m": block_h_m, "block_w_m": block_w_m, "r": r,
        "first_y_lower": first_y_lower, "first_x_left": first_x_left,
        "point_id_prefix": "px",
    }
    return chunks, chunks_meta


def make_chunk_points(chunk: dict, meta: dict) -> list[tuple[str, float, float]]:
    """Generate points for one chunk on demand.

    Call just before the chunk is needed; discard (del) after use so only one
    chunk's points are in memory at a time.

    Point IDs are {prefix}_{xi:04d}_{yi:04d} where xi and yi are global
    easting and northing indices across the full tile grid.
    """
    import numpy as np
    from pyproj import Transformer

    utm_crs       = meta["utm_crs"]
    y0_snap       = meta["y0_snap"]
    y1            = meta["y1"]
    x0_snap       = meta["x0_snap"]
    x1            = meta["x1"]
    block_h_m     = meta["block_h_m"]
    block_w_m     = meta["block_w_m"]
    r             = meta["r"]
    first_y_lower = meta["first_y_lower"]
    first_x_left  = meta["first_x_left"]

    y_lower       = chunk["y_lower"]
    x_left_chunk  = chunk["x_left_chunk"]

    ys = np.arange(y_lower, y_lower + block_h_m, r)
    ys = ys[(ys >= y0_snap) & (ys < y1)]
    xs = np.arange(x_left_chunk, x_left_chunk + block_w_m, r)
    xs = xs[(xs >= x0_snap) & (xs < x1)]

    if len(ys) == 0 or len(xs) == 0:
        return []

    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    # indexing="xy" → y-major ravel: all x for ys[0], then ys[1], ...
    # Points therefore ravel in ascending northing order — required by merge_scenes.
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    lons_arr, lats_arr = to_wgs.transform(xx.ravel(), yy.ravel())
    lons_arr = np.asarray(lons_arr)
    lats_arr = np.asarray(lats_arr)

    # ii = local easting index (xs axis), jj = local northing index (ys axis).
    # Must use same arg order as the xx/yy meshgrid above so the ravel order matches.
    ii, jj = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="xy")
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()

    if len(lons_arr) == 0:
        return []

    # Compute global xi (easting) and yi (northing) indices.
    # Use xs[0]/ys[0] (actual first pixel coordinate after clipping to bbox) relative
    # to x0_snap/y0_snap (the global pixel origin) so that the first non-empty chunk
    # always starts at xi=0/yi=0 and subsequent chunks are offset by their true pixel
    # distance — not by their block-grid position, which overshoots when the first
    # block is partially clipped by the location bbox.
    xi_offset = round((xs[0] - x0_snap) / r)
    yi_offset = round((ys[0] - y0_snap) / r)
    pfx = meta.get("point_id_prefix", "px")
    pids = [
        f"{pfx}_{int(i + xi_offset):04d}_{int(j + yi_offset):04d}"
        for i, j in zip(ii_flat, jj_flat)
    ]
    return list(zip(pids, lons_arr.tolist(), lats_arr.tolist()))


def run_tile_pipeline_v2(
    tile_id: str,
    year: int,
    polygon_geometry,
    tmp: Path,
    cloud_max: int = 80,
    apply_nbar: bool = True,
    chunk_height_px: int = 1024,
    chunk_width_px: int = 1024,
    max_concurrent: int = 64,
    n_workers: int | None = None,
    resume_from_chunk: tuple[int, int] = (0, 0),
    skip_chunks: set[tuple[int, int]] | None = None,
    items=None,
    calibration_out: Path | None = None,
    point_id_prefix: str = "px",
) -> Iterator[tuple[int, int, Path]]:
    """Two-pool network→disk / disk→extract pipeline for memory-constrained machines.

    Pool A (network→disk): fetch_patches_to_tiff() writes one GeoTIFF per
    (item, band) to disk immediately, dereferencing the array after each write.
    Peak RAM during fetch = O(one patch array) regardless of item count.

    Pool B (disk→extract→parquet): _extract_item_from_tiffs() opens on-disk
    tifs, samples pixel values, and writes per-scene parquets.
    Peak RAM during extract = O(n_points × n_bands × 4 bytes) per worker (~MB).

    Stage pipeline (declared via StageSpec; concurrency × ram_gb = budget):

        fetch_tiffs    ×2  0.40 GB  — Pool A: network → GeoTIFFs on disk
        extract_scenes ×1  4.00 GB  — Pool B: tiffs → per-scene parquets
        collect_s1     ×1  0.50 GB  — async S1 fetch + extract
        merge_scenes   ×1  0.05 GB  — k-way heap merge → sorted chunk parquet

    Yields (chunk_row, chunk_col, sorted_chunk_parquet_path) for each completed chunk.
    Chunks are in row-major order: (0,0), (0,1), ..., (1,0), ...
    """
    import asyncio
    import shutil
    import os
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed
    from dataclasses import dataclass, field

    from utils.stac import search_sentinel2
    from utils.pipeline_types import Pipeline, StageSpec
    from utils.pixel_collector import (
        collect, STAC_ENDPOINT, S2_COLLECTION, FETCH_BANDS, BAND_ALIAS,
        _extract_item_from_tiffs, _TILE_ID_RE,
        _band_to_uint16, BANDS,
    )
    from utils.s1_collector import collect_s1_for_tile
    from analysis.constants import SCL_BAND, AOT_BAND, SCL_CLEAR_VALUES
    from utils.pipeline import setup_gdal_env

    # Must be called before any ThreadPoolExecutor so worker threads inherit the env.
    setup_gdal_env()

    extract_workers = int(os.environ.get("EXTRACT_WORKERS", "2"))
    _mem_gb = _system_memory_gb()
    _default_prefetch = 1 if _mem_gb <= 20 else 2
    prefetch_depth = int(os.environ.get("PREFETCH_DEPTH", str(_default_prefetch)))

    bbox_wgs84 = list(polygon_geometry.bounds)
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"

    if items is None:
        logger.info("[v2 tile %s %d] STAC search ...", tile_id, year)
        items = search_sentinel2(
            bbox=bbox_wgs84,
            start=start_date,
            end=end_date,
            cloud_cover_max=cloud_max,
            endpoint=STAC_ENDPOINT,
            collection=S2_COLLECTION,
            mgrs_tile=tile_id,
        )
    logger.info("[v2 tile %s %d] %d STAC items", tile_id, year, len(items))

    if not items:
        return

    cog_utm_crs: str | None  = None
    cog_y_top:   float | None = None
    cog_x_left:  float | None = None
    cog_block_height: int = chunk_height_px
    cog_block_width:  int = chunk_width_px
    for item in items:
        href_obj = item.assets.get("red") or item.assets.get("B04")
        if href_obj is not None:
            try:
                cog_utm_crs, cog_y_top, cog_x_left, cog_block_height, cog_block_width = read_cog_transform(href_obj.href)
                logger.info("[v2 tile %s %d] COG origin: crs=%s y_top=%.1f x_left=%.1f block=%dx%d px",
                            tile_id, year, cog_utm_crs, cog_y_top, cog_x_left,
                            cog_block_height, cog_block_width)
            except Exception as exc:
                logger.warning("[v2 tile %s %d] Could not read COG transform (%s) — using geographic fallback",
                               tile_id, year, exc)
            break

    chunks, chunks_meta = compute_chunks(
        bbox_wgs84, cog_block_height, cog_block_width, polygon_geometry,
        cog_utm_crs=cog_utm_crs, cog_y_top=cog_y_top, cog_x_left=cog_x_left,
    )
    chunks_meta["point_id_prefix"] = point_id_prefix
    logger.info("[v2 tile %s %d] %d chunks (%dx%d px each)",
                tile_id, year, len(chunks), cog_block_height, cog_block_width)

    if not chunks:
        return

    _skip = skip_chunks or set()
    active_chunks = [
        c for c in chunks
        if (c["chunk_row"], c["chunk_col"]) >= resume_from_chunk
        and (c["chunk_row"], c["chunk_col"]) not in _skip
    ]
    for c in chunks:
        key = (c["chunk_row"], c["chunk_col"])
        if key < resume_from_chunk:
            logger.info("[v2 tile %s %d] [chunk %02d_%02d] skipping (resume_from_chunk=%s)",
                        tile_id, year, c["chunk_row"], c["chunk_col"], resume_from_chunk)
        elif key in _skip:
            logger.info("[v2 tile %s %d] [chunk %02d_%02d] skipping (already exists)",
                        tile_id, year, c["chunk_row"], c["chunk_col"])

    if not active_chunks:
        return

    from utils.parquet_utils import _optimise_schema, _WRITE_OPTS
    import pyarrow.parquet as pq
    import numpy as np

    col_order = (
        ["point_id", "lon", "lat", "date", "item_id", "tile_id"]
        + list(BANDS)
        + ["scl_purity", "scl", "aot", "view_zenith", "sun_zenith"]
    )

    # --- Per-chunk work item travelling through the pipeline ------------------

    @dataclass
    class _ChunkWork:
        chunk:       dict
        pts:         list          = field(default_factory=list)
        tiff_dir:    Path | None   = None
        scene_paths: list[Path]    = field(default_factory=list)
        s1_path:     Path | None   = None

    # --- Stage functions (capture only run-scoped constants) ------------------

    def _stage_fetch_tiffs(work: _ChunkWork) -> _ChunkWork:
        """Pool A: fetch all item×band GeoTIFFs to disk for one chunk."""
        crow = work.chunk["chunk_row"]
        ccol = work.chunk["chunk_col"]
        tiff_dir = tmp / f"chunk_{crow:02d}_{ccol:02d}_tiffs"
        tiff_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[v2 tile %s %d] [chunk %02d_%02d] fetch_tiffs → %s",
                    tile_id, year, crow, ccol, tiff_dir.name)

        async def _fetch_async() -> None:
            from utils.fetch import fetch_patches_to_tiff
            await fetch_patches_to_tiff(
                items=items,
                bands=FETCH_BANDS,
                bbox_wgs84=work.chunk["bbox"],
                out_dir=tiff_dir,
                max_concurrent=max_concurrent,
                band_alias=BAND_ALIAS,
            )

        asyncio.run(_fetch_async())
        work.tiff_dir = tiff_dir
        return work

    def _stage_extract_scenes(work: _ChunkWork) -> _ChunkWork | None:
        """Pool B: sample pixel values from tiffs → per-scene parquets."""
        crow      = work.chunk["chunk_row"]
        ccol      = work.chunk["chunk_col"]
        tiff_dir  = work.tiff_dir
        chunk_pts = work.pts
        rg_size   = max(1, len(chunk_pts) // chunk_height_px)

        scene_dir = tmp / f"chunk_{crow:02d}_{ccol:02d}_scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)

        point_ids = [pid      for pid, _, _   in chunk_pts]
        lons      = np.array([lon for _, lon, _ in chunk_pts], dtype=np.float64)
        lats      = np.array([lat for _, _, lat in chunk_pts], dtype=np.float64)

        _utm = cog_utm_crs or "EPSG:32755"
        from pyproj import Transformer as _Transformer
        _chunk_utm_xy = _Transformer.from_crs(
            "EPSG:4326", _utm, always_xy=True
        ).transform(lons, lats)

        if apply_nbar:
            from utils.granule_angles import prefetch_granule_xmls
            prefetch_granule_xmls(items, max_workers=8)

        n_items = len(items)
        logger.info("[v2 tile %s %d] [chunk %02d_%02d] extract_scenes (%d pts, %d items)",
                    tile_id, year, crow, ccol, len(chunk_pts), n_items)

        def _extract_one(item_idx: int, item) -> Path | None:
            scene_id      = item.id
            out_path      = scene_dir / f"scene_{item_idx:04d}.parquet"
            item_tiff_dir = tiff_dir / scene_id

            if out_path.exists() and out_path.stat().st_size > 0:
                try:
                    pq.ParquetFile(out_path).metadata
                    return out_path
                except Exception:
                    out_path.unlink(missing_ok=True)

            if not item_tiff_dir.exists():
                return None

            _prof_path = Path(os.environ.get("PROFILE_SCENE", ""))
            if _prof_path.name and not _prof_path.exists():
                import cProfile, pstats, io as _io
                _pr = cProfile.Profile()
                try:
                    df = _pr.runcall(
                        _extract_item_from_tiffs,
                        item, item_tiff_dir, point_ids, lons, lats,
                        apply_nbar=apply_nbar,
                        utm_crs=_utm,
                        utm_xy=_chunk_utm_xy,
                    )
                finally:
                    _s = _io.StringIO()
                    pstats.Stats(_pr, stream=_s).sort_stats("cumulative").print_stats(30)
                    _prof_path.write_text(_s.getvalue())
                    logger.info("Profile written to %s", _prof_path)
            else:
                df = _extract_item_from_tiffs(
                    item, item_tiff_dir, point_ids, lons, lats,
                    apply_nbar=apply_nbar,
                    utm_crs=_utm,
                    utm_xy=_chunk_utm_xy,
                )
            shutil.rmtree(item_tiff_dir, ignore_errors=True)

            if df is None or len(df) == 0:
                logger.info("[v2] [chunk %02d_%02d] scene %d/%d %s — no clear pixels",
                            crow, ccol, item_idx + 1, n_items, scene_id)
                return None

            tbl = _optimise_schema(df.select(col_order).to_arrow())
            del df

            tmp_path = out_path.with_suffix(".tmp.parquet")
            tmp_path.unlink(missing_ok=True)
            writer = pq.ParquetWriter(str(tmp_path), tbl.schema, compression="none", write_statistics=False)
            for off in range(0, max(len(tbl), 1), rg_size):
                writer.write_table(tbl.slice(off, rg_size))
            writer.close()
            tmp_path.replace(out_path)

            logger.info("[v2] [chunk %02d_%02d] scene %d/%d %s — %d rows",
                        crow, ccol, item_idx + 1, n_items, scene_id, len(tbl))
            return out_path

        scene_paths: list[Path] = []
        with _TPE(max_workers=extract_workers) as pool:
            futs = {pool.submit(_extract_one, idx, item): idx for idx, item in enumerate(items)}
            for fut in _as_completed(futs):
                result = fut.result()
                if result is not None:
                    scene_paths.append(result)

        del _chunk_utm_xy, lons, lats, point_ids
        import gc; gc.collect()

        shutil.rmtree(tiff_dir, ignore_errors=True)
        work.tiff_dir    = None
        work.scene_paths = scene_paths

        if not scene_paths:
            logger.info("[v2 tile %s %d] [chunk %02d_%02d] no scene data — skipping",
                        tile_id, year, crow, ccol)
            return None  # drops this chunk from the pipeline

        return work

    def _stage_collect_s1(work: _ChunkWork) -> _ChunkWork:
        """Async S1 fetch + extract for one chunk."""
        crow = work.chunk["chunk_row"]
        ccol = work.chunk["chunk_col"]
        scene_dir = tmp / f"chunk_{crow:02d}_{ccol:02d}_scenes"
        work.s1_path = collect_s1_for_tile(
            s2_path=None,
            bbox_wgs84=work.chunk["bbox"],
            start=start_date,
            end=end_date,
            out_path=scene_dir / "s1_chunk.parquet",
            cache_dir=scene_dir / "s1_cache",
            max_concurrent=max_concurrent,
            points=work.pts,
        )
        # pts held until S1 is done; free now before merge.
        del work.pts
        import gc; gc.collect()
        return work

    def _stage_merge(work: _ChunkWork) -> _ChunkWork:
        """K-way heap merge of scene parquets + S1 → sorted chunk parquet."""
        crow = work.chunk["chunk_row"]
        ccol = work.chunk["chunk_col"]
        scene_dir = tmp / f"chunk_{crow:02d}_{ccol:02d}_scenes"
        chunk_out = tmp / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
        _meta: dict[str, str] = {
            "chunk_row": str(crow),
            "chunk_col": str(ccol),
        }
        if cog_utm_crs is not None:
            _meta["cog_utm_crs"] = cog_utm_crs
        if cog_y_top is not None:
            _meta["cog_y_top"] = str(cog_y_top)
        if cog_x_left is not None:
            _meta["cog_x_left"] = str(cog_x_left)
        merge_scenes(work.scene_paths, work.s1_path, chunk_out, chunk_metadata=_meta)
        shutil.rmtree(scene_dir, ignore_errors=True)
        logger.info("[v2 tile %s %d] [chunk %02d_%02d] ready → %s",
                    tile_id, year, crow, ccol, chunk_out.name)
        work.scene_paths = []
        work.s1_path     = None
        return work

    # --- Pipeline declaration -------------------------------------------------
    #
    # Memory budget per stage (concurrency × ram_gb):
    #   fetch_tiffs    ×prefetch_depth  × 0.40 GB
    #   extract_scenes ×1               × 4.00 GB
    #   collect_s1     ×1               × 0.50 GB
    #   merge_scenes   ×1               × 0.05 GB
    #
    # At prefetch_depth=1: 0.40 + 4.00 + 0.50 + 0.05 = 4.95 GB declared.
    # At prefetch_depth=2: 0.80 + 4.00 + 0.50 + 0.05 = 5.35 GB declared.

    fetch_pipeline = Pipeline(
        [
            StageSpec("fetch_tiffs",    fn=_stage_fetch_tiffs,    concurrency=prefetch_depth, ram_gb=0.40),
            StageSpec("extract_scenes", fn=_stage_extract_scenes, concurrency=1,              ram_gb=4.00),
            StageSpec("collect_s1",     fn=_stage_collect_s1,     concurrency=1,              ram_gb=0.50),
            StageSpec("merge_scenes",   fn=_stage_merge,          concurrency=1,              ram_gb=0.05),
        ],
        ram_budget_gb=_mem_gb * 0.6,
        name=f"fetch_tile_{tile_id}_{year}",
    )

    def _chunk_inputs() -> Iterator[_ChunkWork]:
        for chunk in active_chunks:
            pts = make_chunk_points(chunk, chunks_meta)
            yield _ChunkWork(chunk=chunk, pts=pts)

    for work in fetch_pipeline.run(_chunk_inputs()):
        crow = work.chunk["chunk_row"]
        ccol = work.chunk["chunk_col"]
        chunk_out = tmp / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
        yield crow, ccol, chunk_out
