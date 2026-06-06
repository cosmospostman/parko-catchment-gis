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
    on_phase: "Callable[[str], None] | None" = None,
) -> None:
    """Merge per-scene parquets into one pixel-sorted chunk parquet.

    Streams through all input files in northing-band passes so peak RAM is
    O(n_files × _NORTHING_BAND) rather than O(total_rows).  For a 1024×1024
    chunk with 82 scenes the old in-memory sort required ~34 GB; this approach
    uses ~200 MB regardless of scene count.

    Input scene parquets are expected to be sorted by northing (ascending yi
    encoded in point_id suffix) — make_chunk_points guarantees this ordering.
    S1 rows are merged in the final pass along with any remaining S2 rows.

    on_phase: optional callback called with a short human-readable string at the
        start of each phase ("reading N files", "sorting NM rows", "writing").
        Used by the progress display to show sub-step detail on the merge line.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import polars as pl
    from utils.parquet_utils import COMBINED_PIXEL_SCHEMA, _conform_table

    # Northings processed per band pass.  1024 northing rows × ~82 scenes ×
    # ~1 clear pixel/point/scene × ~120 bytes ≈ 10 MB per pass.
    _NORTHING_BAND = 1024

    import time as _time_mod
    _t_start = _time_mod.perf_counter()

    def _phase(msg: str) -> None:
        if on_phase is not None:
            on_phase(msg)

    s2_paths: list[Path] = list(scene_paths)
    s1_paths: list[Path] = [s1_path] if (s1_path is not None and s1_path.exists()) else []
    all_paths: list[Path] = s2_paths + s1_paths

    if not all_paths:
        return

    n_files = len(all_paths)
    _phase(f"reading {n_files} files")
    logger.info("merge_scenes: streaming %d files → %s", n_files, out_path.name)

    schema = COMBINED_PIXEL_SCHEMA
    if chunk_metadata:
        schema = schema.with_metadata({k.encode(): v.encode() for k, v in chunk_metadata.items()})

    # --- Pass 1: collect total row count and northing range from file metadata ---
    n_rows_total = 0
    for p in all_paths:
        pf = pq.ParquetFile(str(p))
        n_rows_total += pf.metadata.num_rows

    _t_pass1 = _time_mod.perf_counter()
    logger.info("merge_scenes: pass1 (metadata) %.1fs", _t_pass1 - _t_start)

    _phase(f"sorting {n_rows_total // 1_000_000}M rows")
    logger.info("merge_scenes: %d total rows across %d files", n_rows_total, n_files)

    # --- Pass 2: stream all files, accumulate by northing band, sort+write ---
    # Strategy: read one row group at a time from each file in round-robin order.
    # Buffer rows in a per-northing-band accumulator.  When we've seen all row
    # groups that could contribute to northing band [y0, y0+_NORTHING_BAND), flush
    # that band (sort + write) and advance the band window.
    #
    # Because each input file is sorted by northing ascending, once a file's
    # current row group has min_northing > band_top we know the band is complete.
    # We use a simpler approach: read all files completely into a list of small
    # band-keyed DataFrames, sorted per-file, then concat+sort per band.
    # This avoids complex iterator state while still bounding RAM per band.

    # Open all files.
    pfs = [pq.ParquetFile(str(p)) for p in all_paths]
    n_rgs = [pf.metadata.num_row_groups for pf in pfs]

    def _yi_from_pid(pid: str) -> int:
        """Extract the global northing index from a point_id like 'px_0012_0034'."""
        return int(pid.rsplit("_", 1)[-1])

    # Determine the global northing range by sampling the first/last point_id
    # from each file (cheap — reads one row from first and last row groups).
    yi_min_global = math.inf
    yi_max_global = -math.inf
    for fi, pf in enumerate(pfs):
        if n_rgs[fi] == 0:
            continue
        rg0 = pf.read_row_group(0, columns=["point_id"])
        rg_last = pf.read_row_group(n_rgs[fi] - 1, columns=["point_id"])
        pids_first = rg0.column("point_id").to_pylist()
        pids_last  = rg_last.column("point_id").to_pylist()
        if pids_first:
            yi_min_global = min(yi_min_global, _yi_from_pid(pids_first[0]))
        if pids_last:
            yi_max_global = max(yi_max_global, _yi_from_pid(pids_last[-1]))

    if math.isinf(yi_min_global):
        return  # all files empty

    yi_min_global = int(yi_min_global)
    yi_max_global = int(yi_max_global)

    # Read in large batches (100k rows) regardless of underlying rg size.
    # Scene parquets are written with rg_size=1024 (1 rg per northing row), so
    # reading them row-group by row-group means 130k+ PyArrow calls per chunk.
    # iter_batches amortises that to ~10 calls per file with no algorithm change.
    _BATCH_SIZE = 100_000

    # Build band boundaries: [yi_min, yi_min+BAND), [yi_min+BAND, ...), ...
    band_starts = list(range(
        (yi_min_global // _NORTHING_BAND) * _NORTHING_BAND,
        yi_max_global + _NORTHING_BAND,
        _NORTHING_BAND,
    ))

    tmp_path = out_path.with_suffix(".merge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)
    dict_cols = {"point_id", "item_id", "tile_id"}
    writer = pq.ParquetWriter(
        str(tmp_path), schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=[c for c in schema.names if c in dict_cols],
        write_statistics=["lon", "lat"],
    )

    n_pixels_total = 0
    rg_size_out = 5_000_000

    # Per-file batch-iterator state (replaces rg_cursors / read_row_group calls).
    # iter_batches yields Arrow RecordBatches of up to _BATCH_SIZE rows regardless
    # of the underlying row-group layout — ~10 calls/file instead of ~1024.
    batch_iters = [pf.iter_batches(batch_size=_BATCH_SIZE) for pf in pfs]
    rg_buffers: list[pl.DataFrame | None] = [None] * n_files  # current buffered batch (with _yi)

    # Output accumulator for current band
    band_frames: list[pl.DataFrame] = []

    def _flush_band(frames: list[pl.DataFrame]) -> tuple[pl.DataFrame, int]:
        """Sort and conform a list of frames from one northing band."""
        if not frames:
            return pl.DataFrame(), 0
        combined_df = pl.concat(frames, how="diagonal_relaxed")
        _has_scl = "scl_purity" in combined_df.columns
        sort_cols = ["point_id", "date", *(["scl_purity"] if _has_scl else [])]
        sort_desc  = [False,      False,  *([ True        ] if _has_scl else [])]
        sorted_df = combined_df.sort(sort_cols, descending=sort_desc)
        n_px = sorted_df["point_id"].n_unique()
        return sorted_df, n_px

    write_buffer: pl.DataFrame | None = None
    write_buffer_rows = 0

    def _flush_write_buffer(final: bool = False) -> None:
        nonlocal write_buffer, write_buffer_rows
        if write_buffer is None or len(write_buffer) == 0:
            return
        if not final and write_buffer_rows < rg_size_out:
            return
        tbl = _conform_table(write_buffer.to_arrow(), schema)
        for off in range(0, max(len(tbl), 1), rg_size_out):
            writer.write_table(tbl.slice(off, rg_size_out))
        write_buffer = None
        write_buffer_rows = 0

    def _append_to_write_buffer(df: pl.DataFrame) -> None:
        nonlocal write_buffer, write_buffer_rows
        if len(df) == 0:
            return
        if write_buffer is None:
            write_buffer = df
        else:
            write_buffer = pl.concat([write_buffer, df], how="diagonal_relaxed")
        write_buffer_rows += len(df)
        if write_buffer_rows >= rg_size_out:
            _flush_write_buffer(final=False)

    # Stream through all files band by band.
    # For each band [band_start, band_start+_NORTHING_BAND):
    #   - for each file, read row groups until the row group's min yi >= band_end
    #   - collect rows that fall in [band_start, band_end) from each file
    #   - flush band: sort + write
    #
    # Since files are sorted by yi ascending, once a file's next rg has min_yi
    # >= band_end, we stop reading that file for this band.

    def _load_rg(fi: int) -> pl.DataFrame | None:
        """Pull the next batch from file fi's iterator, pre-computing _yi, or return None."""
        try:
            batch = next(batch_iters[fi])
        except StopIteration:
            return None
        df = pl.from_arrow(batch)
        return df.with_columns(
            pl.col("point_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("_yi")
        )

    # Eagerly load first batch per file into buffer (_yi pre-computed).
    _t_eager0 = _time_mod.perf_counter()
    for fi in range(n_files):
        rg_buffers[fi] = _load_rg(fi)
    logger.info("merge_scenes: eager-load first batches %.1fs", _time_mod.perf_counter() - _t_eager0)

    _t_band_read = 0.0
    _t_band_sort = 0.0
    _t_band_write = 0.0
    _n_load_rg = 0

    _orig_load_rg = _load_rg
    def _load_rg(fi: int):  # type: ignore[no-redef]
        nonlocal _n_load_rg
        _n_load_rg += 1
        return _orig_load_rg(fi)

    for band_start in band_starts:
        band_end = band_start + _NORTHING_BAND
        band_frames = []

        _tb0 = _time_mod.perf_counter()
        for fi in range(n_files):
            file_frames: list[pl.DataFrame] = []

            while True:
                buf = rg_buffers[fi]
                if buf is None or len(buf) == 0:
                    break

                # buf already has _yi column — split into in-band and after-band rows.
                in_band    = buf.filter((pl.col("_yi") >= band_start) & (pl.col("_yi") < band_end))
                after_band = buf.filter( pl.col("_yi") >= band_end)

                if len(in_band) > 0:
                    file_frames.append(in_band.drop("_yi"))

                if len(after_band) > 0:
                    # Keep remainder (yi >= band_end) for future bands.
                    rg_buffers[fi] = after_band
                    break
                else:
                    # Current rg fully consumed — load next.
                    rg_buffers[fi] = _load_rg(fi)
                    if rg_buffers[fi] is not None and rg_buffers[fi]["_yi"].min() >= band_end:
                        # Entire new batch is past this band — leave it buffered and stop.
                        break
                    # Loop again to split the newly loaded rg.

            if file_frames:
                band_frames.append(pl.concat(file_frames, how="diagonal_relaxed"))
        _t_band_read += _time_mod.perf_counter() - _tb0

        if not band_frames:
            continue

        _ts0 = _time_mod.perf_counter()
        sorted_band, n_px = _flush_band(band_frames)
        _t_band_sort += _time_mod.perf_counter() - _ts0

        n_pixels_total += n_px
        _tw0 = _time_mod.perf_counter()
        _append_to_write_buffer(sorted_band)
        _t_band_write += _time_mod.perf_counter() - _tw0
        del band_frames, sorted_band

    _tw0 = _time_mod.perf_counter()
    _flush_write_buffer(final=True)
    writer.close()
    _t_band_write += _time_mod.perf_counter() - _tw0

    logger.info(
        "merge_scenes timing: read=%.1fs  sort=%.1fs  write=%.1fs  load_rg_calls=%d",
        _t_band_read, _t_band_sort, _t_band_write, _n_load_rg,
    )

    tmp_path.replace(out_path)

    _phase(f"writing {n_pixels_total // 1000}k pixels")
    logger.info("merge_scenes: writing %s (%d pixels) ...", out_path.name, n_pixels_total)

    # Write pixel count sidecar atomically so num_pixels() is instant on re-read.
    _sidecar_tmp = out_path.with_suffix(".pixel_count.tmp")
    _sidecar_tmp.write_text(str(n_pixels_total))
    _sidecar_tmp.replace(out_path.with_suffix(".pixel_count"))

    logger.info("merge_scenes done: %d input files → %s (%d pixels)", n_files, out_path.name, n_pixels_total)


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
    years: list[int] | int,
    polygon_geometry,
    tmp: Path,
    cloud_max: int = 80,
    apply_nbar: bool = True,
    chunk_height_px: int = 1024,
    chunk_width_px: int = 1024,
    max_concurrent: int = 64,
    n_workers: int | None = None,
    resume_from_chunk: tuple[int, int] = (0, 0),
    skip_chunks: set[tuple[int, int, int]] | None = None,
    items=None,
    calibration_out: Path | None = None,
    point_id_prefix: str = "px",
    log_dir: Path | None = None,
    progress=None,   # TileProgress | None
    grid_cache: Path | None = None,  # path to write/read _grid.json
) -> Iterator[tuple[int, int, int, Path]]:
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

    The fundamental unit of work is (chunk, year).  All (chunk, year) pairs for the
    tile are queued together so the fetch stage is kept busy while extract processes
    a previous chunk.

    Yields (chunk_row, chunk_col, year, sorted_chunk_parquet_path) for each completed chunk.
    Chunks within each year are in row-major order: (0,0), (0,1), ..., (1,0), ...
    """
    import asyncio
    import shutil
    import os
    import time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed
    from dataclasses import dataclass, field

    from utils.chunk_log import ChunkLogger, get_chunk_logger

    from utils.stac import search_sentinel2
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

    # Accept a single year for backward compatibility.
    _years: list[int] = [years] if isinstance(years, int) else list(years)

    extract_workers = int(os.environ.get("EXTRACT_WORKERS", "2"))
    _mem_gb = _system_memory_gb()
    _default_prefetch = 1 if _mem_gb <= 20 else 2
    prefetch_depth = int(os.environ.get("PREFETCH_DEPTH", str(_default_prefetch)))

    bbox_wgs84 = list(polygon_geometry.bounds)

    # --- STAC search — lazy per year ------------------------------------------
    # items_by_year is populated on demand: year[0] is searched upfront only
    # when the COG grid is not yet cached (grid resolution needs at least one
    # item to read the COG transform).  All other years are searched just before
    # their first chunk is fed into the pipeline.
    #
    # items_by_year may be pre-supplied (training pipeline) only when years is a
    # single int and the caller passes items=<list>.  In that case wrap it.
    if items is not None and isinstance(years, int):
        items_by_year: dict[int, list] = {years: items}
        _years_pending: list[int] = []
    else:
        items_by_year = {}
        _years_pending = list(_years)

    def _search_year(yr: int) -> None:
        """Search STAC for *yr* and insert into items_by_year (or drop if empty)."""
        if progress is not None:
            progress.set_status(f"STAC search {yr}…")
        logger.info("[v2 tile %s %d] STAC search ...", tile_id, yr)
        yr_items = search_sentinel2(
            bbox=bbox_wgs84,
            start=f"{yr}-01-01",
            end=f"{yr}-12-31",
            cloud_cover_max=cloud_max,
            endpoint=STAC_ENDPOINT,
            collection=S2_COLLECTION,
            mgrs_tile=tile_id,
        )
        logger.info("[v2 tile %s %d] %d STAC items", tile_id, yr, len(yr_items))
        if yr_items:
            items_by_year[yr] = yr_items
        if progress is not None and not yr_items:
            progress.set_status("")

    # Search year[0] upfront only if needed for COG grid resolution.
    grid_already_cached = grid_cache is not None and grid_cache.exists()
    if _years_pending and not grid_already_cached:
        _search_year(_years_pending.pop(0))

    if not items_by_year and not _years_pending:
        return

    # --- Resolve COG transform for chunk-grid snapping ------------------------
    # Order of preference:
    #   1. Cached _grid.json written by a previous run (grid_cache path)
    #   2. Live read from first STAC item's COG header
    #   3. Geographic fallback (no snapping)
    import json as _json

    cog_utm_crs: str | None  = None
    cog_y_top:   float | None = None
    cog_x_left:  float | None = None
    cog_block_height: int = chunk_height_px
    cog_block_width:  int = chunk_width_px

    if grid_cache is not None and grid_cache.exists():
        try:
            _g = _json.loads(grid_cache.read_text())
            cog_utm_crs    = _g["crs"]
            cog_y_top      = _g["y_top"]
            cog_x_left     = _g["x_left"]
            cog_block_height = _g["block_height"]
            cog_block_width  = _g["block_width"]
            logger.info("[v2 tile %s] COG origin loaded from cache: crs=%s y_top=%.1f x_left=%.1f block=%dx%d px",
                        tile_id, cog_utm_crs, cog_y_top, cog_x_left, cog_block_height, cog_block_width)
        except Exception as exc:
            logger.warning("[v2 tile %s] Could not read grid cache (%s) — will fetch from COG", tile_id, exc)
            cog_utm_crs = cog_y_top = cog_x_left = None
            cog_block_height = chunk_height_px
            cog_block_width  = chunk_width_px

    if cog_utm_crs is None:
        if progress is not None and items_by_year:
            progress.set_status("reading COG grid…")
        for _yr_items in items_by_year.values():
            for item in _yr_items:
                href_obj = item.assets.get("red") or item.assets.get("B04")
                if href_obj is not None:
                    try:
                        cog_utm_crs, cog_y_top, cog_x_left, cog_block_height, cog_block_width = read_cog_transform(href_obj.href)
                        logger.info("[v2 tile %s] COG origin: crs=%s y_top=%.1f x_left=%.1f block=%dx%d px",
                                    tile_id, cog_utm_crs, cog_y_top, cog_x_left,
                                    cog_block_height, cog_block_width)
                    except Exception as exc:
                        logger.warning("[v2 tile %s] Could not read COG transform (%s) — using geographic fallback",
                                       tile_id, exc)
                    break
            else:
                continue
            break

    if cog_utm_crs is not None and grid_cache is not None and not grid_cache.exists():
        try:
            grid_cache.parent.mkdir(parents=True, exist_ok=True)
            grid_cache.write_text(_json.dumps({
                "crs": cog_utm_crs, "y_top": cog_y_top, "x_left": cog_x_left,
                "block_height": cog_block_height, "block_width": cog_block_width,
            }))
            logger.info("[v2 tile %s] COG origin cached to %s", tile_id, grid_cache)
        except Exception as exc:
            logger.warning("[v2 tile %s] Could not write grid cache (%s)", tile_id, exc)

    chunks, chunks_meta = compute_chunks(
        bbox_wgs84, cog_block_height, cog_block_width, polygon_geometry,
        cog_utm_crs=cog_utm_crs, cog_y_top=cog_y_top, cog_x_left=cog_x_left,
    )
    chunks_meta["point_id_prefix"] = point_id_prefix
    logger.info("[v2 tile %s] %d chunks (%dx%d px each) × %d years",
                tile_id, len(chunks), cog_block_height, cog_block_width, len(items_by_year))

    if not chunks:
        return

    _skip: set[tuple[int, int, int]] = skip_chunks or set()

    # Search remaining years in parallel now that the chunk grid is known.
    # _chunk_inputs() will find them in items_by_year and skip the lazy search.
    if _years_pending:
        if progress is not None:
            progress.set_status(f"STAC search ({len(_years_pending)} years)…")
        from concurrent.futures import ThreadPoolExecutor as _SearchTPE, as_completed as _search_ac
        _pending_copy = list(_years_pending)
        _years_pending.clear()
        with _SearchTPE(max_workers=len(_pending_copy)) as _pool:
            _futs = {_pool.submit(_search_year, yr): yr for yr in _pending_copy}
            for _f in _search_ac(_futs):
                _f.result()  # re-raise any search exception
        if progress is not None:
            progress.set_status("")

    # Inform the progress display of the chunk total now that the grid is known.
    for yr in _years:
        if yr not in items_by_year:
            continue
        n_skip_yr = sum(1 for (r, c, y) in _skip if y == yr)
        n_fetch_yr = len(chunks) - n_skip_yr
        logger.info("[v2 tile %s %d] %d chunks total: %d to fetch, %d already done",
                    tile_id, yr, len(chunks), n_fetch_yr, n_skip_yr)
        if progress is not None:
            progress.set_total(yr, len(chunks), already_done=n_skip_yr)

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
        chunk:        dict
        year:         int
        items:        list
        pts:          list          = field(default_factory=list)
        tiff_dir:     Path | None   = None
        scene_paths:  list[Path]    = field(default_factory=list)
        s1_path:      Path | None   = None
        s1_cache_dir: Path | None   = None  # populated by _stage_fetch_s1; consumed by _stage_extract_s1
        clog:         object        = None  # ChunkLogger | None (avoid import cycle in type)

    # --- Stage functions (use work.year / work.items — no captured year) ------

    def _stage_fetch_tiffs(work: _ChunkWork) -> _ChunkWork:
        """Pool A: fetch all item×band GeoTIFFs to disk for one chunk.

        If work.tiff_dir is already set (surviving tiff dir from a prior interrupted
        run), the network fetch is skipped entirely and the work item passes straight
        through to extract.
        """
        if work.clog:
            work.clog.activate()
        crow = work.chunk["chunk_row"]
        ccol = work.chunk["chunk_col"]

        if work.scene_paths:
            # Scenes already extracted in a prior run — skip fetch entirely.
            logger.info("[v2 tile %s %d] [chunk %02d_%02d] fetch_tiffs — skipped (scenes cached)",
                        tile_id, work.year, crow, ccol)
            if work.clog:
                work.clog.info("fetch_tiffs: skipped — %d scene parquets already present",
                               len(work.scene_paths))
            return work

        if work.tiff_dir is not None:
            # Tiffs already on disk from a prior run — skip fetch.
            logger.info("[v2 tile %s %d] [chunk %02d_%02d] fetch_tiffs — skipped (tiffs cached)",
                        tile_id, work.year, crow, ccol)
            if work.clog:
                work.clog.info("fetch_tiffs: skipped — tiff dir already present (%s)", work.tiff_dir)
            return work

        tiff_dir = tmp / str(work.year) / f"chunk_{crow:02d}_{ccol:02d}_tiffs"
        tiff_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[v2 tile %s %d] [chunk %02d_%02d] fetch_tiffs → %s",
                    tile_id, work.year, crow, ccol, tiff_dir.name)

        _yr_items = work.items
        n_items   = len(_yr_items)

        if work.clog:
            work.clog.info("fetch_tiffs: %d S2 items, bbox=%s", n_items, work.chunk["bbox"])

        _chunk_id = f"r{crow:02d}_c{ccol:02d}"
        if progress is not None:
            progress.fetch_update("S2 fetch", tile_id, work.year, _chunk_id, done=0, total=n_items)

        _t0 = _time.perf_counter()

        def _fetch_progress_cb(done: int, total: int) -> None:
            if progress is not None:
                progress.fetch_update("S2 fetch", done=done, total=total)

        async def _fetch_async() -> None:
            from utils.fetch import fetch_patches_to_tiff
            await fetch_patches_to_tiff(
                items=_yr_items,
                bands=FETCH_BANDS,
                bbox_wgs84=work.chunk["bbox"],
                out_dir=tiff_dir,
                max_concurrent=max_concurrent,
                band_alias=BAND_ALIAS,
                progress_cb=_fetch_progress_cb,
            )

        asyncio.run(_fetch_async())
        elapsed = _time.perf_counter() - _t0
        if work.clog:
            work.clog.info("fetch_tiffs: done (%.1fs, %d items)", elapsed, n_items)
        work.tiff_dir = tiff_dir
        return work

    def _stage_extract_scenes(work: _ChunkWork) -> _ChunkWork | None:
        """Pool B: sample pixel values from tiffs → per-scene parquets."""
        if work.clog:
            work.clog.activate()
        crow      = work.chunk["chunk_row"]
        ccol      = work.chunk["chunk_col"]
        tiff_dir  = work.tiff_dir
        chunk_pts = work.pts
        rg_size   = max(1, len(chunk_pts) // chunk_height_px)
        yr_items  = work.items

        # Scenes were pre-loaded from a surviving scenes dir — skip extraction.
        if work.scene_paths:
            logger.info("[v2 tile %s %d] [chunk %02d_%02d] extract_scenes — skipped (%d cached)",
                        tile_id, work.year, crow, ccol, len(work.scene_paths))
            if work.clog:
                work.clog.info("extract_scenes: skipped — %d scene parquets pre-loaded",
                               len(work.scene_paths))
            return work

        scene_dir = tmp / str(work.year) / f"chunk_{crow:02d}_{ccol:02d}_scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)

        point_ids = [pid      for pid, _, _   in chunk_pts]
        lons      = np.array([lon for _, lon, _ in chunk_pts], dtype=np.float64)
        lats      = np.array([lat for _, _, lat in chunk_pts], dtype=np.float64)

        _utm = cog_utm_crs or "EPSG:32755"
        from pyproj import Transformer as _Transformer
        _chunk_utm_xy = _Transformer.from_crs(
            "EPSG:4326", _utm, always_xy=True
        ).transform(lons, lats)

        n_items = len(yr_items)
        logger.info("[v2 tile %s %d] [chunk %02d_%02d] extract_scenes (%d pts, %d items)",
                    tile_id, work.year, crow, ccol, len(chunk_pts), n_items)
        if work.clog:
            work.clog.info("extract_scenes: %d pts, %d S2 items", len(chunk_pts), n_items)

        _chunk_id = f"r{crow:02d}_c{ccol:02d}"
        if progress is not None:
            progress.process_update("S2 extract", tile_id, work.year, _chunk_id, done=0, total=n_items)

        if apply_nbar:
            from utils.granule_angles import prefetch_granule_xmls
            prefetch_granule_xmls(yr_items, max_workers=8)
        _scenes_done_count = [0]  # list so closure can mutate it

        _extract_t0 = _time.perf_counter()
        _clog_ref = work.clog  # capture for use in worker threads below

        def _extract_one(item_idx: int, item) -> Path | None:
            # Re-activate in this worker thread (each _TPE thread is independent).
            if _clog_ref:
                _clog_ref.activate()
            clog_w = get_chunk_logger()

            scene_id      = item.id
            out_path      = scene_dir / f"scene_{item_idx:04d}.parquet"
            item_tiff_dir = tiff_dir / scene_id

            if out_path.exists() and out_path.stat().st_size > 0:
                try:
                    pq.ParquetFile(out_path).metadata
                    if clog_w:
                        clog_w.info(
                            "scene %d/%d %s — cached", item_idx + 1, n_items, scene_id
                        )
                    return out_path
                except Exception:
                    out_path.unlink(missing_ok=True)

            if not item_tiff_dir.exists():
                if clog_w:
                    clog_w.info(
                        "scene %d/%d %s — no tiffs (skipped in fetch)",
                        item_idx + 1, n_items, scene_id,
                    )
                return None

            _t_scene = _time.perf_counter()
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
                if clog_w:
                    clog_w.info(
                        "scene %d/%d %s — 0 clear pixels (%.2fs)",
                        item_idx + 1, n_items, scene_id, _time.perf_counter() - _t_scene,
                    )
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

            n_rows = len(tbl)
            logger.info("[v2] [chunk %02d_%02d] scene %d/%d %s — %d rows",
                        crow, ccol, item_idx + 1, n_items, scene_id, n_rows)
            if clog_w:
                clog_w.info(
                    "scene %d/%d %s — %d rows (%.2fs)",
                    item_idx + 1, n_items, scene_id, n_rows, _time.perf_counter() - _t_scene,
                )
            return out_path

        scene_paths: list[Path] = []
        with _TPE(max_workers=extract_workers) as pool:
            futs = {pool.submit(_extract_one, idx, item): idx for idx, item in enumerate(yr_items)}
            for fut in _as_completed(futs):
                result = fut.result()
                if result is not None:
                    scene_paths.append(result)
                _scenes_done_count[0] += 1
                if progress is not None:
                    progress.process_update("S2 extract", done=_scenes_done_count[0], total=n_items)

        extract_elapsed = _time.perf_counter() - _extract_t0
        del _chunk_utm_xy, lons, lats, point_ids
        import gc; gc.collect()

        shutil.rmtree(tiff_dir, ignore_errors=True)
        work.tiff_dir    = None
        work.scene_paths = scene_paths

        if not scene_paths:
            logger.info("[v2 tile %s %d] [chunk %02d_%02d] no scene data — skipping",
                        tile_id, work.year, crow, ccol)
            if work.clog:
                work.clog.info(
                    "extract_scenes: 0/%d scenes had clear pixels — chunk skipped (%.1fs)",
                    n_items, extract_elapsed,
                )
                work.clog.close(success=False)
            if progress is not None:
                progress.process_update("waiting")
                progress.chunk_skipped(f"r{crow:02d}_c{ccol:02d}", year=work.year)
            return None  # drops this chunk from the pipeline

        if work.clog:
            work.clog.info(
                "extract_scenes: %d/%d scenes produced data (%.1fs)",
                len(scene_paths), n_items, extract_elapsed,
            )
        return work

    def _stage_fetch_s1(work: _ChunkWork) -> _ChunkWork:
        """Fetch S1 patches for this chunk's bbox to the on-disk cache (network I/O only)."""
        if work.clog:
            work.clog.activate()
        crow = work.chunk["chunk_row"]
        ccol = work.chunk["chunk_col"]
        scene_dir  = tmp / str(work.year) / f"chunk_{crow:02d}_{ccol:02d}_scenes"
        s1_cache   = scene_dir / "s1_cache"
        _chunk_id  = f"r{crow:02d}_c{ccol:02d}"
        chunk_bbox = work.chunk["bbox"]

        _s1_total: list[int] = [0]

        def _on_items_resolved(n_items: int) -> None:
            if progress is not None:
                _s1_total[0] = n_items
                progress.fetch_update("S1 fetch", tile_id, work.year, _chunk_id, done=0, total=n_items)

        def _on_fetch_tick(n_done: int) -> None:
            if progress is not None:
                progress.fetch_update("S1 fetch", done=n_done, total=_s1_total[0])

        if progress is not None:
            progress.fetch_update("S1 search", tile_id, work.year, _chunk_id)
        _t0 = _time.perf_counter()
        collect_s1_for_tile(
            s2_path=None,
            bbox_wgs84=chunk_bbox,
            start=f"{work.year}-01-01",
            end=f"{work.year}-12-31",
            out_path=scene_dir / "s1_chunk.parquet",
            cache_dir=s1_cache,
            max_concurrent=max_concurrent,
            points=work.pts,
            phases={"fetch"},
            on_items_resolved=_on_items_resolved if progress is not None else None,
            on_fetch_tick=_on_fetch_tick if progress is not None else None,
        )
        if work.clog:
            work.clog.info("fetch_s1: done (%.1fs)", _time.perf_counter() - _t0)
        work.s1_cache_dir = s1_cache
        return work

    def _stage_extract_s1(work: _ChunkWork) -> _ChunkWork:
        """Extract S1 observations from the on-disk cache → s1_chunk.parquet."""
        if work.clog:
            work.clog.activate()
        crow = work.chunk["chunk_row"]
        ccol = work.chunk["chunk_col"]
        scene_dir = tmp / str(work.year) / f"chunk_{crow:02d}_{ccol:02d}_scenes"
        _chunk_id = f"r{crow:02d}_c{ccol:02d}"

        def _on_items_resolved(n_items: int) -> None:
            if progress is not None:
                progress.process_update("S1 extract", tile_id, work.year, _chunk_id, done=0, total=n_items)

        def _on_extract_tick(n_done: int) -> None:
            if progress is not None:
                progress.process_update("S1 extract", done=n_done)

        _t0 = _time.perf_counter()
        work.s1_path = collect_s1_for_tile(
            s2_path=None,
            bbox_wgs84=work.chunk["bbox"],
            start=f"{work.year}-01-01",
            end=f"{work.year}-12-31",
            out_path=scene_dir / "s1_chunk.parquet",
            cache_dir=work.s1_cache_dir,
            max_concurrent=max_concurrent,
            points=work.pts,
            phases={"extract"},
            on_items_resolved=_on_items_resolved if progress is not None else None,
            on_extract_tick=_on_extract_tick if progress is not None else None,
        )
        if work.clog:
            s1_rows = 0
            if work.s1_path and work.s1_path.exists():
                try:
                    import pyarrow.parquet as _pq2
                    s1_rows = _pq2.read_metadata(work.s1_path).num_rows
                except Exception:
                    pass
            work.clog.info(
                "extract_s1: %d S1 rows (%.1fs)",
                s1_rows, _time.perf_counter() - _t0,
            )
        return work

    def _stage_merge(work: _ChunkWork) -> _ChunkWork:
        """K-way heap merge of scene parquets + S1 → sorted chunk parquet."""
        if work.clog:
            work.clog.activate()
        crow = work.chunk["chunk_row"]
        ccol = work.chunk["chunk_col"]
        scene_dir = tmp / str(work.year) / f"chunk_{crow:02d}_{ccol:02d}_scenes"
        chunk_out = tmp / str(work.year) / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
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
        _chunk_id = f"r{crow:02d}_c{ccol:02d}"
        if progress is not None:
            progress.process_update("reading", tile_id, work.year, _chunk_id)

        def _merge_phase(label: str) -> None:
            if progress is not None:
                # map merge_scenes on_phase strings to our exhaustive stage names
                if label.startswith("sorting"):
                    progress.process_update("sorting", label=label)
                elif label.startswith("writing"):
                    progress.process_update("writing", label=label)
                else:
                    progress.process_update("reading", label=label)

        _t0 = _time.perf_counter()
        merge_scenes(work.scene_paths, work.s1_path, chunk_out, chunk_metadata=_meta, on_phase=_merge_phase)
        if progress is not None:
            progress.process_update("waiting")
            progress.chunk_done(_chunk_id, tile_id=tile_id, year=work.year)
        if work.clog:
            final_rows = 0
            try:
                import pyarrow.parquet as _pq3
                final_rows = _pq3.read_metadata(chunk_out).num_rows
            except Exception:
                pass
            work.clog.info(
                "merge_scenes: %d input files → %d total rows (%.1fs)",
                len(work.scene_paths), final_rows, _time.perf_counter() - _t0,
            )
            size_mb = chunk_out.stat().st_size / 1e6 if chunk_out.exists() else 0.0
            work.clog.info("output: %s (%.1f MB)", chunk_out.name, size_mb)
            work.clog.close(success=True)
        shutil.rmtree(scene_dir, ignore_errors=True)
        logger.info("[v2 tile %s %d] [chunk %02d_%02d] ready → %s",
                    tile_id, work.year, crow, ccol, chunk_out.name)
        work.scene_paths = []
        work.s1_path     = None
        return work

    # --- Pipeline: fetch thread + per-chunk S1 thread + process thread ---------
    #
    # Three-thread pipeline:
    #   fetch thread:   S2 fetch → hands off to a per-chunk S1 thread
    #   S1 thread:      S1 fetch (one per chunk, overlaps with next S2 fetch)
    #   process thread: extract → merge (drains _fetched_q)
    #
    # The fetch thread starts S2 for chunk N+1 while S1 runs for chunk N.
    import queue as _queue

    # Semaphore (not queue maxsize) bounds concurrent disk usage; queue is unbounded
    # so S1 threads can always post without deadlocking against the semaphore.
    _fetched_q: _queue.Queue = _queue.Queue()
    _done_q:    _queue.Queue = _queue.Queue()
    _STOP = object()
    _fetch_error:   list[BaseException] = []
    _process_error: list[BaseException] = []

    def _chunk_inputs():
        for yr in _years:
            # Search this year on demand if not yet in items_by_year.
            if yr not in items_by_year:
                if yr not in _years_pending:
                    continue   # already searched, came back empty
                _years_pending.remove(yr)
                if progress is not None:
                    progress.fetch_update("STAC search", tile_id, yr, "")
                _search_year(yr)
                if progress is not None:
                    progress.fetch_update("waiting")
                if yr not in items_by_year:
                    continue   # no scenes for this year
                # Set progress total now that we know the chunk count.
                if progress is not None:
                    n_skip_yr = sum(1 for (r, c, y) in _skip if y == yr)
                    progress.set_total(yr, len(chunks), already_done=n_skip_yr)
                    logger.info("[v2 tile %s %d] %d chunks to fetch (lazy search)",
                                tile_id, yr, len(chunks) - n_skip_yr)
            yr_items = items_by_year[yr]
            for chunk in chunks:
                crow = chunk["chunk_row"]
                ccol = chunk["chunk_col"]
                key  = (crow, ccol, yr)
                if (crow, ccol) < resume_from_chunk:
                    logger.info("[v2 tile %s %d] [chunk %02d_%02d] skipping (resume_from_chunk=%s)",
                                tile_id, yr, crow, ccol, resume_from_chunk)
                    continue
                if key in _skip:
                    logger.info("[v2 tile %s %d] [chunk %02d_%02d] skipping (already exists)",
                                tile_id, yr, crow, ccol)
                    continue
                pts = make_chunk_points(chunk, chunks_meta)
                clog: object = None
                if log_dir is not None:
                    _ldir = log_dir / str(yr) / tile_id / "fetchlogs"
                    clog = ChunkLogger(tile_id, yr, crow, ccol, _ldir)
                    clog.open()
                cached_tiff_dir = tmp / str(yr) / f"chunk_{crow:02d}_{ccol:02d}_tiffs"
                pre_fetched = (
                    cached_tiff_dir.exists()
                    and any(cached_tiff_dir.iterdir())
                )
                # If a scenes dir survived from a prior interrupted run (extract
                # finished but merge didn't), skip both fetch and extract entirely.
                cached_scene_dir = tmp / str(yr) / f"chunk_{crow:02d}_{ccol:02d}_scenes"
                pre_extracted: list[Path] = []
                if not pre_fetched and cached_scene_dir.exists():
                    pre_extracted = sorted(
                        p for p in cached_scene_dir.glob("scene_????.parquet")
                        if p.stat().st_size > 0
                    )
                    if pre_extracted:
                        logger.info(
                            "[v2 tile %s %d] [chunk %02d_%02d] %d scene parquets cached — "
                            "skipping fetch+extract",
                            tile_id, yr, crow, ccol, len(pre_extracted),
                        )
                yield _ChunkWork(
                    chunk=chunk,
                    year=yr,
                    items=yr_items,
                    pts=pts,
                    clog=clog,
                    tiff_dir=cached_tiff_dir if pre_fetched else None,
                    scene_paths=pre_extracted,
                )

    import threading as _threading

    # Semaphore caps how many chunks have tiffs+S1 cache on disk simultaneously.
    _fetch_sem = _threading.Semaphore(prefetch_depth)

    def _fetch_thread():
        try:
            if progress is not None:
                progress.fetch_update("waiting")
            for work in _chunk_inputs():
                _fetch_sem.acquire()          # wait until process thread has capacity
                work = _stage_fetch_tiffs(work)
                work = _stage_fetch_s1(work)  # S1 fetch is network I/O: stays in fetch thread
                _chunk_id = f"r{work.chunk['chunk_row']:02d}_c{work.chunk['chunk_col']:02d}"
                if progress is not None:
                    progress.fetch_update("waiting")
                    progress.chunk_fetched(_chunk_id)
                _fetched_q.put(work)
        except Exception as exc:
            _fetch_error.append(exc)
        finally:
            if progress is not None:
                progress.fetch_update("done")
            _fetched_q.put(_STOP)

    def _process_thread():
        work = None
        try:
            if progress is not None:
                progress.process_update("waiting")
            while True:
                work = _fetched_q.get()
                if work is _STOP:
                    break
                _chunk_id = f"r{work.chunk['chunk_row']:02d}_c{work.chunk['chunk_col']:02d}"
                if progress is not None:
                    progress.chunk_dequeued(_chunk_id)
                _fetch_sem.release()          # a slot freed; fetch thread can advance
                work = _stage_extract_scenes(work)
                if work is None:
                    if progress is not None:
                        progress.process_update("waiting")
                    continue  # semaphore already released above
                work = _stage_extract_s1(work)  # S1 extract is memory-bound: stays in process thread
                if hasattr(work, "pts"):
                    del work.pts
                import gc; gc.collect()
                work = _stage_merge(work)
                crow = work.chunk["chunk_row"]
                ccol = work.chunk["chunk_col"]
                chunk_out = tmp / str(work.year) / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
                if work.clog and work.clog.logger is not None:
                    work.clog.close(success=True)
                _done_q.put((crow, ccol, work.year, chunk_out))
        except Exception as exc:
            if work is not None and work is not _STOP and hasattr(work, "clog") and work.clog:
                work.clog.error("_process_thread EXCEPTION: %s", exc)
                work.clog.close(success=False)
            _process_error.append(exc)
        finally:
            if progress is not None:
                progress.process_update("done")
            _done_q.put(_STOP)

    _ft = _threading.Thread(target=_fetch_thread,   daemon=True, name=f"fetch_tile_{tile_id}.fetch")
    _pt = _threading.Thread(target=_process_thread, daemon=True, name=f"fetch_tile_{tile_id}.process")
    _ft.start()
    _pt.start()

    while True:
        result = _done_q.get()
        if result is _STOP:
            break
        if _fetch_error:
            raise _fetch_error[0]
        if _process_error:
            raise _process_error[0]
        crow, ccol, yr, chunk_out = result
        if progress is not None:
            progress.refresh()
        yield crow, ccol, yr, chunk_out

    _ft.join()
    _pt.join()
    if _fetch_error:
        raise _fetch_error[0]
    if _process_error:
        raise _process_error[0]
