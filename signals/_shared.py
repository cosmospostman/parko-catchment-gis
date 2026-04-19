"""signals/_shared.py — Shared computational kernels and site-param loader.

Three pure functions used by two or more signal classes:
  load_and_filter   — parquet load + SCL purity filter + year/month columns
  annual_percentile — per-pixel per-year percentile, averaged across years
  dry_season_cv     — per-pixel inter-annual CV of dry-season median

All three kernels operate on Polars DataFrames internally for parallel
groupby/agg performance on large parquets (tens of millions of rows).
Inputs may be a pandas DataFrame or a Path; outputs are always pandas.

Plus load_signal_params() which reads signal overrides from a Location's YAML
and returns a populated Params instance merged with the signal's defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Computational kernels
# ---------------------------------------------------------------------------

def load_and_filter(
    path_or_df: Union[Path, pd.DataFrame, pl.DataFrame],
    scl_purity_min: float,
    load_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Load a pixel parquet (or accept a pre-loaded DataFrame), apply SCL
    purity filter, and add ``year`` / ``month`` integer columns.

    Parameters
    ----------
    path_or_df:
        Path to a parquet file, or a pandas DataFrame.
    scl_purity_min:
        Minimum ``scl_purity`` fraction to retain a row.
    load_cols:
        If given, only these columns are loaded from disk (ignored when a
        DataFrame is passed directly).

    Returns
    -------
    Polars DataFrame with ``year`` and ``month`` columns added.
    """
    if isinstance(path_or_df, pl.DataFrame):
        df = path_or_df
    elif isinstance(path_or_df, pd.DataFrame):
        df = pl.from_pandas(path_or_df)
    else:
        df = pl.read_parquet(path_or_df, columns=load_cols)

    df = df.filter(pl.col("scl_purity") >= scl_purity_min)
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    ])
    return df


def compute_features_chunked(
    path: Path,
    scl_purity_min: float,
    dry_months: list[int],
    min_obs_per_year: int,
    min_obs_dry: int,
    rec_p_percentile: float = 0.10,
    re_floor_percentile: float = 0.10,
    swir_floor_percentile: float = 0.10,
    year_from: int | None = None,
    year_to: int | None = None,
    curve_out_path: Path | None = None,
    smooth_days: int = 30,
    compute_ndvi_integral: bool = False,
    calibration_path: Path | None = None,
    tile_id: str | None = None,
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """Compute all four Parkinsonia signal features in a single row-group pass.

    Requires the parquet to be sorted by ``point_id`` (pixel-sorted) so that
    each row group contains complete pixel histories with no cross-group splits.
    Use ``sort_parquet_by_pixel()`` to prepare the file if needed.

    Each row group is read, filtered, and aggregated independently. Peak RAM
    is proportional to one row group (~1 GB for 5M rows × 10 cols), not the
    full file size.

    Parameters
    ----------
    path:
        Path to the pixel-sorted parquet file.
    scl_purity_min:
        Minimum scl_purity to retain an observation.
    dry_months:
        Month numbers defining the dry season (for nir_cv).
    min_obs_per_year:
        Minimum observations per (pixel, year) for rec_p / re_p10 / swir_p10.
    min_obs_dry:
        Minimum dry-season observations per (pixel, year) for nir_cv.
    rec_p_percentile, re_floor_percentile, swir_floor_percentile:
        Percentile overrides (default 0.10 for floor signals; rec_p uses
        p90-p10 amplitude, not a configurable percentile).
    year_from, year_to:
        Optional inclusive year bounds. Observations outside this range are
        dropped before any aggregation.
    curve_out_path:
        If provided, the smoothed NDVI curve is written to this path in the
        same row-group pass (no second scan needed for ndvi_integral).
        Deprecated in favour of ``compute_ndvi_integral=True``, which avoids
        writing a large intermediate file.
    smooth_days:
        Rolling-median window for the NDVI curve (used when
        ``curve_out_path`` is given or ``compute_ndvi_integral=True``).
    compute_ndvi_integral:
        If True, also compute per-pixel-per-year ndvi_smooth mean (and
        is_sparse_year flag) in the same pass and return it as a second
        DataFrame alongside the base features.  No temp file is written —
        only the small per-(pixel,year) aggregates are accumulated.
        When True, returns ``(base_df, integral_yearly_df)`` where
        ``integral_yearly_df`` has columns
        ``[point_id, year, ndvi_mean, n_obs, is_sparse_year]``.

    Returns
    -------
    If ``compute_ndvi_integral`` is False (default):
        Pandas DataFrame with columns
        ``[point_id, lon, lat, nir_cv, rec_p, re_p10, swir_p10]``.
    If ``compute_ndvi_integral`` is True:
        Tuple ``(base_df, integral_yearly_df)`` — base_df as above, plus
        ``integral_yearly_df`` with per-(pixel,year) ndvi stats.
    """
    import pyarrow.parquet as pq
    from scipy.ndimage import median_filter

    re_n  = int(round(re_floor_percentile * 100))
    sw_n  = int(round(swir_floor_percentile * 100))
    window_rows = max(3, smooth_days // 5)

    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups

    _schema_names = set(pf.schema_arrow.names)
    LOAD_COLS = ["point_id", "lon", "lat", "date", "scl_purity",
                 "B04", "B05", "B07", "B08", "B11"]
    if "tile_id" in _schema_names:
        LOAD_COLS = ["point_id", "lon", "lat", "date", "tile_id", "scl_purity",
                     "B04", "B05", "B07", "B08", "B11"]
    if tile_id is not None and "item_id" in _schema_names and "item_id" not in LOAD_COLS:
        LOAD_COLS = LOAD_COLS + ["item_id"]

    # Load correction table once — None if no calibration_path or file absent
    from utils.tile_harmonisation import load_corrections
    corrections = load_corrections(calibration_path) if calibration_path else None
    if corrections:
        _corr_df = pl.DataFrame([
            {"tile_id": t, "band": b, "year": y, "scale": s}
            for (t, b, y), s in corrections.items()
        ])
    else:
        _corr_df = None

    parts: list[pd.DataFrame] = []
    integral_parts: list[pd.DataFrame] = []
    curve_writer = None
    print(f"  [chunked] {path.name}: {n_rg} row groups, {pf.metadata.num_rows:,} rows total")

    tail_buf: pl.DataFrame | None = None  # rows of a split pixel carried from previous rg

    for rg_idx in range(n_rg):
        if rg_idx % max(1, n_rg // 10) == 0:
            pct = 100 * rg_idx // n_rg
            print(f"  [chunked] row group {rg_idx + 1}/{n_rg} ({pct}%)", flush=True)
        table = pf.read_row_groups([rg_idx], columns=LOAD_COLS)
        chunk = pl.from_arrow(table)
        del table

        # Re-attach rows of any pixel that was split at the previous row group boundary
        if tail_buf is not None:
            if chunk["point_id"][0] == tail_buf["point_id"][0]:
                chunk = pl.concat([tail_buf, chunk])
            tail_buf = None

        # Peel off the last pixel — it may continue into the next row group
        if rg_idx < n_rg - 1:
            last_pid = chunk["point_id"][-1]
            tail_mask = chunk["point_id"] == last_pid
            tail_buf = chunk.filter(tail_mask)
            chunk = chunk.filter(~tail_mask)

        if chunk.is_empty():
            continue

        chunk = chunk.filter(pl.col("scl_purity") >= scl_purity_min)

        # Stage 1 — temporal columns (needed as join key for corrections)
        chunk = chunk.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        ])

        # Stage 2 — apply tile corrections (join on tile_id + year)
        if _corr_df is not None and "tile_id" in chunk.columns:
            for band in ["B04", "B05", "B07", "B08", "B11"]:
                band_corr = (
                    _corr_df.filter(pl.col("band") == band)
                    .select(["tile_id", "year", "scale"])
                )
                chunk = (
                    chunk
                    .join(band_corr, on=["tile_id", "year"], how="left")
                    .with_columns(
                        pl.when(pl.col("scale").is_not_null())
                          .then(pl.col(band) * pl.col("scale"))
                          .otherwise(pl.col(band))
                          .alias(band)
                    )
                    .drop("scale")
                )

        # Stage 3 — derived bands (use corrected B04/B05/B07/B08/B11)
        chunk = chunk.with_columns([
            ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))).alias("ndvi"),
            (pl.col("B07") / pl.col("B05")).alias("re_ratio"),
            ((pl.col("B08") - pl.col("B11")) / (pl.col("B08") + pl.col("B11"))).alias("swir_mi"),
        ])

        if tile_id is not None:
            if "item_id" in chunk.columns:
                chunk = chunk.filter(
                    pl.col("item_id").str.split("_").list.get(1) == tile_id
                )
            elif "tile_id" in chunk.columns:
                chunk = chunk.filter(pl.col("tile_id") == tile_id)
        if year_from is not None:
            chunk = chunk.filter(pl.col("year") >= year_from)
        if year_to is not None:
            chunk = chunk.filter(pl.col("year") <= year_to)
        if chunk.is_empty():
            continue

        # nir_cv: dry-season B08 inter-annual CV — reduce to (pixel, year) first
        nir_annual = (
            chunk
            .filter(pl.col("month").is_in(dry_months))
            .group_by(["point_id", "year"])
            .agg([
                pl.col("B08").median().alias("_nir_med"),
                pl.col("B08").count().alias("_nir_n"),
            ])
            .filter(pl.col("_nir_n") >= min_obs_dry)
        )
        nir_stats = (
            nir_annual
            .group_by("point_id")
            .agg([
                pl.col("_nir_med").mean().alias("_nir_mean"),
                pl.col("_nir_med").std().alias("_nir_std"),
            ])
            .with_columns(
                (pl.col("_nir_std") / pl.col("_nir_mean")).alias("nir_cv")
            )
            .select(["point_id", "nir_cv"])
        )

        # rec_p / re_p10 / swir_p10 — annual agg then mean across years
        annual = (
            chunk
            .group_by(["point_id", "year"])
            .agg([
                pl.col("ndvi").quantile(0.90, interpolation="linear").alias("_ndvi_p90"),
                pl.col("ndvi").quantile(0.10, interpolation="linear").alias("_ndvi_p10"),
                pl.col("ndvi").count().alias("_ndvi_n"),
                pl.col("re_ratio").quantile(re_floor_percentile, interpolation="linear").alias(f"_re_p{re_n}"),
                pl.col("re_ratio").count().alias("_re_n"),
                pl.col("swir_mi").quantile(swir_floor_percentile, interpolation="linear").alias(f"_sw_p{sw_n}"),
                pl.col("swir_mi").count().alias("_sw_n"),
                pl.col("lon").first(),
                pl.col("lat").first(),
            ])
            .filter(pl.col("_ndvi_n") >= min_obs_per_year)
            .with_columns(
                (pl.col("_ndvi_p90") - pl.col("_ndvi_p10")).alias("_rec_p")
            )
        )

        pixel_stats = (
            annual
            .group_by("point_id")
            .agg([
                pl.col("_rec_p").mean().alias("rec_p"),
                pl.col(f"_re_p{re_n}").filter(pl.col("_re_n") >= min_obs_per_year).mean().alias("re_p10"),
                pl.col(f"_sw_p{sw_n}").filter(pl.col("_sw_n") >= min_obs_per_year).mean().alias("swir_p10"),
                pl.col("lon").first(),
                pl.col("lat").first(),
            ])
        )

        chunk_result = (
            pixel_stats
            .join(nir_stats, on="point_id", how="left")
            .select(["point_id", "lon", "lat", "nir_cv", "rec_p", "re_p10", "swir_p10"])
            .to_pandas()
        )
        parts.append(chunk_result)
        del annual, nir_annual, nir_stats, pixel_stats

        # Compute NDVI curve and handle curve_out_path / compute_ndvi_integral
        if curve_out_path is not None or compute_ndvi_integral:
            obs_counts = (
                chunk.group_by(["point_id", "year"])
                .agg(pl.len().alias("_n_obs"))
            )
            curve_chunk = (
                chunk
                .with_columns(pl.col("date").dt.ordinal_day().alias("doy"))
                .join(obs_counts, on=["point_id", "year"], how="left")
                .with_columns((pl.col("_n_obs") < min_obs_per_year).alias("is_sparse_year"))
                .drop("_n_obs")
                .sort(["point_id", "year", "date"])
            )

            ndvi_arr = curve_chunk["ndvi"].to_numpy(allow_copy=True)
            pid_arr  = curve_chunk["point_id"].to_numpy(allow_copy=True)
            yr_arr   = curve_chunk["year"].to_numpy(allow_copy=True)

            change = np.ones(len(ndvi_arr), dtype=bool)
            change[1:] = (pid_arr[1:] != pid_arr[:-1]) | (yr_arr[1:] != yr_arr[:-1])
            starts = np.flatnonzero(change)
            ends   = np.append(starts[1:], len(ndvi_arr))

            ndvi_smooth = np.empty_like(ndvi_arr)
            for s, e in zip(starts, ends):
                ndvi_smooth[s:e] = median_filter(ndvi_arr[s:e], size=window_rows, mode="nearest")

            curve_with_smooth = curve_chunk.with_columns(pl.Series("ndvi_smooth", ndvi_smooth))

            if curve_out_path is not None:
                curve_result = curve_with_smooth.select(
                    ["point_id", "lon", "lat", "date", "year", "month", "doy",
                     "ndvi", "ndvi_smooth", "is_sparse_year"]
                )
                arrow_table = curve_result.to_arrow()
                if curve_writer is None:
                    curve_writer = pq.ParquetWriter(str(curve_out_path), arrow_table.schema)
                curve_writer.write_table(arrow_table)
                del curve_result, arrow_table

            if compute_ndvi_integral:
                # Accumulate only the tiny per-(pixel,year) aggregates — no raw curve written
                integral_chunk = (
                    curve_with_smooth
                    .group_by(["point_id", "year"])
                    .agg([
                        pl.col("ndvi_smooth").mean().alias("ndvi_mean"),
                        pl.len().alias("n_obs"),
                        pl.col("is_sparse_year").first().alias("is_sparse_year"),
                    ])
                    .to_pandas()
                )
                integral_parts.append(integral_chunk)
                del integral_chunk

            del curve_chunk, curve_with_smooth

        del chunk

    if curve_writer is not None:
        curve_writer.close()

    base_df = pd.concat(parts, ignore_index=True)
    if compute_ndvi_integral:
        integral_yearly_df = pd.concat(integral_parts, ignore_index=True) if integral_parts else pd.DataFrame(
            columns=["point_id", "year", "ndvi_mean", "n_obs", "is_sparse_year"]
        )
        return base_df, integral_yearly_df
    return base_df


def is_pixel_sorted(path: Path, n_check: int = 2) -> bool:
    """Return True if ``path`` is pixel-sorted (no point_id overlap between adjacent row groups).

    Checks the first ``n_check`` pairs of adjacent row groups.  A file with
    only one row group is trivially sorted.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    if n_rg <= 1:
        return True

    pairs = min(n_check, n_rg - 1)
    for i in range(pairs):
        ids_a = set(
            pl.from_arrow(pf.read_row_groups([i],     columns=["point_id"]))["point_id"].to_list()
        )
        ids_b = set(
            pl.from_arrow(pf.read_row_groups([i + 1], columns=["point_id"]))["point_id"].to_list()
        )
        if ids_a & ids_b:
            return False
    return True


def ensure_pixel_sorted(path: Path, row_group_size: int = 5_000_000) -> Path:
    """Return a pixel-sorted version of ``path``, sorting it first if needed.

    If the parquet is already pixel-sorted the original path is returned
    unchanged.  Otherwise a ``<stem>-by-pixel.parquet`` sibling is written (or
    reused if it already exists) and its path is returned.

    Parameters
    ----------
    path:
        Path to the candidate parquet file.
    row_group_size:
        Target rows per row group passed to ``sort_parquet_by_pixel``.

    Returns
    -------
    Path to a pixel-sorted parquet (may equal ``path``).
    """
    if is_pixel_sorted(path):
        print(f"  [sort-check] {path.name}: already pixel-sorted")
        return path

    sorted_path = path.with_name(path.stem + "-by-pixel.parquet")
    if sorted_path.exists():
        if sorted_path.stat().st_size == 0:
            print(f"  [sort-check] {sorted_path.name}: 0-byte file (previous crash?) — deleting and re-sorting")
            sorted_path.unlink()
        else:
            print(f"  [sort-check] {path.name}: using cached pixel-sorted file → {sorted_path.name}")
            return sorted_path

    import pyarrow.parquet as pq
    n_rg = pq.ParquetFile(path).metadata.num_row_groups
    print(
        f"  [sort-check] {path.name}: not pixel-sorted ({n_rg} row groups) — "
        f"sorting to {sorted_path.name} (this runs once) ..."
    )
    sort_parquet_by_pixel(path, sorted_path, row_group_size=row_group_size)
    print(f"  [sort-check] sort complete → {sorted_path.name}")
    return sorted_path


def sort_parquet_by_pixel(
    src: Path,
    dst: Path,
    row_group_size: int = 5_000_000,
    ram_budget_gb: float = 8.0,
) -> None:
    """Write a copy of ``src`` sorted by ``point_id`` using an in-memory multi-pass sort.

    Algorithm:
    1. Discover all unique row-coords by scanning ``point_id`` (column-only read).
    2. Divide coords into passes that fit within ``ram_budget_gb``.
    3. For each pass: scan all source row groups once, accumulate rows for the
       target coords in RAM, sort by ``point_id``, append to output.

    No temp files are written — all sorting is in RAM.  The source file is read
    sequentially (compressed, ~50 MB/rg) which is fast and cache-friendly.

    Peak RAM ≈ one pass worth of buckets (≤ ``ram_budget_gb``).

    Parameters
    ----------
    src:
        Source parquet path (any row order).
    dst:
        Destination path for the pixel-sorted parquet.
    row_group_size:
        Target rows per row group in the output.
    ram_budget_gb:
        How many GB of RAM to use per pass for in-memory buckets.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    n_rg = pf.metadata.num_row_groups

    # ── Step 1: discover all unique row-coords (point_id column only) ─────
    print(f"  [sort] scanning {n_rg} row groups for unique row-coords ...", flush=True)
    all_coords: set[str] = set()
    for i in range(n_rg):
        tbl = pf.read_row_group(i, columns=["point_id"])
        ids = tbl.column("point_id")
        row_coords = pc.list_flatten(pc.list_slice(pc.split_pattern(ids, "_"), 1, 2))
        all_coords.update(pc.unique(row_coords).to_pylist())
        if i % 400 == 0:
            print(f"  [sort] coord scan: {i}/{n_rg}, {len(all_coords)} coords so far", flush=True)

    sorted_coords = sorted(all_coords)
    n_coords = len(sorted_coords)

    # ── Step 2: divide coords into passes ─────────────────────────────────
    total_rows = pf.metadata.num_rows
    # Measure actual uncompressed bytes/row from Parquet metadata.
    # Apply a 2× overhead factor: PyArrow in-memory tables carry dict
    # metadata, validity bitmaps, and string offsets above the raw bytes.
    meta = pf.metadata
    total_uncompressed = sum(
        meta.row_group(i).column(j).total_uncompressed_size
        for i in range(meta.num_row_groups)
        for j in range(meta.num_columns)
    )
    bytes_per_row = (total_uncompressed / total_rows * 2.0) if total_rows else 260
    rows_per_coord = total_rows / n_coords
    bytes_per_coord = rows_per_coord * bytes_per_row
    coords_per_pass = max(1, int(ram_budget_gb * 1e9 / bytes_per_coord))
    passes = [sorted_coords[i:i + coords_per_pass] for i in range(0, n_coords, coords_per_pass)]
    print(
        f"  [sort] {n_coords} coords, ~{bytes_per_coord/1e6:.0f} MB/coord, "
        f"{coords_per_pass} coords/pass → {len(passes)} passes",
        flush=True,
    )

    # ── Step 3: multi-pass scan → sort → write ────────────────────────────
    out_writer: "pq.ParquetWriter | None" = None
    try:
        out_writer = pq.ParquetWriter(str(dst), schema, compression="snappy")

        for pass_idx, pass_coords in enumerate(passes):
            coord_set = set(pass_coords)
            buckets: dict[str, list[pa.Table]] = {c: [] for c in pass_coords}

            print(
                f"  [sort] pass {pass_idx+1}/{len(passes)}: "
                f"scanning {n_rg} row groups for {len(pass_coords)} coords ...",
                flush=True,
            )
            for i in range(n_rg):
                batch = pf.read_row_group(i)
                ids = batch.column("point_id")
                row_coords = pc.list_flatten(pc.list_slice(pc.split_pattern(ids, "_"), 1, 2))

                # Filter to only rows whose coord is in this pass
                in_pass = pc.is_in(row_coords, value_set=pa.array(list(coord_set)))
                if not pc.any(in_pass).as_py():
                    continue
                batch = batch.filter(in_pass)
                row_coords = row_coords.filter(in_pass)

                # Split into per-coord buckets
                unique_in_batch = pc.unique(row_coords).to_pylist()
                if len(unique_in_batch) == 1:
                    buckets[unique_in_batch[0]].append(batch)
                else:
                    for coord in unique_in_batch:
                        mask = pc.equal(row_coords, coord)
                        buckets[coord].append(batch.filter(mask))

                if i % 400 == 0:
                    print(f"  [sort] pass {pass_idx+1}: {i}/{n_rg} row groups", flush=True)

            # Sort each coord bucket and write to output.
            # We accumulate whole row-coord tables and write one row group per
            # coord — never slicing mid-pixel — so adjacent row groups always
            # contain disjoint point_id sets (pixel-sorted invariant holds).
            pending: list[pa.Table] = []
            pending_rows = 0

            def _flush() -> None:
                nonlocal pending_rows
                if not pending:
                    return
                # Write each coord table as its own row group so we never split
                # a pixel across a row-group boundary.
                for tbl in pending:
                    out_writer.write_table(tbl)
                pending.clear()
                pending_rows = 0

            for coord in pass_coords:
                chunks = buckets.pop(coord)
                if not chunks:
                    continue
                bucket_tbl = pa.concat_tables(chunks)
                order = pc.sort_indices(bucket_tbl, sort_keys=[("point_id", "ascending")])
                bucket_tbl = bucket_tbl.take(order)
                pending.append(bucket_tbl)
                pending_rows += len(bucket_tbl)
                if pending_rows >= row_group_size:
                    _flush()

            _flush()
            print(f"  [sort] pass {pass_idx+1}/{len(passes)} done", flush=True)

        out_writer.close()

    except Exception:
        if out_writer is not None:
            try:
                out_writer.close()
            except Exception:
                pass
        if dst.exists():
            dst.unlink()
        raise


def annual_percentile(
    df: pl.DataFrame,
    value_col: str,
    percentile: float,
    min_obs_per_year: int,
) -> pd.DataFrame:
    """Per-pixel per-year percentile, averaged across qualifying years.

    Parameters
    ----------
    df:
        Polars observation-level DataFrame with ``point_id``, ``year``, and
        ``value_col`` columns.
    value_col:
        Column name to compute percentile on.
    percentile:
        Percentile in [0, 1], e.g. 0.10 for the 10th percentile.
    min_obs_per_year:
        Minimum observations per (pixel, year) to include that year.

    Returns
    -------
    Pandas DataFrame with columns ``[point_id, <value_col>_p<n>, n_years]``
    where ``<n>`` is the percentile as an integer (e.g. ``re_ratio_p10``).
    """
    n = int(round(percentile * 100))
    out_col = f"{value_col}_p{n}"

    # Annual percentile per (pixel, year), filtering sparse years
    annual = (
        df.group_by(["point_id", "year"])
        .agg([
            pl.col(value_col).quantile(percentile, interpolation="linear").alias(out_col),
            pl.col(value_col).count().alias("_n_obs"),
        ])
        .filter(pl.col("_n_obs") >= min_obs_per_year)
    )

    # Mean across years + year count
    result = (
        annual.group_by("point_id")
        .agg([
            pl.col(out_col).mean(),
            pl.col(out_col).count().alias("n_years"),
        ])
    )

    return result.to_pandas()


def dry_season_cv(
    df: pl.DataFrame,
    value_col: str,
    dry_months: list[int],
    min_obs_dry: int,
) -> pd.DataFrame:
    """Per-pixel inter-annual coefficient of variation of dry-season median.

    Parameters
    ----------
    df:
        Polars observation-level DataFrame with ``point_id``, ``year``,
        ``month``, and ``value_col`` columns.
    value_col:
        Column name to compute CV on (typically ``B08`` for NIR).
    dry_months:
        Month numbers that define the dry season (from ``loc.dry_months``).
    min_obs_dry:
        Minimum observations per (pixel, year) within dry months to include
        that year.

    Returns
    -------
    Pandas DataFrame with columns
    ``[point_id, <value_col>_mean, <value_col>_std, <value_col>_cv, n_years]``.
    """
    df_dry = df.filter(pl.col("month").is_in(dry_months))

    # Annual dry-season median per (pixel, year), filtering sparse years
    medians = (
        df_dry.group_by(["point_id", "year"])
        .agg([
            pl.col(value_col).median().alias("_annual_med"),
            pl.col(value_col).count().alias("_n_obs"),
        ])
        .filter(pl.col("_n_obs") >= min_obs_dry)
    )

    # Inter-annual mean, std, CV
    result = (
        medians.group_by("point_id")
        .agg([
            pl.col("_annual_med").mean().alias(f"{value_col}_mean"),
            pl.col("_annual_med").std().alias(f"{value_col}_std"),
            pl.col("_annual_med").count().alias("n_years"),
        ])
        .with_columns([
            (pl.col(f"{value_col}_std") / pl.col(f"{value_col}_mean")).alias(f"{value_col}_cv"),
        ])
    )

    return result.to_pandas()


# ---------------------------------------------------------------------------
# Annual NDVI curve kernel
# ---------------------------------------------------------------------------

def annual_ndvi_curve_chunked(
    sorted_parquet_path: "Path",
    out_path: "Path",
    smooth_days: int,
    min_obs_per_year: int,
    scl_purity_min: float = 0.0,
    year_from: int | None = None,
    year_to: int | None = None,
) -> "Path":
    """Like ``annual_ndvi_curve`` but reads a pixel-sorted parquet row-group
    by row-group, writing results incrementally to ``out_path`` rather than
    accumulating in memory.

    The parquet must be sorted by ``point_id`` with complete pixel histories
    within each row group (no cross-group splits).  Use
    ``sort_parquet_by_pixel()`` to prepare the file if needed.

    Peak RAM is proportional to one row group (~5M rows ≈ 400 MB) rather than
    the full dataset.  The accumulated result is the same as
    ``annual_ndvi_curve`` applied to the full frame, but stored on disk.
    Callers should use ``pl.scan_parquet(out_path)`` with streaming to avoid
    loading the full observation-level curve into RAM.

    Parameters
    ----------
    sorted_parquet_path:
        Path to the pixel-sorted parquet (e.g. ``*-by-pixel.parquet``).
    out_path:
        Destination path for the curve parquet file.
    smooth_days, min_obs_per_year:
        Same semantics as ``annual_ndvi_curve``.
    scl_purity_min:
        Rows with ``scl_purity < scl_purity_min`` are dropped before
        processing.  Pass 0.0 to skip (e.g. if the parquet is pre-filtered).
    year_from, year_to:
        Optional inclusive year bounds applied before curve computation.

    Returns
    -------
    ``out_path`` — path to the written curve parquet.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa
    from scipy.ndimage import median_filter

    LOAD_COLS = ["point_id", "lon", "lat", "date", "B08", "B04"]
    if scl_purity_min > 0.0:
        LOAD_COLS.append("scl_purity")

    window_rows = max(3, smooth_days // 5)
    pf = pq.ParquetFile(sorted_parquet_path)
    n_rg = pf.metadata.num_row_groups
    writer = None

    tail_buf: pl.DataFrame | None = None  # rows of a split pixel carried from previous rg

    for rg_idx in range(n_rg):
        table = pf.read_row_groups([rg_idx], columns=LOAD_COLS)
        chunk = pl.from_arrow(table)
        del table

        # Re-attach rows of any pixel that was split at the previous row group boundary
        if tail_buf is not None:
            if chunk["point_id"][0] == tail_buf["point_id"][0]:
                chunk = pl.concat([tail_buf, chunk])
            tail_buf = None

        # Peel off the last pixel — it may continue into the next row group
        if rg_idx < n_rg - 1:
            last_pid = chunk["point_id"][-1]
            tail_mask = chunk["point_id"] == last_pid
            tail_buf = chunk.filter(tail_mask)
            chunk = chunk.filter(~tail_mask)

        if scl_purity_min > 0.0:
            chunk = chunk.filter(pl.col("scl_purity") >= scl_purity_min).drop("scl_purity")

        chunk = chunk.with_columns([
            ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))).alias("ndvi"),
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.ordinal_day().alias("doy"),
        ]).drop(["B08", "B04"])

        if year_from is not None:
            chunk = chunk.filter(pl.col("year") >= year_from)
        if year_to is not None:
            chunk = chunk.filter(pl.col("year") <= year_to)
        if chunk.is_empty():
            continue

        # Flag sparse years within this chunk
        obs_counts = (
            chunk.group_by(["point_id", "year"])
            .agg(pl.len().alias("_n_obs"))
        )
        chunk = (
            chunk.join(obs_counts, on=["point_id", "year"], how="left")
            .with_columns((pl.col("_n_obs") < min_obs_per_year).alias("is_sparse_year"))
            .drop("_n_obs")
        )

        # Sort within chunk and apply rolling median per (point_id, year)
        chunk = chunk.sort(["point_id", "year", "date"])

        ndvi_arr = chunk["ndvi"].to_numpy(allow_copy=True)
        pid_arr  = chunk["point_id"].to_numpy(allow_copy=True)
        yr_arr   = chunk["year"].to_numpy(allow_copy=True)

        change = np.ones(len(ndvi_arr), dtype=bool)
        change[1:] = (pid_arr[1:] != pid_arr[:-1]) | (yr_arr[1:] != yr_arr[:-1])
        starts = np.flatnonzero(change)
        ends   = np.append(starts[1:], len(ndvi_arr))

        ndvi_smooth = np.empty_like(ndvi_arr)
        for s, e in zip(starts, ends):
            ndvi_smooth[s:e] = median_filter(ndvi_arr[s:e], size=window_rows, mode="nearest")

        result = (
            chunk.with_columns(pl.Series("ndvi_smooth", ndvi_smooth))
            .select(["point_id", "lon", "lat", "date", "year", "month", "doy", "ndvi", "ndvi_smooth", "is_sparse_year"])
        )

        arrow_table = result.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), arrow_table.schema)
        writer.write_table(arrow_table)
        del result, arrow_table, chunk

    if writer is not None:
        writer.close()

    return out_path


def annual_ndvi_curve(
    df: pl.DataFrame,
    smooth_days: int,
    min_obs_per_year: int,
) -> pl.DataFrame:
    """Smoothed per-pixel per-date NDVI, with sparse years flagged.

    Computes NDVI = (B08 - B04) / (B08 + B04), then applies a rolling
    median over a ``smooth_days`` window within each (point_id, year) group.
    Years with fewer than ``min_obs_per_year`` clean observations are flagged
    via an ``is_sparse_year`` boolean column rather than dropped, so callers
    can decide how to handle them.

    Parameters
    ----------
    df:
        Polars observation-level DataFrame — must already have ``year`` and
        ``month`` columns (i.e. output of ``load_and_filter``). Must contain
        ``point_id``, ``date``, ``B08``, ``B04``.
    smooth_days:
        Rolling-median window width in days. Applied within each
        (point_id, year) group sorted by date.
    min_obs_per_year:
        Minimum observations per (point_id, year) to mark the year as
        non-sparse.

    Returns
    -------
    Polars DataFrame with columns:
        ``point_id``, ``date``, ``year``, ``month``, ``doy``,
        ``ndvi``, ``ndvi_smooth``, ``is_sparse_year``.
    """
    # Drop all columns not needed for curve computation before the sort/smooth.
    # The input frame may carry all band columns (~7 GB for 64M rows); retaining
    # them through the sort and numpy extraction doubles peak memory.
    keep = {"point_id", "date", "year", "month", "B08", "B04"}
    drop_cols = [c for c in df.columns if c not in keep]
    if drop_cols:
        df = df.drop(drop_cols)

    df = df.with_columns([
        ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))).alias("ndvi"),
        pl.col("date").dt.ordinal_day().alias("doy"),
    ])

    # Count observations per (point_id, year) to flag sparse years
    obs_counts = (
        df.group_by(["point_id", "year"])
        .agg(pl.len().alias("_n_obs"))
    )
    df = df.join(obs_counts, on=["point_id", "year"], how="left")
    df = df.with_columns([
        (pl.col("_n_obs") < min_obs_per_year).alias("is_sparse_year"),
    ]).drop("_n_obs")

    # Rolling median within each (point_id, year) group, sorted by date.
    # Polars rolling_median operates over a row-count window; convert
    # smooth_days to an approximate row count using median S2 cadence (~5 days).
    #
    # We sort once, then compute the smoothed series entirely in numpy using
    # group-boundary splits — no Polars group infrastructure, no list
    # materialisation, no re-sorting per group. Peak memory stays at ~1× the
    # input frame size.
    window_rows = max(3, smooth_days // 5)
    df = df.sort(["point_id", "year", "date"])

    ndvi_arr = df["ndvi"].to_numpy(allow_copy=True)
    pid_arr = df["point_id"].to_numpy(allow_copy=True)
    yr_arr = df["year"].to_numpy(allow_copy=True)

    # Group boundaries: positions where (point_id, year) changes
    change = np.ones(len(ndvi_arr), dtype=bool)
    change[1:] = (pid_arr[1:] != pid_arr[:-1]) | (yr_arr[1:] != yr_arr[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(ndvi_arr))

    from scipy.ndimage import median_filter

    ndvi_smooth = np.empty_like(ndvi_arr)
    for s, e in zip(starts, ends):
        # scipy median_filter with 'reflect' mode; center=True equivalent
        ndvi_smooth[s:e] = median_filter(ndvi_arr[s:e], size=window_rows, mode="nearest")

    return df.with_columns(
        pl.Series("ndvi_smooth", ndvi_smooth)
    ).select([
        "point_id", "date", "year", "month", "doy",
        "ndvi", "ndvi_smooth", "is_sparse_year",
    ])


# ---------------------------------------------------------------------------
# Site-param loader
# ---------------------------------------------------------------------------

def load_signal_params(loc: object, signal: str) -> object:
    """Return a signal-specific Params instance for a given location.

    Reads the ``signals:`` section from the Location's YAML (exposed as
    ``loc.signal_params``), merges the site overrides with signal defaults,
    and returns a populated ``<SignalClass>.Params`` dataclass.

    Parameters
    ----------
    loc:
        A ``utils.location.Location`` instance.
    signal:
        One of ``'nir_cv'``, ``'rec_p'``, ``'red_edge'``, ``'swir'``,
        ``'flowering'``.

    Returns
    -------
    A ``<SignalClass>.Params`` dataclass instance with site overrides applied.
    """
    # Import here to avoid circular imports at module load time
    from signals.nir_cv import NirCvSignal
    from signals.wet_dry_amp import RecPSignal
    from signals.red_edge import RedEdgeSignal
    from signals.swir import SwirSignal
    from signals.flowering import FloweringSignal
    from signals.integral import NdviIntegralSignal
    from signals import QualityParams

    _signal_map = {
        "nir_cv": NirCvSignal,
        "rec_p": RecPSignal,
        "red_edge": RedEdgeSignal,
        "swir": SwirSignal,
        "flowering": FloweringSignal,
        "ndvi_integral": NdviIntegralSignal,
    }

    if signal not in _signal_map:
        raise ValueError(f"Unknown signal '{signal}'. Choose from: {list(_signal_map)}")

    cls = _signal_map[signal]
    site_overrides: dict = (loc.signal_params or {}).get(signal, {})

    if not site_overrides:
        return cls.Params()

    # Separate quality-related keys from signal-level keys
    quality_keys = {"scl_purity_min", "min_obs_per_year", "min_obs_dry"}
    quality_overrides = {k: v for k, v in site_overrides.items() if k in quality_keys}
    signal_overrides = {k: v for k, v in site_overrides.items() if k not in quality_keys}

    quality = QualityParams(**quality_overrides) if quality_overrides else QualityParams()

    return cls.Params(quality=quality, **signal_overrides)
