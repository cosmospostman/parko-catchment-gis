"""utils/parquet_utils.py — Parquet write options, schema helpers, and pixel-sort utilities.

Functions previously in signals/_shared.py that are needed by multiple modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pyarrow as pa
    import polars as pl


_WRITE_OPTS = dict(
    compression="zstd",
    compression_level=3,
    use_dictionary=["point_id", "item_id", "tile_id"],
    write_statistics=True,
)


# ---------------------------------------------------------------------------
# S1/S2 schema helpers — shared by training_collector and location pipelines
# ---------------------------------------------------------------------------

def _extend_schema(s2_schema: "pa.Schema") -> "pa.Schema":
    """Return s2_schema extended with source, vh, vv columns if not already present."""
    import pyarrow as pa
    extra = []
    names = set(s2_schema.names)
    if "source" not in names:
        extra.append(pa.field("source", pa.string()))
    if "vh" not in names:
        extra.append(pa.field("vh", pa.float32()))
    if "vv" not in names:
        extra.append(pa.field("vv", pa.float32()))
    if not extra:
        return s2_schema
    return pa.schema(list(s2_schema) + extra)


def _conform_table(tbl: "pa.Table", schema: "pa.Schema") -> "pa.Table":
    """Return tbl conformed to schema: add missing columns as null, cast types."""
    import pyarrow as pa
    for field in schema:
        if field.name not in tbl.schema.names:
            tbl = tbl.append_column(
                field,
                pa.nulls(len(tbl), type=field.type),
            )
    arrays = []
    for field in schema:
        col = tbl.column(field.name)
        try:
            col = col.cast(field.type)
        except Exception:
            pass
        arrays.append(col)
    return pa.table(
        {field.name: arrays[i] for i, field in enumerate(schema)},
        schema=schema,
    )


def _s1_df_to_arrow(df_s1: "pl.DataFrame", schema: "pa.Schema") -> "pa.Table":
    """Convert an S1 Polars DataFrame to a PyArrow table conforming to the combined schema.

    S2-only columns (B02…B12, scl_purity, etc.) are filled with null.
    """
    import pyarrow as pa

    s1_cols = set(df_s1.columns)
    rows = len(df_s1)
    arrays = []
    for field in schema:
        if field.name in s1_cols:
            col = df_s1[field.name]
            try:
                # Use to_arrow() — stays in Arrow kernels, avoids Python list round-trip
                arr = col.to_arrow()
                arrays.append(arr.cast(field.type) if arr.type != field.type else arr)
            except Exception:
                arrays.append(pa.nulls(rows, type=field.type))
        else:
            arrays.append(pa.array([None] * rows, type=field.type))
    return pa.table(
        {field.name: arrays[i] for i, field in enumerate(schema)},
        schema=schema,
    )


def _sort_s1_shards(
    shard_paths: list[Path],
    out_path: Path,
    combined_schema: "pa.Schema",
    n_workers: int | None = None,
) -> None:
    """Merge S1 shards into one pixel-sorted parquet conforming to combined_schema.

    Strategy: sort each shard independently (each covers ≤50 k points, so small),
    then merge-sort via Polars scan → sort → sink (Rust engine, ~2 M rows/s,
    streaming so peak RAM stays bounded regardless of total shard count).
    """
    import logging
    import multiprocessing
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    from concurrent.futures import ProcessPoolExecutor, as_completed

    _log = logging.getLogger(__name__)

    if len(shard_paths) == 1:
        _log.info("append_s1: sorting 1 shard (single, no merge needed)")
        sort_parquet_by_pixel(shard_paths[0], out_path, row_group_size=5_000_000)
        _log.info("append_s1: shard sort done")
        return

    # Step 1: sort each shard independently into a sibling .sorted.parquet.
    # Sorts are CPU-bound and independent — parallelise with processes.
    # Must use "spawn" not "fork": Polars holds Rayon thread-pool locks that
    # cause forked children to deadlock immediately on futex_wait.
    sorted_paths: list[Path] = [sp.with_suffix(".sorted.parquet") for sp in shard_paths]
    n_workers = min(len(shard_paths), n_workers or os.cpu_count() or 4)
    _log.info("append_s1: sorting %d shards with %d workers (spawn) ...", len(shard_paths), n_workers)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futs = {
            pool.submit(sort_parquet_by_pixel, sp, sp_sorted, 5_000_000): idx
            for idx, (sp, sp_sorted) in enumerate(zip(shard_paths, sorted_paths))
        }
        done = 0
        for fut in as_completed(futs):
            fut.result()  # re-raise any worker exception
            done += 1
            _log.info("append_s1: shard sort %d/%d done", done, len(shard_paths))

    _log.info("append_s1: all shard sorts done, starting k-way merge ...")

    # Step 2: merge-sort all sorted shards via Polars scan → sort → sink.
    # Polars pushes the sort into its Rust engine (~2.4 M rows/s vs ~200 K rows/s
    # for the Python k-way heap merge), keeping peak RAM to one streaming batch.
    import polars as pl

    tmp_path = out_path.with_suffix(".s1_merge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)

    try:
        (
            pl.scan_parquet([str(p) for p in sorted_paths], missing_columns="insert")
            .with_columns(
                pl.col("point_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("_northing")
            )
            .sort(["_northing", "date"])
            .drop("_northing")
            .sink_parquet(
                str(tmp_path),
                compression="zstd",
                row_group_size=5_000_000,
            )
        )
        tmp_path.replace(out_path)
        _log.info("append_s1: k-way merge done → %s", out_path.name)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        for p in sorted_paths:
            p.unlink(missing_ok=True)


def _merge_sorted_parquets(
    s2_path: Path,
    s1_path: Path,
    out_path: Path,
    combined_schema: "pa.Schema",
    tag_s2_source: bool = False,
) -> None:
    """Streaming 2-way merge of two pixel-sorted parquets into out_path.

    Both inputs must be sorted by (_northing, point_id, date) — the key
    produced by sort_parquet_by_pixel.  This merge reads one row-group at a
    time from each source and only requires O(2 row-groups) of RAM, making it
    safe for S2 tile parquets of any size.

    S2 rows take priority over S1 rows with the same (point_id, date) key
    (S2 is is_s2=True, so its tiebreak digit is 0 vs 1 for S1).

    tag_s2_source:
        If True, fill source="S2" on each S2 row-group as it is loaded,
        eliminating the need for a separate pre-tagging copy of the S2 file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    def _add_sort_key(tbl: pa.Table) -> pa.Table:
        """Add integer _northing column extracted from point_id strings."""
        pid = tbl.column("point_id").combine_chunks()
        # point_id: "{prefix}_{easting}_{northing}" — northing is the last segment.
        # Reverse split with max_splits=2 absorbs any underscores in the prefix into
        # index 0, leaving [prefix, easting, northing] at indices 0/1/2.
        # pc.list_element stays in Arrow kernels; avoids the to_pylist() + list-comprehension.
        parts = pc.split_pattern(pid, "_", max_splits=2, reverse=True)
        northings = pc.list_element(parts, 2).cast(pa.int32())
        return tbl.append_column(pa.field("_northing", pa.int32()), northings)

    # Pre-extracted key arrays per loaded block — populated in _read_block, used in
    # _row_key.  Avoids per-probe random-access PyArrow column lookups inside binary search.
    _key_arrays: dict[str, tuple] = {}  # block id → (northings[], pids[], dates[], tiebreak)

    pf_s2 = pq.ParquetFile(s2_path)
    pf_s1 = pq.ParquetFile(s1_path)
    n_rg_s2 = pf_s2.metadata.num_row_groups
    n_rg_s1 = pf_s1.metadata.num_row_groups

    dict_cols = {"point_id", "item_id", "tile_id"}
    writer = pq.ParquetWriter(
        out_path, combined_schema, compression="zstd",
        use_dictionary=[c for c in combined_schema.names if c in dict_cols],
        write_statistics=True,
    )

    FLUSH_ROWS = 5_000_000
    buf: list[pa.Table] = []
    buf_rows = 0

    def _flush():
        nonlocal buf_rows
        if not buf:
            return
        writer.write_table(pa.concat_tables(buf))
        buf.clear()
        buf_rows = 0

    def _emit(tbl: pa.Table) -> None:
        nonlocal buf_rows
        # Drop _northing before writing
        if "_northing" in tbl.schema.names:
            tbl = tbl.remove_column(tbl.schema.get_field_index("_northing"))
        tbl = _conform_table(tbl, combined_schema)
        buf.append(tbl)
        buf_rows += len(tbl)
        if buf_rows >= FLUSH_ROWS:
            _flush()

    # State for each stream: current block (Table with _northing) and row offset
    def _read_block(pf: pq.ParquetFile, rg: int, is_s2: bool) -> pa.Table | None:
        if rg >= (n_rg_s2 if is_s2 else n_rg_s1):
            _key_arrays.pop(("s2" if is_s2 else "s1"), None)
            return None
        tbl = pf.read_row_group(rg)
        tbl = _conform_table(tbl, combined_schema)
        if is_s2 and tag_s2_source:
            src_col = pa.repeat("S2", len(tbl)).cast(pa.string())
            tbl = tbl.set_column(tbl.schema.get_field_index("source"), "source", src_col)
        tbl = _add_sort_key(tbl)
        # Pre-extract key columns into Python lists once per block load.
        # Binary search probes these lists with O(1) index — no Arrow overhead per probe.
        key = "s2" if is_s2 else "s1"
        _key_arrays[key] = (
            tbl.column("_northing").to_pylist(),
            tbl.column("point_id").to_pylist(),
            tbl.column("date").to_pylist(),
            0 if is_s2 else 1,
        )
        return tbl

    def _row_key(tbl: pa.Table, row: int, is_s2: bool) -> tuple:
        northings, pids, dates, tb = _key_arrays["s2" if is_s2 else "s1"]
        return (northings[row], pids[row], dates[row], tb)

    rg_s2 = rg_s1 = 0
    pos_s2 = pos_s1 = 0
    blk_s2 = _read_block(pf_s2, rg_s2, True)
    blk_s1 = _read_block(pf_s1, rg_s1, False)

    # Row-by-row merge is too slow for 200M rows. Instead: find the boundary
    # in the current S1 block that falls before the start of the next S2 block,
    # and emit entire S2 row-groups at a time (the common case is S2 rows are
    # much more dense and already sorted — we only need to interleave S1).
    #
    # Algorithm:
    #   While both streams have data:
    #     key_s1_start = first row key in current S1 block
    #     Emit all S2 rows with key < key_s1_start (entire row-groups at a time)
    #     Emit all S1 rows with key <= next S2 row key (or all remaining S1 if S2 exhausted)
    #   Drain whichever stream remains.

    while blk_s2 is not None and blk_s1 is not None:
        # Key of first S1 row in current block
        k_s1 = _row_key(blk_s1, pos_s1, False)

        # Emit S2 rows that are strictly before the first S1 row.
        # Optimisation: if the entire current S2 block ends before k_s1,
        # emit the whole block without row-by-row comparison.
        while blk_s2 is not None:
            last_s2_key = _row_key(blk_s2, len(blk_s2) - 1, True)
            if last_s2_key < k_s1:
                # Whole S2 block goes before any S1 row — emit it in one shot
                _emit(blk_s2.slice(pos_s2))
                pos_s2 = 0
                rg_s2 += 1
                blk_s2 = _read_block(pf_s2, rg_s2, True)
            else:
                break

        if blk_s2 is None:
            break

        # Find cut point in S2 block: first row with key >= k_s1
        # Binary search on the sorted block
        lo, hi = pos_s2, len(blk_s2)
        while lo < hi:
            mid = (lo + hi) // 2
            if _row_key(blk_s2, mid, True) < k_s1:
                lo = mid + 1
            else:
                hi = mid
        cut = lo  # rows [pos_s2, cut) go before S1

        if cut > pos_s2:
            _emit(blk_s2.slice(pos_s2, cut - pos_s2))
            pos_s2 = cut
            if pos_s2 >= len(blk_s2):
                rg_s2 += 1
                pos_s2 = 0
                blk_s2 = _read_block(pf_s2, rg_s2, True)

        # Now emit S1 rows up to the current S2 row's key
        k_s2 = _row_key(blk_s2, pos_s2, True) if blk_s2 is not None else None
        while blk_s1 is not None:
            k_s1_cur = _row_key(blk_s1, pos_s1, False)
            if k_s2 is not None and k_s1_cur >= k_s2:
                break
            # Find cut in S1 block
            lo, hi = pos_s1, len(blk_s1)
            while lo < hi:
                mid = (lo + hi) // 2
                if _row_key(blk_s1, mid, False) < k_s2:
                    lo = mid + 1
                else:
                    hi = mid
            cut_s1 = lo
            if cut_s1 > pos_s1:
                _emit(blk_s1.slice(pos_s1, cut_s1 - pos_s1))
                pos_s1 = cut_s1
            if pos_s1 >= len(blk_s1):
                rg_s1 += 1
                pos_s1 = 0
                blk_s1 = _read_block(pf_s1, rg_s1, False)
            else:
                break

    # Drain remaining S2
    while blk_s2 is not None:
        _emit(blk_s2.slice(pos_s2))
        pos_s2 = 0
        rg_s2 += 1
        blk_s2 = _read_block(pf_s2, rg_s2, True)

    # Drain remaining S1
    while blk_s1 is not None:
        _emit(blk_s1.slice(pos_s1))
        pos_s1 = 0
        rg_s1 += 1
        blk_s1 = _read_block(pf_s1, rg_s1, False)

    _flush()
    writer.close()


def merge_strips(
    strip_paths: "list[Path]",
    out_path: Path,
) -> Path:
    """N-way merge of already-sorted strip parquets into a single output parquet.

    Each strip must already be pixel-sorted (as produced by merge_tile or
    sort_parquet_by_pixel).  Strips are assumed to cover non-overlapping northing
    bands (south-to-north order), so this is primarily a concatenation with a
    final sort pass to handle any overlap at strip boundaries.

    Idempotent: skips if out_path already exists with the correct total row count.
    Atomic write via .strips_tmp suffix → rename.
    """
    import pyarrow.parquet as pq
    import polars as pl

    if not strip_paths:
        raise ValueError("merge_strips: strip_paths is empty")

    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in strip_paths)

    if out_path.exists():
        existing = pq.ParquetFile(out_path).metadata.num_rows
        if existing == total_rows:
            logger.info("merge_strips: %s already up-to-date (%d rows) — skipping", out_path.name, existing)
            return out_path
        logger.info(
            "merge_strips: %s exists with %d rows but expected %d — rebuilding",
            out_path.name, existing, total_rows,
        )

    tmp_path = out_path.with_suffix(".strips_tmp.parquet")
    tmp_path.unlink(missing_ok=True)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        (
            pl.scan_parquet([str(p) for p in strip_paths], missing_columns="insert")
            .with_columns(
                pl.col("point_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("_northing")
            )
            .sort(["_northing", "date"])
            .drop("_northing")
            .sink_parquet(
                str(tmp_path),
                compression="zstd",
                row_group_size=5_000_000,
            )
        )
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("merge_strips: wrote %s (%d rows from %d strips)", out_path.name, total_rows, len(strip_paths))
    return out_path


def merge_tile(
    s2_path: Path,
    s1_path: "Path | None",
    out_path: Path,
) -> Path:
    """Merge an S2 parquet and optional S1 parquet into a combined tile parquet.

    No S1: copies s2_path → out_path tagging source="S2".
    With S1: 2-way pixel-sorted merge via _merge_sorted_parquets.
    Atomic write via .merge_tmp → rename.  Idempotent: skips if out_path
    already exists with the correct row count (s2 rows + s1 rows).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    s2_rows = pq.ParquetFile(s2_path).metadata.num_rows
    s1_rows = pq.ParquetFile(s1_path).metadata.num_rows if s1_path and s1_path.exists() else 0
    expected = s2_rows + s1_rows

    if out_path.exists():
        existing_rows = pq.ParquetFile(out_path).metadata.num_rows
        if existing_rows == expected:
            logger.info("merge_tile: %s already up-to-date (%d rows) — skipping", out_path.name, existing_rows)
            return out_path
        logger.info(
            "merge_tile: %s exists with %d rows but expected %d — rebuilding",
            out_path.name, existing_rows, expected,
        )

    pf_s2 = pq.ParquetFile(s2_path)
    combined_schema = _extend_schema(pf_s2.schema_arrow)
    tmp_path = out_path.with_name(out_path.stem + ".merge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)

    try:
        if not s1_path or not s1_path.exists() or s1_rows == 0:
            # No S1 — copy S2 with source="S2" tagged
            n_rg = pf_s2.metadata.num_row_groups
            writer = pq.ParquetWriter(tmp_path, combined_schema)
            for rg_idx in range(n_rg):
                tbl = pf_s2.read_row_group(rg_idx)
                tbl = _conform_table(tbl, combined_schema)
                src_col = pa.repeat("S2", len(tbl)).cast(pa.string())
                tbl = tbl.set_column(tbl.schema.get_field_index("source"), "source", src_col)
                writer.write_table(tbl)
            writer.close()
        else:
            _merge_sorted_parquets(s2_path, s1_path, tmp_path, combined_schema, tag_s2_source=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("merge_tile: wrote %s (%d rows)", out_path.name, expected)
    return out_path


# ---------------------------------------------------------------------------
# Schema optimisation
# ---------------------------------------------------------------------------

def _optimise_schema(tbl: "pa.Table") -> "pa.Table":
    """Cast lon/lat → float32 and date → date32 to reduce parquet file size."""
    import pyarrow as pa
    for col in ("lon", "lat"):
        if col in tbl.schema.names:
            tbl = tbl.set_column(
                tbl.schema.get_field_index(col), col,
                tbl.column(col).cast(pa.float32()),
            )
    if "date" in tbl.schema.names:
        tbl = tbl.set_column(
            tbl.schema.get_field_index("date"), "date",
            tbl.column("date").cast(pa.date32()),
        )
    return tbl


def is_pixel_sorted(path: Path, n_check: int = 20) -> bool:
    """Return True if ``path`` is pixel-sorted (no point_id overlap between adjacent row groups).

    Checks ``n_check`` adjacent pairs sampled evenly across the file. Files
    written by this module are always sorted on write, so this is a safety-net
    check rather than the primary enforcement. n_check=20 gives one sample per
    ~70 row groups on a 1380-rg file, which is sufficient to catch gross
    violations while remaining fast (< 1 s on typical tile parquets).

    Note: point_id strings like ``px_{xi}_{yi}`` are NOT in lexicographic order
    so min/max Parquet statistics cannot be used for this check.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    if n_rg <= 1:
        return True

    n_pairs = n_rg - 1
    if n_pairs <= n_check:
        pair_indices = list(range(n_pairs))
    else:
        step = n_pairs / n_check
        pair_indices = sorted({0, n_pairs - 1} | {round(i * step) for i in range(n_check)})
        pair_indices = [i for i in pair_indices if i < n_pairs]

    import pyarrow.compute as pc
    for i in pair_indices:
        ids_a = pc.unique(pf.read_row_group(i,     columns=["point_id"]).column("point_id").combine_chunks())
        ids_b = pc.unique(pf.read_row_group(i + 1, columns=["point_id"]).column("point_id").combine_chunks())
        if pc.any(pc.is_in(ids_a, value_set=ids_b)).as_py():
            return False
    return True


def ensure_pixel_sorted(path: Path, row_group_size: int = 5_000_000) -> Path:
    """Return a pixel-sorted version of ``path``, sorting it first if needed.

    If the parquet is already pixel-sorted the original path is returned
    unchanged. Otherwise a ``<stem>-by-pixel.parquet`` sibling is written (or
    reused if it already exists) and its path is returned.
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
    ram_budget_gb: float = 8.0,  # kept for call-site compatibility, unused
    read_workers: int = 6,       # kept for call-site compatibility, unused
) -> None:
    """Write a copy of ``src`` sorted by ``(point_id, date, scl_purity desc)``.

    Uses Polars' streaming engine (scan → sort → sink) which reads the file once
    and spills to disk only if the sort exceeds available RAM — no manual multi-pass
    bucketing required.  A PyArrow rewrite pass is applied afterwards to restore
    dictionary encoding on string columns and the float32/date32 schema optimisations.
    """
    import pyarrow.parquet as pq

    tmp = dst.with_suffix(".sorting.parquet")
    tmp.unlink(missing_ok=True)
    try:
        # Derive a northing sort key from point_id ("{prefix}_{easting}_{northing}") so
        # pixels in the same geographic row are co-located in output row groups.
        # list.get(-1) extracts the last "_"-delimited segment (northing), which is
        # correct for both "px_0042_0031" and "rupert_ck_presence_1_0042_0031".
        # For IDs with no underscores (edge-case test data) we fall back to 0.
        schema_cols = pl.scan_parquet(src).collect_schema().names()
        cast_exprs = [
            *(
                [pl.col("lon").cast(pl.Float32), pl.col("lat").cast(pl.Float32)]
                if "lon" in schema_cols else []
            ),
            *(
                [pl.col("date").cast(pl.Date)]
                if "date" in schema_cols else []
            ),
            pl.col("point_id").str.split("_").list.get(-1).cast(pl.Int32, strict=False).fill_null(0).alias("_northing"),
        ]
        sort_cols = ["_northing", "point_id", "date", "scl_purity"] if "scl_purity" in schema_cols \
            else ["_northing", "point_id", "date"]
        sort_desc = [False] * (len(sort_cols) - 1) + ([True] if "scl_purity" in schema_cols else [False])
        (
            pl.scan_parquet(src)
            .with_columns(cast_exprs)
            .sort(sort_cols, descending=sort_desc)
            .drop("_northing")
            .sink_parquet(
                tmp,
                compression="zstd",
                compression_level=3,
                row_group_size=row_group_size,
                statistics=True,
            )
        )

        # Rewrite with dictionary encoding on string columns — sink_parquet doesn't
        # expose this, but it meaningfully reduces file size for point_id/item_id/tile_id.
        dict_cols = {"point_id", "item_id", "tile_id"}
        pf = pq.ParquetFile(tmp)
        schema = pf.schema_arrow
        writer = pq.ParquetWriter(
            dst, schema, compression="zstd",
            use_dictionary=[c for c in schema.names if c in dict_cols],
            write_statistics=True,
        )
        for i in range(pf.metadata.num_row_groups):
            writer.write_table(pf.read_row_group(i))
        writer.close()
        tmp.unlink(missing_ok=True)

    except Exception:
        tmp.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        raise
