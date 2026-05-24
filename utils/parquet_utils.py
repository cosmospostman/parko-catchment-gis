"""utils/parquet_utils.py — Parquet write options, schema helpers, and pixel-sort utilities.

Functions previously in signals/_shared.py that are needed by multiple modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

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
                pa.array([None] * len(tbl), type=field.type),
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
                arrays.append(pa.array([None] * rows, type=field.type))
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
) -> None:
    """Merge S1 shards into one pixel-sorted parquet conforming to combined_schema.

    Each shard is small (≤50 k points × items rows) so sorting all shards
    together via Polars streaming is cheap even for large locations.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Collect all S1 rows into one unsorted temp, then sort with Polars streaming.
    # Total S1 volume is small relative to S2 (no NBAR angles, fewer columns).
    cat_tmp = out_path.with_name(out_path.stem + "_cat.parquet")
    cat_tmp.unlink(missing_ok=True)
    try:
        writer = pq.ParquetWriter(cat_tmp, combined_schema)
        for sp in shard_paths:
            s1_pf = pq.ParquetFile(sp)
            for rg in range(s1_pf.metadata.num_row_groups):
                tbl = s1_pf.read_row_group(rg)
                tbl = _conform_table(tbl, combined_schema)
                writer.write_table(tbl)
        writer.close()
        sort_parquet_by_pixel(cat_tmp, out_path, row_group_size=5_000_000)
    finally:
        cat_tmp.unlink(missing_ok=True)


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
            src_col = pa.array(["S2"] * len(tbl), type=pa.string())
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


def append_s1_to_tile_parquet(
    tile_path: Path,
    bbox_wgs84: list[float],
    start: str,
    end: str,
    collect_s1_fn,
    s1_cache_dir: Path | None = None,
) -> None:
    """Append S1 rows to an existing S2-only tile parquet, in-place and atomically.

    Idempotent: skips the file if it already contains at least one S1 row.

    Strategy for large tiles (avoids sorting the full ~10 GB merged file):
      1. Collect S1 into per-point-shard parquets (small, already streamed).
      2. Sort and merge the S1 shards into one sorted S1 parquet (cheap).
      3. 2-way merge-sort the sorted S2 parquet with the sorted S1 parquet
         into the final output, reading one row-group at a time from each.
    """
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.s1_collector import (
        _resolve_s1_items,
        _collect_s1_shards,
        _DEFAULT_CACHE_DIR as _S1_DEFAULT_CACHE_DIR,
    )

    pf = pq.ParquetFile(tile_path)

    if "source" in pf.schema_arrow.names and "vh" in pf.schema_arrow.names:
        import pyarrow.compute as pc
        source_col = pf.read(columns=["source"]).column("source").combine_chunks()
        if pc.any(pc.equal(source_col, "S1")).as_py():
            return

    combined_schema = _extend_schema(pf.read_row_group(0).schema)
    n_rg = pf.metadata.num_row_groups
    # Collect unique (point_id → lon, lat) across all row-groups.  We read every
    # row-group because small region parquets may have each pixel in a different
    # row-group.  For large tile parquets (same pixels repeated across many date
    # row-groups) the `if pid not in seen` guard short-circuits quickly.
    seen: dict[str, tuple[float, float]] = {}
    for rg_idx in range(n_rg):
        coord_tbl = pf.read_row_group(rg_idx, columns=["point_id", "lon", "lat"])
        for pid, lon, lat in zip(
            coord_tbl.column("point_id").to_pylist(),
            coord_tbl.column("lon").to_pylist(),
            coord_tbl.column("lat").to_pylist(),
        ):
            if pid not in seen:
                seen[pid] = (lon, lat)

    points_for_s1: list[tuple[str, float, float]] = [
        (p, lo, la) for p, (lo, la) in seen.items()
    ]

    resolved_cache = s1_cache_dir if s1_cache_dir is not None else _S1_DEFAULT_CACHE_DIR
    items = _resolve_s1_items(bbox_wgs84, start, end, resolved_cache)

    out_path = tile_path.with_suffix(".merged_tmp.parquet")
    out_path.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="s1_merge_") as _tmp:
            tmp_dir = Path(_tmp)
            if items:
                shard_paths = _collect_s1_shards(
                    out_dir=tmp_dir,
                    items=items,
                    bbox_wgs84=bbox_wgs84,
                    points=points_for_s1,
                    resolved_cache=resolved_cache,
                )
            else:
                shard_paths = []

            if not shard_paths:
                # No S1 data — just add source column and write in-place.
                tmp_nosort = tile_path.with_suffix(".nosort_tmp.parquet")
                writer = pq.ParquetWriter(tmp_nosort, combined_schema)
                for rg_idx in range(n_rg):
                    tbl = pf.read_row_group(rg_idx)
                    tbl = _conform_table(tbl, combined_schema)
                    source_col = pa.array(["S2"] * len(tbl), type=pa.string())
                    tbl = tbl.set_column(tbl.schema.get_field_index("source"), "source", source_col)
                    writer.write_table(tbl)
                writer.close()
                del pf
                tmp_nosort.replace(tile_path)
                return

            # Step 1: sort S1 shards → one sorted S1 parquet (small, cheap)
            s1_sorted = tmp_dir / "s1_sorted.parquet"
            import logging as _logging
            _log = _logging.getLogger(__name__)
            _log.info("append_s1: sorting %d S1 shard(s) ...", len(shard_paths))
            _sort_s1_shards(shard_paths, s1_sorted, combined_schema)
            _log.info("append_s1: S1 sort done")

            del pf  # release file handle before the merge reads tile_path directly

            # Step 2: 2-way merge-sort into out_path.
            # tag_s2_source=True fills source="S2" per row-group during the merge,
            # avoiding a separate full-file copy just to add that column.
            _log.info("append_s1: merging S2 (%s) + S1 → %s ...", tile_path.name, out_path.name)
            _merge_sorted_parquets(tile_path, s1_sorted, out_path, combined_schema, tag_s2_source=True)
            _log.info("append_s1: merge done")

        out_path.replace(tile_path)
    except Exception:
        out_path.unlink(missing_ok=True)
        raise


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

    for i in pair_indices:
        ids_a = set(pl.from_arrow(pf.read_row_groups([i],     columns=["point_id"]))["point_id"].to_list())
        ids_b = set(pl.from_arrow(pf.read_row_groups([i + 1], columns=["point_id"]))["point_id"].to_list())
        if ids_a & ids_b:
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
