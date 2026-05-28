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
# Canonical combined S2+S1 schema for VM-facing code paths.
# Hardcoded so proxy/server.py and collect_s1_for_tile(points=...) have no
# dependency on a live parquet file for schema derivation.
# Matches the output of _extend_schema() applied to a pixel_collector output.
# ---------------------------------------------------------------------------

def _make_combined_pixel_schema() -> "pa.Schema":
    import pyarrow as pa
    return pa.schema([
        pa.field("point_id",    pa.string()),
        pa.field("lon",         pa.float32()),
        pa.field("lat",         pa.float32()),
        pa.field("date",        pa.date32()),
        pa.field("item_id",     pa.string()),
        pa.field("tile_id",     pa.string()),
        pa.field("B02",         pa.uint16()),
        pa.field("B03",         pa.uint16()),
        pa.field("B04",         pa.uint16()),
        pa.field("B05",         pa.uint16()),
        pa.field("B06",         pa.uint16()),
        pa.field("B07",         pa.uint16()),
        pa.field("B08",         pa.uint16()),
        pa.field("B8A",         pa.uint16()),
        pa.field("B11",         pa.uint16()),
        pa.field("B12",         pa.uint16()),
        pa.field("scl_purity",  pa.int8()),
        pa.field("scl",         pa.int8()),
        pa.field("aot",         pa.uint8()),
        pa.field("view_zenith", pa.uint8()),
        pa.field("sun_zenith",  pa.uint8()),
        pa.field("source",      pa.string()),
        pa.field("vh",          pa.float32()),
        pa.field("vv",          pa.float32()),
        pa.field("orbit",       pa.string()),
    ])


COMBINED_PIXEL_SCHEMA: "pa.Schema" = _make_combined_pixel_schema()


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
            arrays.append(pa.nulls(rows, type=field.type))
    return pa.table(
        {field.name: arrays[i] for i, field in enumerate(schema)},
        schema=schema,
    )


def _kway_merge_parquets(
    input_paths: list[Path],
    out_path: Path,
    schema: "pa.Schema",
) -> None:
    """Streaming k-way merge of northing-sorted parquets into out_path.

    Each input file must be sorted by northing (last '_'-delimited segment of
    point_id).  Streams one row-group at a time through a min-heap — peak RAM
    is O(k * row_group_size), constant regardless of total dataset size.

    Northing values are extracted once per row-group via vectorised NumPy string
    ops; run-end finding uses np.searchsorted — no per-row Python loops.

    Heap entries: (northing: int, date_str: str, cursor_id: int,
                   row: int, batch: pa.Table, ns: np.ndarray)
    ns is the int32 northing array for the full batch, computed once on load.
    cursor_id breaks ties before reaching batch/ns, so no Arrow comparison occurs.
    """
    import heapq
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_names = schema.names

    def _cast(tbl: "pa.Table") -> "pa.Table":
        cols = {}
        for field in schema:
            if field.name in tbl.schema.names:
                arr = tbl.column(field.name)
                cols[field.name] = arr.cast(field.type, safe=False) if arr.type != field.type else arr
            else:
                cols[field.name] = pa.nulls(len(tbl), type=field.type)
        return pa.table(cols, schema=schema)

    def _northings_np(tbl: "pa.Table") -> "np.ndarray":
        # Extract the last '_'-delimited segment of every point_id as an int32
        # NumPy array — fully vectorised, no Python loop over rows.
        import pyarrow.compute as pc
        # split_pattern(reverse=True, max_splits=1) on "px_0003" → ["px", "0003"]
        parts = pc.split_pattern(
            tbl.column("point_id"), pattern="_", reverse=True, max_splits=1,
        )
        # list_slice([1,2)) extracts the last segment (the northing number).
        suffix = pc.list_slice(parts, 1, 2).combine_chunks().flatten()
        return pc.cast(suffix, pa.int32()).to_numpy(zero_copy_only=False)

    class _Cursor:
        __slots__ = ("_pf", "_rg", "_n_rg", "_cols")
        def __init__(self, path: Path) -> None:
            self._pf   = pq.ParquetFile(path)
            self._rg   = 0
            self._n_rg = self._pf.metadata.num_row_groups
            file_names = set(self._pf.schema_arrow.names)
            self._cols = [c for c in out_names if c in file_names]
        def next_batch(self) -> "tuple[pa.Table, np.ndarray] | None":
            while self._rg < self._n_rg:
                tbl = self._pf.read_row_group(self._rg, columns=self._cols)
                self._rg += 1
                tbl = _cast(tbl)
                if len(tbl) == 0:
                    continue
                ns = _northings_np(tbl)
                return tbl, ns
            return None

    heap: list[tuple] = []
    cursors: list[_Cursor] = []
    for path in input_paths:
        cur = _Cursor(path)
        result = cur.next_batch()
        if result is None:
            continue
        batch, ns = result
        cid = len(cursors)
        cursors.append(cur)
        heapq.heappush(heap, (int(ns[0]), str(batch.column("date")[0].as_py()), cid, 0, batch, ns))

    tmp_path = out_path.with_suffix(".kmerge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)

    ROW_GROUP = 250_000
    out_batches: list["pa.Table"] = []
    out_rows = 0
    writer: "pq.ParquetWriter | None" = None

    try:
        def _flush() -> None:
            nonlocal out_batches, out_rows, writer
            if not out_batches:
                return
            chunk = pa.concat_tables(out_batches)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, schema=schema, **_WRITE_OPTS)
            writer.write_table(chunk)
            out_batches = []
            out_rows = 0

        while heap:
            n, dt, cid, row, batch, ns = heapq.heappop(heap)
            # Find end of current northing run via binary search — O(log batch_size).
            # Slice from `row` so the search is within the unprocessed suffix.
            end = row + int(np.searchsorted(ns[row:], n + 1, side="left"))
            out_batches.append(batch.slice(row, end - row))
            out_rows += end - row
            if out_rows >= ROW_GROUP:
                _flush()
            if end < len(ns):
                next_n = int(ns[end])
                next_dt = str(batch.column("date")[end].as_py())
                heapq.heappush(heap, (next_n, next_dt, cid, end, batch, ns))
            else:
                result = cursors[cid].next_batch()
                if result is not None:
                    nb, nns = result
                    heapq.heappush(heap, (int(nns[0]), str(nb.column("date")[0].as_py()), cid, 0, nb, nns))

        _flush()
    finally:
        if writer is not None:
            writer.close()

    tmp_path.replace(out_path)


def _sort_s1_shards(
    shard_paths: list[Path],
    out_path: Path,
    combined_schema: "pa.Schema",
    n_workers: int | None = None,
) -> None:
    """Merge S1 shards into one northing-sorted parquet conforming to combined_schema.

    Points are passed to _collect_s1_shards in northing order (from make_strip_points),
    so each shard's rows are already northing-sorted within each date.  Shards cover
    non-overlapping northing ranges, so a k-way merge produces a fully sorted output
    without any per-shard sort step.
    """
    _log = logging.getLogger(__name__)
    _log.info("merging %d S1 shards → %s ...", len(shard_paths), out_path.name)
    _kway_merge_parquets(shard_paths, out_path, combined_schema)
    _log.info("merge done → %s", out_path.name)


def _merge_sorted_parquets(
    s2_path: Path,
    s1_path: Path,
    out_path: Path,
    combined_schema: "pa.Schema",
    tag_s2_source: bool = False,
    memory_limit_gb: int = 16,
) -> None:
    """2-way k-merge of northing-sorted S2 and S1 parquets into out_path.

    Both inputs are already northing-sorted, so this is a streaming merge —
    no full sort needed.  Peak RAM is O(one row-group each).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    s2_rows = pq.ParquetFile(s2_path).metadata.num_rows
    s1_rows = pq.ParquetFile(s1_path).metadata.num_rows
    logger.info(
        "merge_tile: k-merging %s (%s S2 rows + %s S1 rows) ...",
        out_path.name, f"{s2_rows:,}", f"{s1_rows:,}",
    )

    if tag_s2_source:
        # Write a source-tagged copy of S2 into a sibling tmp, then k-merge.
        s2_tagged = out_path.with_suffix(".s2_tagged_tmp.parquet")
        s2_tagged.unlink(missing_ok=True)
        try:
            pf = pq.ParquetFile(s2_path)
            s2_schema = pf.schema_arrow
            if "source" not in s2_schema.names:
                s2_schema = s2_schema.append(pa.field("source", pa.string()))
            w = pq.ParquetWriter(s2_tagged, s2_schema, **_WRITE_OPTS)
            try:
                for rg in range(pf.metadata.num_row_groups):
                    blk = pf.read_row_group(rg)
                    src_col = pa.repeat("S2", len(blk))
                    if "source" in blk.schema.names:
                        blk = blk.set_column(blk.schema.get_field_index("source"), "source", src_col)
                    else:
                        blk = blk.append_column(pa.field("source", pa.string()), src_col)
                    w.write_table(_conform_table(blk, s2_schema))
            finally:
                w.close()
            _kway_merge_parquets([s2_tagged, s1_path], out_path, combined_schema)
        finally:
            s2_tagged.unlink(missing_ok=True)
    else:
        _kway_merge_parquets([s2_path, s1_path], out_path, combined_schema)


def merge_strips(
    strip_paths: "list[Path]",
    out_path: Path,
) -> Path:
    """Concatenate already-sorted strip parquets into a single output parquet.

    Strips cover non-overlapping northing bands in south-to-north order, so
    concatenation preserves pixel-sort order.  Row groups are copied verbatim
    — no re-sort, O(1 row group) peak RAM.

    Idempotent: skips if out_path already exists with the correct total row count.
    Atomic write via .strips_tmp suffix → rename.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

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
        # Determine schema from first strip (all strips share identical schema)
        first_pf = pq.ParquetFile(strip_paths[0])
        schema = first_pf.schema_arrow
        writer = pq.ParquetWriter(
            tmp_path, schema, compression="zstd",
            use_dictionary=["point_id", "item_id", "tile_id"],
            write_statistics=True,
        )
        for p in strip_paths:
            pf = pq.ParquetFile(p)
            for rg in range(pf.metadata.num_row_groups):
                writer.write_table(pf.read_row_group(rg))
        writer.close()
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

    import pyarrow as pa

    tmp_path = out_path.with_name(out_path.stem + ".merge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)

    try:
        if not s1_path or not s1_path.exists() or s1_rows == 0:
            # No S1 — copy S2 row-groups verbatim, tagging source="S2"
            s2_pf = pq.ParquetFile(s2_path)
            out_schema = _extend_schema(s2_pf.schema_arrow)
            writer = pq.ParquetWriter(
                tmp_path, out_schema, compression="zstd",
                use_dictionary=["point_id", "item_id", "tile_id"],
                write_statistics=True,
            )
            try:
                for rg in range(s2_pf.metadata.num_row_groups):
                    blk = s2_pf.read_row_group(rg)
                    src_col = pa.repeat("S2", len(blk))
                    if "source" in blk.schema.names:
                        blk = blk.set_column(blk.schema.get_field_index("source"), "source", src_col)
                    else:
                        blk = blk.append_column(pa.field("source", pa.string()), src_col)
                    writer.write_table(_conform_table(blk, out_schema))
            finally:
                writer.close()
        else:
            combined_schema = _extend_schema(pq.ParquetFile(s2_path).schema_arrow)
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
    _skip_dict_rewrite: bool = False,
) -> None:
    """Write a copy of ``src`` sorted by ``(point_id, date, scl_purity desc)``.

    Uses Polars' streaming engine (scan → sort → sink) which reads the file once
    and spills to disk only if the sort exceeds available RAM — no manual multi-pass
    bucketing required.  A PyArrow rewrite pass is applied afterwards to restore
    dictionary encoding on string columns and the float32/date32 schema optimisations.

    _skip_dict_rewrite: skip the second-pass PyArrow rewrite.  Use this when dst
    is a transient intermediate file that will be immediately consumed and deleted
    (e.g. shard pre-sort before a Polars merge), saving ~1 s/shard.
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

        sort_dst = dst if _skip_dict_rewrite else tmp
        (
            pl.scan_parquet(src)
            .with_columns(cast_exprs)
            .sort(sort_cols, descending=sort_desc)
            .drop("_northing")
            .sink_parquet(
                sort_dst,
                compression="zstd",
                compression_level=3,
                row_group_size=row_group_size,
                statistics=True,
            )
        )

        if not _skip_dict_rewrite:
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
