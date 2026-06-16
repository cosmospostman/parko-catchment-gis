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


def _arrow_to_polars(arrow_type: "pa.DataType") -> "pl.PolarsDataType":
    """Map a PyArrow type to the equivalent Polars dtype."""
    import pyarrow as pa
    _map = {
        pa.string():  pl.String,
        pa.float32(): pl.Float32,
        pa.float64(): pl.Float64,
        pa.int8():    pl.Int8,
        pa.int16():   pl.Int16,
        pa.int32():   pl.Int32,
        pa.int64():   pl.Int64,
        pa.uint8():   pl.UInt8,
        pa.uint16():  pl.UInt16,
        pa.uint32():  pl.UInt32,
        pa.uint64():  pl.UInt64,
        pa.date32():  pl.Date,
        pa.bool_():   pl.Boolean,
    }
    return _map.get(arrow_type, pl.String)


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
    """Sort-merge parquets into out_path by northing via Polars streaming.

    Uses Polars' streaming engine (scan_parquet → sort → sink_parquet).
    Handles mixed schemas by null-filling columns absent in any input file.
    Spills to disk automatically if working set exceeds RAM.
    """
    out_col_names = schema.names
    pl_dtype = {f.name: _arrow_to_polars(f.type) for f in schema}
    str_paths = [str(p) for p in input_paths]

    lf = pl.scan_parquet(str_paths, glob=False)
    existing = lf.collect_schema().names()
    exprs = [
        pl.col(name) if name in existing
        else pl.lit(None, dtype=pl_dtype[name]).alias(name)
        for name in out_col_names
    ]
    lf = (
        lf.select(exprs)
        .with_columns(
            pl.col("point_id").str.split("_").list.get(-1)
            .cast(pl.Int32, strict=False).fill_null(0)
            .alias("_northing")
        )
        .sort(["_northing", "date"])
        .drop("_northing")
    )

    tmp_path = out_path.with_suffix(".kmerge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)
    lf.sink_parquet(
        tmp_path,
        compression="uncompressed",
        row_group_size=250_000,
        statistics=False,
        engine="streaming",
    )
    tmp_path.replace(out_path)


def _sort_s1_shards(
    shard_paths: list[Path],
    out_path: Path,
    combined_schema: "pa.Schema",
    n_workers: int | None = None,
) -> None:
    """Concat S1 shards and sort the result by **northing (yi) ascending**.

    The S1 shards are written *date-major* (rows accumulated in item submission
    order — see _collect_s1_shards), NOT northing-sorted.  merge_scenes streams
    its inputs in northing-band passes and assumes each input file is sorted by
    yi ascending; feeding it date-major data makes it silently drop every S1 row
    whose yi is below the current band cursor (all acquisitions after the first
    collapse to a thin band — the S1_TRUNC defect).  So we must materialise a
    real northing sort here, keyed on the yi encoded in the point_id suffix.

    Uses Polars' streaming engine so peak RAM stays bounded.
    """
    import polars as pl

    _log = logging.getLogger(__name__)
    _log.info("sorting %d S1 shards by northing → %s ...", len(shard_paths), out_path.name)

    tmp_path = out_path.with_suffix(".shards_tmp.parquet")
    tmp_path.unlink(missing_ok=True)
    (
        pl.scan_parquet([str(p) for p in shard_paths])
        # yi (northing index) is the trailing _\d+ group of point_id; merge_scenes
        # extracts the same field, so sort on it to match its band expectation.
        .with_columns(
            pl.col("point_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("_yi")
        )
        .sort(["_yi", "date"])
        .drop("_yi")
        .sink_parquet(str(tmp_path), compression="zstd", compression_level=3)
    )
    tmp_path.replace(out_path)
    _log.info("sort done → %s", out_path.name)


def _merge_sorted_parquets(
    s2_path: Path,
    s1_path: Path,
    out_path: Path,
    combined_schema: "pa.Schema",
    tag_s2_source: bool = False,
    memory_limit_gb: int = 16,
) -> None:
    """2-way sort-merge of S2 and S1 parquets into out_path via Polars streaming."""
    import pyarrow.parquet as pq

    s2_rows = pq.ParquetFile(s2_path).metadata.num_rows
    s1_rows = pq.ParquetFile(s1_path).metadata.num_rows
    logger.info(
        "merge_tile: merging %s (%s S2 rows + %s S1 rows) ...",
        out_path.name, f"{s2_rows:,}", f"{s1_rows:,}",
    )

    out_col_names = combined_schema.names
    pl_dtype = {f.name: _arrow_to_polars(f.type) for f in combined_schema}

    def _scan_with_source(path: Path, source_val: str | None) -> pl.LazyFrame:
        lf = pl.scan_parquet(str(path), glob=False)
        existing = lf.collect_schema().names()
        exprs = []
        for name in out_col_names:
            if name == "source" and source_val is not None:
                exprs.append(pl.lit(source_val).alias("source"))
            elif name in existing:
                exprs.append(pl.col(name))
            else:
                exprs.append(pl.lit(None, dtype=pl_dtype[name]).alias(name))
        return lf.select(exprs)

    s2_lf = _scan_with_source(s2_path, "S2" if tag_s2_source else None)
    s1_lf = _scan_with_source(s1_path, None)

    tmp_path = out_path.with_suffix(".kmerge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)
    (
        pl.concat([s2_lf, s1_lf])
        .with_columns(
            pl.col("point_id").str.split("_").list.get(-1)
            .cast(pl.Int32, strict=False).fill_null(0)
            .alias("_northing")
        )
        .sort(["_northing", "date"])
        .drop("_northing")
        .sink_parquet(
            tmp_path,
            compression="uncompressed",
            row_group_size=250_000,
            statistics=False,
            engine="streaming",
        )
    )
    tmp_path.replace(out_path)


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
            for batch in pf.iter_batches():
                writer.write_table(pa.Table.from_batches([batch]))
        writer.close()
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("merge_strips: wrote %s (%d rows from %d strips)", out_path.name, total_rows, len(strip_paths))
    return out_path


def merge_chunks(
    chunk_paths: "list[Path]",
    out_path: Path,
) -> Path:
    """Concatenate chunk parquets (in row-major order) into a single output parquet.

    Chunks must be sorted by (chunk_row, chunk_col) before calling.
    Within each row-band northing order is preserved; row-bands are ordered
    north-to-south.  Delegates to merge_strips — implementation is identical.
    """
    return merge_strips(chunk_paths, out_path)


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
    """Return True if ``path`` is pixel-sorted (all obs for each pixel are contiguous).

    Two checks:
    1. Adjacent-pair check: no point_id overlap between n_check sampled adjacent
       row group pairs (necessary condition for pixel-sorted).
    2. Distant-pair check: row groups 0 and n_rg//2 share at least one point_id.
       In a pixel-sorted file the same pixels appear throughout; in a date-sorted
       file each row group is a disjoint spatial slice so distant groups never share
       pixels — this distinguishes the two layouts.

    Note: point_id strings like ``px_{xi}_{yi}`` are NOT in lexicographic order
    so min/max Parquet statistics cannot be used for this check.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    if n_rg <= 1:
        return True

    # Adjacent-pair check
    n_pairs = n_rg - 1
    if n_pairs <= n_check:
        pair_indices = list(range(n_pairs))
    else:
        step = n_pairs / n_check
        pair_indices = sorted({0, n_pairs - 1} | {round(i * step) for i in range(n_check)})
        pair_indices = [i for i in pair_indices if i < n_pairs]

    for i in pair_indices:
        ids_a = pc.unique(pf.read_row_group(i,     columns=["point_id"]).column("point_id").combine_chunks())
        ids_b = pc.unique(pf.read_row_group(i + 1, columns=["point_id"]).column("point_id").combine_chunks())
        if pc.any(pc.is_in(ids_a, value_set=ids_b)).as_py():
            return False

    # Distant-pair check: pixel-sorted files have the same pixel set throughout;
    # date-sorted files have disjoint spatial slices so distant groups never overlap.
    if n_rg >= 4:
        ids_first = pc.unique(pf.read_row_group(0,        columns=["point_id"]).column("point_id").combine_chunks())
        ids_mid   = pc.unique(pf.read_row_group(n_rg // 2, columns=["point_id"]).column("point_id").combine_chunks())
        if not pc.any(pc.is_in(ids_first, value_set=ids_mid)).as_py():
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
    sequential_read: bool = False,
) -> None:
    """Write a copy of ``src`` sorted by ``(point_id, date, scl_purity desc)``.

    Uses Polars' streaming engine (scan → sort → sink) which reads the file once
    and spills to disk only if the sort exceeds available RAM — no manual multi-pass
    bucketing required.  A PyArrow rewrite pass is applied afterwards to restore
    dictionary encoding on string columns and the float32/date32 schema optimisations.

    sequential_read: pass parallel='none' to scan_parquet so row groups are read
        in a single sequential pass.  Use this when src is on a rotational or
        external HDD where concurrent random reads across 38k row groups would
        thrash the heads.  The sort and write stages still use all CPU threads.

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
        _parallel = "none" if sequential_read else "auto"
        schema_cols = pl.scan_parquet(src, parallel=_parallel).collect_schema().names()
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
            pl.scan_parquet(src, parallel=_parallel)
            .with_columns(cast_exprs)
            .sort(sort_cols, descending=sort_desc)
            .drop("_northing")
            .sink_parquet(
                sort_dst,
                compression="zstd",
                compression_level=3,
                row_group_size=row_group_size,
                statistics=False,
            )
        )

        if not _skip_dict_rewrite:
            # Rewrite with dictionary encoding on string columns — sink_parquet doesn't
            # expose this, but it meaningfully reduces file size for point_id/item_id/tile_id.
            dict_cols = {"point_id", "item_id", "tile_id"}
            pf = pq.ParquetFile(tmp)
            schema = pf.schema_arrow
            n_rg = pf.metadata.num_row_groups
            stats_cols = [c for c in schema.names if c in {"lon", "lat"}]
            writer = pq.ParquetWriter(
                dst, schema, compression="zstd",
                use_dictionary=[c for c in schema.names if c in dict_cols],
                write_statistics=stats_cols if stats_cols else False,
            )
            from concurrent.futures import ThreadPoolExecutor as _TPE
            with _TPE(max_workers=2) as _pool:
                # Overlap read of row-group i+1 with write of row-group i.
                _pending = None
                for i in range(n_rg):
                    _next = _pool.submit(pf.read_row_group, i)
                    if _pending is not None:
                        writer.write_table(_pending.result())
                    _pending = _next
                if _pending is not None:
                    writer.write_table(_pending.result())
            writer.close()
            tmp.unlink(missing_ok=True)

    except Exception:
        tmp.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        raise
