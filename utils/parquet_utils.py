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


def _sort_s1_shards(
    shard_paths: list[Path],
    out_path: Path,
    combined_schema: "pa.Schema",
    n_workers: int | None = None,
) -> None:
    """Merge S1 shards into one pixel-sorted parquet conforming to combined_schema.

    Strategy: sort each shard independently (each covers ≤50 k points, so small),
    then merge-sort via DuckDB with spill-to-disk so peak RAM stays bounded
    regardless of total shard count.
    """
    import logging
    import multiprocessing
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    from concurrent.futures import ProcessPoolExecutor, as_completed

    _log = logging.getLogger(__name__)

    if len(shard_paths) == 1:
        _log.info("sorting 1 shard ...")
        sort_parquet_by_pixel(shard_paths[0], out_path, row_group_size=5_000_000, _skip_dict_rewrite=True)
        _log.info("sort done → %s", out_path.name)
        return

    # Step 1: sort each shard independently into a sibling .sorted.parquet.
    # Sorts are CPU-bound and independent — parallelise with processes.
    # Must use "spawn" not "fork": Polars holds Rayon thread-pool locks that
    # cause forked children to deadlock immediately on futex_wait.
    sorted_paths: list[Path] = [sp.with_suffix(".sorted.parquet") for sp in shard_paths]
    n_workers = min(len(shard_paths), n_workers or os.cpu_count() or 4)
    _log.info("sorting %d shards (%d workers) ...", len(shard_paths), n_workers)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futs = {
            pool.submit(sort_parquet_by_pixel, sp, sp_sorted, 5_000_000, _skip_dict_rewrite=True): idx
            for idx, (sp, sp_sorted) in enumerate(zip(shard_paths, sorted_paths))
        }
        done = 0
        for fut in as_completed(futs):
            fut.result()  # re-raise any worker exception
            done += 1
            _log.info("shard sort %d/%d done", done, len(shard_paths))

    _log.info("merging %d sorted shards → %s ...", len(shard_paths), out_path.name)

    # Step 2: sort-merge via DuckDB with an explicit memory cap and spill-to-disk.
    # The shards are partitioned by point batch, not northing band, so their
    # northing ranges overlap fully — a k-way merge gains nothing over a sort.
    # DuckDB spills to temp_directory when the memory limit is hit, so peak RSS
    # stays bounded on the 8 GB fetcher machine.
    import duckdb

    tmp_path = out_path.with_suffix(".s1_merge_tmp.parquet")
    tmp_path.unlink(missing_ok=True)

    shard_glob = str(sorted_paths[0].parent / "*.sorted.parquet")
    tmp_dir    = str(out_path.parent)

    sql = f"""
        COPY (
            SELECT * EXCLUDE (_n)
            FROM (
                SELECT *,
                    regexp_extract(point_id, '_([0-9]+)$', 1)::INTEGER AS _n
                FROM read_parquet('{shard_glob}', union_by_name=true)
            ) t
            ORDER BY _n, date
        ) TO '{tmp_path!s}' (
            FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 5000000
        )
    """

    try:
        con = duckdb.connect(config={"temp_directory": tmp_dir, "memory_limit": "2GB"})
        try:
            con.execute("SET threads = 4")
            con.execute("SET preserve_insertion_order = false")
            con.execute(sql)
        finally:
            con.close()
        tmp_path.replace(out_path)
        _log.info("merge done → %s", out_path.name)
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
    memory_limit_gb: int = 16,
) -> None:
    """2-way sort-merge of S2 and S1 parquets into out_path via DuckDB.

    DuckDB reads both files, tags S2 rows with source='S2', unions them, sorts
    by (northing, date), and writes to parquet — spilling to disk if needed so
    peak RAM stays bounded regardless of file size.
    """
    import duckdb
    import os
    import pyarrow.parquet as pq

    s2_p = str(s2_path)
    s1_p = str(s1_path)
    dst  = str(out_path)
    tmp_dir = str(out_path.parent)

    # Determine S2 and S1 column sets to build safe SELECT clauses.
    # S2 may not have 'source'; S1 always has it (set to "S1").
    s2_cols = set(pq.ParquetFile(s2_path).schema_arrow.names)
    s1_cols = set(pq.ParquetFile(s1_path).schema_arrow.names)
    out_names = combined_schema.names  # ordered output columns

    def _select_clause(file_cols: set[str], override_source: str | None) -> str:
        parts = []
        for name in out_names:
            if name == "source" and override_source is not None:
                parts.append(f"'{override_source}' AS source")
            elif name in file_cols:
                parts.append(f'"{name}"')
            else:
                parts.append(f"NULL AS \"{name}\"")
        return ", ".join(parts)

    s2_src_override = "S2" if tag_s2_source else None
    s2_select = _select_clause(s2_cols, s2_src_override)
    s1_select = _select_clause(s1_cols, None)

    col_list = ", ".join(f'"{c}"' for c in out_names)

    sql = f"""
        COPY (
            SELECT {col_list}
            FROM (
                SELECT {s2_select} FROM read_parquet('{s2_p}')
                UNION ALL
                SELECT {s1_select} FROM read_parquet('{s1_p}')
            ) t
            ORDER BY regexp_extract(point_id, '_([0-9]+)$', 1)::INTEGER, date
        ) TO '{dst}' (
            FORMAT PARQUET, COMPRESSION ZSTD,
            ROW_GROUP_SIZE 5000000
        )
    """

    s2_rows = pq.ParquetFile(s2_path).metadata.num_rows
    s1_rows = pq.ParquetFile(s1_path).metadata.num_rows
    n_threads = max(1, (os.cpu_count() or 4) // 2)
    logger.info(
        "merge_tile: sort-merging %s (%s S2 rows + %s S1 rows) via DuckDB (%d threads, %d GB limit) ...",
        out_path.name, f"{s2_rows:,}", f"{s1_rows:,}", n_threads, memory_limit_gb,
    )

    con = duckdb.connect(config={"temp_directory": tmp_dir, "memory_limit": f"{memory_limit_gb}GB"})
    try:
        con.execute(f"SET threads = {n_threads}")
        con.execute("SET preserve_insertion_order = false")
        con.execute(sql)
    finally:
        con.close()


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
