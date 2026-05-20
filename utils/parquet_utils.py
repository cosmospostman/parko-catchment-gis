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
                arrays.append(pa.array(col.to_list(), type=field.type))
            except Exception:
                arrays.append(pa.array([None] * rows, type=field.type))
        else:
            arrays.append(pa.array([None] * rows, type=field.type))
    return pa.table(
        {field.name: arrays[i] for i, field in enumerate(schema)},
        schema=schema,
    )


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

    Streams S1 shard parquets row-group by row-group to avoid materialising
    the full S1 dataset in memory (which OOMs on large tiles like legune).
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
        has_s1 = any(
            "S1" in pf.read_row_group(rg, columns=["source"]).column("source").to_pylist()
            for rg in range(pf.metadata.num_row_groups)
        )
        if has_s1:
            return

    combined_schema = _extend_schema(pf.read_row_group(0).schema)
    seen: dict[str, tuple[float, float]] = {}
    n_rg = pf.metadata.num_row_groups
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

    tmp_path = tile_path.with_suffix(".tmp.parquet")
    with tempfile.TemporaryDirectory(prefix="s1_merge_") as _tmp:
        if items:
            shard_paths = _collect_s1_shards(
                out_dir=Path(_tmp),
                items=items,
                bbox_wgs84=bbox_wgs84,
                points=points_for_s1,
                resolved_cache=resolved_cache,
            )
        else:
            shard_paths = []

        writer = pq.ParquetWriter(tmp_path, combined_schema)
        for rg_idx in range(n_rg):
            tbl = pf.read_row_group(rg_idx)
            tbl = _conform_table(tbl, combined_schema)
            source_col = pa.array(["S2"] * len(tbl), type=pa.string())
            tbl = tbl.set_column(tbl.schema.get_field_index("source"), "source", source_col)
            writer.write_table(tbl)

        for shard_path in shard_paths:
            s1_pf = pq.ParquetFile(shard_path)
            for rg_idx in range(s1_pf.metadata.num_row_groups):
                tbl = s1_pf.read_row_group(rg_idx)
                tbl = _conform_table(tbl, combined_schema)
                writer.write_table(tbl)

        writer.close()

    # S2 rows followed by S1 rows are source-separated, not pixel-sorted.
    # Sort by pixel so all observations for each pixel are contiguous.
    sorted_tmp = tile_path.with_suffix(".sorted_tmp.parquet")
    sorted_tmp.unlink(missing_ok=True)
    sort_parquet_by_pixel(tmp_path, sorted_tmp, row_group_size=5_000_000)
    tmp_path.unlink(missing_ok=True)
    sorted_tmp.replace(tile_path)


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
        # Derive a northing-row sort key from point_id ("px_<easting>_<northing>")
        # so pixels in the same geographic row are co-located in output row groups.
        # Lexicographic sort on the full point_id would group by easting instead.
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
            pl.col("point_id").str.splitn("_", 4).struct.field("field_2").alias("_northing"),
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
