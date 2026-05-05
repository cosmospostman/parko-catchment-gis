"""utils/parquet_utils.py — Parquet write options, schema helpers, and pixel-sort utilities.

Functions previously in signals/_shared.py that are needed by multiple modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pyarrow as pa


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


def _s1_df_to_arrow(df_s1: "pd.DataFrame", schema: "pa.Schema") -> "pa.Table":
    """Convert an S1 DataFrame to a PyArrow table conforming to the combined schema.

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
                arrays.append(pa.array(col.tolist(), type=field.type))
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
    ram_budget_gb: float = 8.0,
    read_workers: int = 6,
) -> None:
    """Write a copy of ``src`` sorted by ``point_id`` using an in-memory multi-pass sort.

    Algorithm:
    1. Discover all unique row-coords by scanning ``point_id`` (column-only read).
    2. Divide coords into passes that fit within ``ram_budget_gb``.
    3. For each pass: scan all source row groups once, accumulate rows for the
       target coords in RAM, sort by ``point_id``, append to output.

    No temp files are written — all sorting is in RAM. The source file is read
    sequentially (compressed, ~50 MB/rg) which is fast and cache-friendly.

    Peak RAM ≈ one pass worth of buckets (≤ ``ram_budget_gb``).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    pf = pq.ParquetFile(src)
    n_rg = pf.metadata.num_row_groups

    # Step 1: discover all unique row-coords (point_id column only)
    print(f"  [sort] scanning {n_rg} row groups for unique row-coords ...", flush=True)
    all_coords: set[str] = set()
    for i in range(n_rg):
        tbl = pf.read_row_group(i, columns=["point_id"])
        ids = tbl.column("point_id")
        row_coords = pc.list_flatten(pc.list_slice(pc.split_pattern(ids, "_"), 2, 3))
        all_coords.update(pc.unique(row_coords).to_pylist())
        if i % 400 == 0:
            print(f"  [sort] coord scan: {i}/{n_rg}, {len(all_coords)} coords so far", flush=True)

    sorted_coords = sorted(all_coords)
    n_coords = len(sorted_coords)

    # Step 2: divide coords into passes
    total_rows = pf.metadata.num_rows
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

    # Step 3: multi-pass scan → sort → write
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    out_writer: "pq.ParquetWriter | None" = None
    try:
        for pass_idx, pass_coords in enumerate(passes):
            coord_set = set(pass_coords)
            coord_set_arr = pa.array(list(coord_set))
            buckets: dict[str, list[pa.Table]] = {c: [] for c in pass_coords}
            buckets_lock = threading.Lock()
            rg_counter = [0]

            print(
                f"  [sort] pass {pass_idx+1}/{len(passes)}: "
                f"scanning {n_rg} row groups for {len(pass_coords)} coords "
                f"({read_workers} workers) ...",
                flush=True,
            )

            def _read_and_bucket(rg_idx: int) -> None:
                batch = pf.read_row_group(rg_idx)
                ids = batch.column("point_id")
                row_coords = pc.list_flatten(pc.list_slice(pc.split_pattern(ids, "_"), 2, 3))
                in_pass = pc.is_in(row_coords, value_set=coord_set_arr)
                if not pc.any(in_pass).as_py():
                    with buckets_lock:
                        rg_counter[0] += 1
                        if rg_counter[0] % 400 == 0:
                            print(f"  [sort] pass {pass_idx+1}: {rg_counter[0]}/{n_rg} row groups", flush=True)
                    return
                batch = batch.filter(in_pass)
                row_coords = row_coords.filter(in_pass)
                unique_in_batch = pc.unique(row_coords).to_pylist()
                if len(unique_in_batch) == 1:
                    slices = [(unique_in_batch[0], batch)]
                else:
                    slices = [
                        (coord, batch.filter(pc.equal(row_coords, coord)))
                        for coord in unique_in_batch
                    ]
                with buckets_lock:
                    for coord, tbl in slices:
                        buckets[coord].append(tbl)
                    rg_counter[0] += 1
                    if rg_counter[0] % 400 == 0:
                        print(f"  [sort] pass {pass_idx+1}: {rg_counter[0]}/{n_rg} row groups", flush=True)

            with ThreadPoolExecutor(max_workers=read_workers) as pool:
                futures = [pool.submit(_read_and_bucket, i) for i in range(n_rg)]
                for fut in as_completed(futures):
                    fut.result()

            pending: list[pa.Table] = []
            pending_rows = 0

            def _flush() -> None:
                nonlocal out_writer, pending_rows
                if not pending:
                    return
                for tbl in pending:
                    if out_writer is None:
                        out_writer = pq.ParquetWriter(str(dst), tbl.schema, **_WRITE_OPTS)
                    out_writer.write_table(tbl)
                pending.clear()
                pending_rows = 0

            for coord in pass_coords:
                chunks = buckets.pop(coord)
                if not chunks:
                    continue
                bucket_tbl = pa.concat_tables(chunks)
                order = pc.sort_indices(bucket_tbl, sort_keys=[("point_id", "ascending"), ("date", "ascending"), ("scl_purity", "descending")])
                bucket_tbl = _optimise_schema(bucket_tbl.take(order))
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
