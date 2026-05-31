"""One-time script to pixel-sort existing Mitchell strip parquets in-place.

The fetch pipeline now writes pixel-sorted output directly (via merge_scenes),
but strips created before this change are northing-heap-sorted and must be
sorted once before scoring.

Uses a chunked external sort to stay within ~50 GB RAM:
  1. Split the strip into N shards, each sorted in its own subprocess (guarantees
     full OS memory reclaim between shards — no reliance on Python GC).
  2. K-way merge the sorted partials with PyArrow (streaming, low RAM).
  3. Dict-rewrite pass to restore dictionary encoding on string columns.
  4. Replace the original file in-place on the HDD.

Any temp files land in /data/tmp/pixel_sort_work (NVMe).

Usage:
    python scripts/backfill_pixel_sort.py [--dir PATH] [--dry-run]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.parquet_utils import is_pixel_sorted

_WORK_DIR = Path("/data/tmp/pixel_sort_work")
_DICT_COLS = {"point_id", "item_id", "tile_id"}
_ROW_GROUP_SIZE = 5_000_000
_SHARD_COMPRESSED_BYTES = 1_000_000_000  # ~1 GB compressed per shard → ~6 GB uncompressed
_N_PHASES = 4


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Shard worker — invoked in a subprocess, sorts one shard and exits
# ---------------------------------------------------------------------------

def _worker(src: str, shard_path: str, rg_start: int, rg_end: int) -> None:
    """Sort row groups [rg_start, rg_end) of src and write to shard_path."""
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    schema_cols = pf.schema_arrow.names

    has_scl = "scl_purity" in schema_cols
    sort_cols = ["_northing", "point_id", "date", "scl_purity"] if has_scl \
        else ["_northing", "point_id", "date"]
    sort_desc = [False] * (len(sort_cols) - 1) + ([True] if has_scl else [False])

    cast_exprs = [
        *(
            [pl.col("lon").cast(pl.Float32), pl.col("lat").cast(pl.Float32)]
            if "lon" in schema_cols else []
        ),
        *(
            [pl.col("date").cast(pl.Date)]
            if "date" in schema_cols else []
        ),
        pl.col("point_id").str.split("_").list.get(-1)
            .cast(pl.Int32, strict=False).fill_null(0).alias("_northing"),
    ]

    arrow_table = pa.Table.from_batches(
        pf.iter_batches(row_groups=list(range(rg_start, rg_end)))
    )
    df = pl.from_arrow(arrow_table)
    del arrow_table

    df = (
        df.lazy()
        .with_columns(cast_exprs)
        .sort(sort_cols, descending=sort_desc)
        .drop("_northing")
        .collect()
    )

    df.write_parquet(
        shard_path,
        compression="zstd",
        compression_level=3,
        row_group_size=_ROW_GROUP_SIZE,
        statistics=True,
    )


# ---------------------------------------------------------------------------
# Phase 1: spawn one subprocess per shard
# ---------------------------------------------------------------------------

def _shard_ranges(pf: pq.ParquetFile) -> list[tuple[int, int]]:
    """Return (rg_start, rg_end) pairs where each shard's compressed size <= budget."""
    meta = pf.metadata
    ranges: list[tuple[int, int]] = []
    shard_start = 0
    shard_bytes = 0
    for rg in range(meta.num_row_groups):
        rg_bytes = meta.row_group(rg).total_byte_size
        if shard_bytes + rg_bytes > _SHARD_COMPRESSED_BYTES and shard_bytes > 0:
            ranges.append((shard_start, rg))
            shard_start = rg
            shard_bytes = 0
        shard_bytes += rg_bytes
    ranges.append((shard_start, meta.num_row_groups))
    return ranges


def _sort_shards(src: Path, work_dir: Path) -> list[Path]:
    pf = pq.ParquetFile(src)
    total_rgs = pf.metadata.num_row_groups
    ranges = _shard_ranges(pf)
    n_shards = len(ranges)
    print(f"    [{_ts()}] phase 1/{_N_PHASES}: sorting {n_shards} shards "
          f"({total_rgs} row groups, ~{_SHARD_COMPRESSED_BYTES//1e9:.0f} GB compressed/shard) ...", flush=True)

    shard_paths: list[Path] = []
    for i, (rg_start, rg_end) in enumerate(ranges):
        shard_path = work_dir / f"shard_{i:03d}.parquet"

        t0 = time.monotonic()
        result = subprocess.run(
            [
                sys.executable, __file__,
                "--_worker", str(src), str(shard_path),
                str(rg_start), str(rg_end),
            ],
            check=True,
        )
        elapsed = time.monotonic() - t0
        shard_gb = shard_path.stat().st_size / 1e9
        print(f"      shard {i+1}/{n_shards}: RGs {rg_start}–{rg_end-1}, "
              f"{shard_gb:.1f} GB, {elapsed:.0f}s", flush=True)
        shard_paths.append(shard_path)

    return shard_paths


# ---------------------------------------------------------------------------
# Phase 2: k-way merge via PyArrow dataset (streaming)
# ---------------------------------------------------------------------------

def _merge_shards(shard_paths: list[Path], dst: Path, schema: pa.Schema) -> None:
    """Merge sorted shards via DuckDB out-of-core sort.

    DuckDB has a genuine spill-to-disk sort engine — it won't OOM regardless
    of data size.  Spill goes to _WORK_DIR (NVMe).  Output is written as
    Parquet via COPY ... TO with row_group_size and zstd compression.
    """
    import duckdb

    print(f"    [{_ts()}] phase 2/{_N_PHASES}: merge {len(shard_paths)} shards via DuckDB out-of-core sort ...", flush=True)

    spill_dir = _WORK_DIR / "duckdb_tmp"
    spill_dir.mkdir(parents=True, exist_ok=True)

    has_scl = "scl_purity" in schema.names
    sort_expr = "TRY_CAST(split_part(point_id, '_', -1) AS INTEGER) NULLIF 0, point_id, date" \
        + (", scl_purity DESC" if has_scl else "")

    shard_list = ", ".join(f"'{p}'" for p in shard_paths)

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{spill_dir}'")
    con.execute("SET memory_limit = '40GB'")
    con.execute("SET threads = 28")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet([{shard_list}])
            ORDER BY {sort_expr}
        ) TO '{dst}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {_ROW_GROUP_SIZE})
    """)
    con.close()


# ---------------------------------------------------------------------------
# Main sort routine
# ---------------------------------------------------------------------------

def sort_strip(src: Path, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    merged = work_dir / "merged.parquet"
    merged.unlink(missing_ok=True)

    # Phase 1: sort shards (each in its own subprocess)
    t1 = time.monotonic()
    shard_paths = _sort_shards(src, work_dir)
    print(f"    [{_ts()}] phase 1/{_N_PHASES} done in {(time.monotonic()-t1)/60:.1f} min", flush=True)

    # Phase 2: k-way merge (includes dict-rewrite via ParquetWriter)
    t2 = time.monotonic()
    schema = pq.ParquetFile(shard_paths[0]).schema_arrow
    _merge_shards(shard_paths, merged, schema)
    print(f"    [{_ts()}] phase 2/{_N_PHASES} done in {(time.monotonic()-t2)/60:.1f} min", flush=True)

    for p in shard_paths:
        p.unlink(missing_ok=True)

    # Phase 3: (dict encoding already applied in merge writer — skip separate pass)
    print(f"    [{_ts()}] phase 3/{_N_PHASES}: skipped (dict encoding applied during merge)", flush=True)

    # Phase 4: move to HDD
    t4 = time.monotonic()
    print(f"    [{_ts()}] phase 4/{_N_PHASES}: replace original on HDD ...", flush=True)
    import shutil
    shutil.move(str(merged), str(src))
    print(f"    [{_ts()}] phase 4/{_N_PHASES} done in {(time.monotonic()-t4)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def backfill(strips_dir: Path, *, dry_run: bool = False) -> None:
    strips = sorted(strips_dir.glob("*_strip_??.parquet"))
    if not strips:
        print(f"No strip parquets found in {strips_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(strips)} strips in {strips_dir}")

    for p in strips:
        size_gb = p.stat().st_size / 1e9
        print(f"\n[{_ts()}] checking {p.name} ({size_gb:.1f} GB compressed) ...", flush=True)

        if is_pixel_sorted(p):
            print(f"  already pixel-sorted, skipping")
            continue

        if dry_run:
            print(f"  [dry-run] would sort")
            continue

        work_dir = _WORK_DIR / p.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        try:
            sort_strip(p, work_dir)
        except Exception:
            print(f"  ERROR — leaving work dir {work_dir} for inspection", file=sys.stderr)
            raise
        elapsed = time.monotonic() - t0
        print(f"  [{_ts()}] strip done: {elapsed/60:.1f} min total  "
              f"({size_gb / (elapsed/60):.0f} MB/min throughput)", flush=True)

        try:
            work_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="/mnt/external/mitchell/mitchell/2025/54LWH",
                        help="Directory containing strip parquets (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true")
    # Internal: called by _sort_shards to sort a single shard in isolation
    parser.add_argument("--_worker", nargs=4, metavar=("SRC", "DST", "RG_START", "RG_END"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        src, dst, rg_start, rg_end = args._worker
        _worker(src, dst, int(rg_start), int(rg_end))
        return

    strips_dir = Path(args.dir)
    if not strips_dir.is_dir():
        print(f"Directory not found: {strips_dir}", file=sys.stderr)
        sys.exit(1)

    backfill(strips_dir, dry_run=args.dry_run)
    print("\nAll done.")


if __name__ == "__main__":
    main()
