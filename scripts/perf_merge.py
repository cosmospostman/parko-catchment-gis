"""Merge performance harness for _sort_s1_shards and merge_strips.

Synthesises sorted parquet shards that match the real S1/S2 schema and
measures k-way merge throughput under two access patterns:

  sequential  — shards cover non-overlapping northing bands (best case:
                whole-block fast path dominates)
  interleaved — shards share the same northing range (worst case: every
                block boundary straddles, exercising binary-search path)

Run:
    python scripts/perf_merge.py
    python scripts/perf_merge.py --n-pixels 2_000_000 --n-shards 8
    python scripts/perf_merge.py --mode interleaved --n-pixels 500_000
    python scripts/perf_merge.py --scale 1.0   # approximate real 2M-pixel job
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class Probe(NamedTuple):
    tag: str
    elapsed_s: float
    rows: int | None


_probes: list[Probe] = []
_t0 = time.perf_counter()


def probe(tag: str, rows: int | None = None) -> None:
    _probes.append(Probe(tag=tag, elapsed_s=time.perf_counter() - _t0, rows=rows))
    rows_s = f"  ({rows:>12,} rows)" if rows is not None else ""
    print(f"  [{_probes[-1].elapsed_s:7.2f}s]  {tag}{rows_s}")


def elapsed_between(tag_a: str, tag_b: str) -> float:
    ta = next(p.elapsed_s for p in _probes if p.tag == tag_a)
    tb = next(p.elapsed_s for p in _probes if p.tag == tag_b)
    return tb - ta


def print_report(total_rows: int) -> None:
    print("\n" + "=" * 72)
    print(f"{'Stage':<38}  {'Elapsed s':>10}  {'Rows/s':>12}")
    print("-" * 72)
    prev = _probes[0].elapsed_s
    for p in _probes:
        dt = p.elapsed_s - prev
        rows_s = f"{p.rows / dt:12,.0f}" if (p.rows and dt > 0.001) else f"{'':>12}"
        print(f"  {p.tag:<36}  {p.elapsed_s:10.2f}  {rows_s}")
        prev = p.elapsed_s
    print("=" * 72)

    merge_stages = [p for p in _probes if "merge" in p.tag.lower()]
    if len(merge_stages) >= 2:
        t_start = next(p.elapsed_s for p in _probes if "merge start" in p.tag)
        t_end   = next(p.elapsed_s for p in _probes if "merge done" in p.tag)
        dt = t_end - t_start
        rate = total_rows / dt if dt > 0 else 0
        print(f"\nMerge wall time: {dt:.2f}s   throughput: {rate:,.0f} rows/s")

        # Project to real workload sizes
        for label, n in [("80M rows (typical tile)", 80_000_000),
                          ("200M rows (large location)", 200_000_000)]:
            proj = n / rate if rate > 0 else float("inf")
            print(f"  Projected for {label}: {proj:.1f}s")


# ---------------------------------------------------------------------------
# Synthetic shard builder
# ---------------------------------------------------------------------------

# Matches the real combined S1+S2 schema from 54LWH.s2.parquet + source/vh/vv
_FLOAT_COLS = [
    "lon", "lat", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
    "B11", "B12", "scl_purity", "aot", "view_zenith", "sun_zenith",
    "NDVI", "NDWI", "EVI", "MAVI", "NDRE", "CI_RE", "vh", "vv",
]


def _make_combined_schema() -> "pa.Schema":
    import pyarrow as pa
    return pa.schema([
        pa.field("point_id",   pa.large_string()),
        pa.field("lon",        pa.float32()),
        pa.field("lat",        pa.float32()),
        pa.field("date",       pa.date32()),
        pa.field("item_id",    pa.large_string()),
        pa.field("tile_id",    pa.large_string()),
        pa.field("source",     pa.string()),
        *[pa.field(c, pa.float32()) for c in _FLOAT_COLS if c not in ("lon", "lat")],
        pa.field("scl",        pa.int8()),
    ])


def _make_shard(
    out_path: Path,
    point_ids: list[str],        # already-assigned pids for this shard
    n_dates: int,
    schema: "pa.Schema",
    seed: int,
    row_group_size: int,
) -> int:
    """Write one sorted shard parquet; returns row count."""
    import datetime
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    rng = np.random.default_rng(seed)
    n_px = len(point_ids)
    total = n_px * n_dates

    # Build dates: each pixel gets n_dates random dates, then sort within pixel
    base = datetime.date(2021, 1, 1).toordinal()
    dates_per_px = np.sort(rng.integers(0, 365, size=(n_px, n_dates)), axis=1)
    dates_flat = (base + dates_per_px).flatten().astype(np.int32)

    pid_col   = np.repeat(point_ids, n_dates)
    item_col  = np.array([f"S1GRDH_{i % 200:04d}" for i in range(total)], dtype=object)
    tile_col  = np.full(total, "54LWH", dtype=object)
    src_col   = np.full(total, "S1", dtype=object)
    scl_col   = rng.integers(4, 7, size=total, dtype=np.int8)

    # Extract northings from point_ids for sort key
    northings = np.array([int(pid.rsplit("_", 1)[-1]) for pid in point_ids], dtype=np.int32)
    northings_rep = np.repeat(northings, n_dates)

    # Sort by (northing, point_id, date) — the merge order
    order = np.lexsort((dates_flat, pid_col, northings_rep))
    pid_col      = pid_col[order]
    dates_flat   = dates_flat[order]
    item_col     = item_col[order]
    scl_col      = scl_col[order]
    northings_rep = northings_rep[order]

    floats = {c: rng.standard_normal(total).astype(np.float32) for c in _FLOAT_COLS}
    for c in _FLOAT_COLS:
        floats[c] = floats[c][order]

    arrays = {
        "point_id": pa.array(pid_col, type=pa.large_string()),
        "lon":      pa.array(floats["lon"], type=pa.float32()),
        "lat":      pa.array(floats["lat"], type=pa.float32()),
        "date":     pa.array(dates_flat, type=pa.date32()),
        "item_id":  pa.array(item_col, type=pa.large_string()),
        "tile_id":  pa.array(tile_col, type=pa.large_string()),
        "source":   pa.array(src_col, type=pa.string()),
        **{c: pa.array(floats[c], type=pa.float32()) for c in _FLOAT_COLS if c not in ("lon", "lat")},
        "scl":      pa.array(scl_col, type=pa.int8()),
    }
    tbl = pa.table({f.name: arrays[f.name] for f in schema}, schema=schema)

    writer = pq.ParquetWriter(
        out_path, schema, compression="zstd",
        use_dictionary=["point_id", "item_id", "tile_id"],
        write_statistics=True,
    )
    for start in range(0, total, row_group_size):
        writer.write_table(tbl.slice(start, min(row_group_size, total - start)))
    writer.close()
    return total


def build_shards(
    out_dir: Path,
    n_pixels: int,
    n_shards: int,
    n_dates: int,
    mode: str,          # "sequential" | "interleaved"
    schema: "pa.Schema",
    row_group_size: int,
    seed: int = 0,
) -> tuple[list[Path], int]:
    """Build n_shards sorted parquet files; return (paths, total_rows)."""
    rng = np.random.default_rng(seed)

    # Generate point_ids as "px_{easting}_{northing}" matching real format
    grid = int(np.ceil(np.sqrt(n_pixels)))
    xs = rng.integers(0, grid, size=n_pixels)
    ys = rng.integers(0, grid, size=n_pixels)
    all_pids = [f"px_{x:04d}_{y:04d}" for x, y in zip(xs, ys)]

    shard_paths: list[Path] = []
    total_rows = 0

    if mode == "sequential":
        # Each shard gets a non-overlapping northing band — sort pixels by northing first
        # so the split produces truly distinct northing ranges (best-case for merge).
        northing_order = np.argsort([int(pid.rsplit("_", 1)[-1]) for pid in all_pids])
        slices = np.array_split(northing_order, n_shards)
        for idx, sl in enumerate(slices):
            pids = [all_pids[i] for i in sl]
            p = out_dir / f"shard_{idx:02d}.parquet"
            rows = _make_shard(p, pids, n_dates, schema, seed=seed + idx, row_group_size=row_group_size)
            shard_paths.append(p)
            total_rows += rows
            print(f"    shard {idx+1}/{n_shards}: {rows:,} rows → {p.name}")
    else:
        # interleaved: all shards share the same pixel set — maximum block straddling
        for idx in range(n_shards):
            p = out_dir / f"shard_{idx:02d}.parquet"
            rows = _make_shard(p, all_pids, n_dates // n_shards or 1, schema,
                               seed=seed + idx, row_group_size=row_group_size)
            shard_paths.append(p)
            total_rows += rows
            print(f"    shard {idx+1}/{n_shards}: {rows:,} rows → {p.name}")

    return shard_paths, total_rows


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------

def run_harness(
    n_pixels: int,
    n_shards: int,
    n_dates: int,
    mode: str,
    row_group_size: int,
) -> None:
    import pyarrow as pa
    from utils.parquet_utils import _sort_s1_shards

    schema = _make_combined_schema()
    total_rows_est = n_pixels * n_dates

    print(f"\nMerge performance harness")
    print(f"  mode={mode}  n_pixels={n_pixels:,}  n_shards={n_shards}  "
          f"n_dates={n_dates}  row_group_size={row_group_size:,}")
    print(f"  estimated total rows: {total_rows_est:,}")
    print()

    with tempfile.TemporaryDirectory(prefix="perf_merge_") as tmp:
        tmp_dir = Path(tmp)
        out_path = tmp_dir / "merged.parquet"

        # --- Build shards ---------------------------------------------------
        probe("harness start")
        print(f"Building {n_shards} synthetic shards ...")
        shard_paths, total_rows = build_shards(
            tmp_dir, n_pixels, n_shards, n_dates, mode, schema, row_group_size,
        )
        probe("shards built", total_rows)
        print(f"  actual total rows: {total_rows:,}")
        gc.collect()

        # --- Run merge (monkey-patch _emit to count calls) ------------------
        print(f"\nRunning _sort_s1_shards ({mode}) ...")
        import utils.parquet_utils as pu
        _orig_conform = pu._conform_table
        emit_calls = [0]
        emit_rows  = [0]
        t_conform  = [0.0]
        def _counted_conform(tbl, schema):
            emit_calls[0] += 1
            emit_rows[0]  += len(tbl)
            t0 = time.perf_counter()
            result = _orig_conform(tbl, schema)
            t_conform[0] += time.perf_counter() - t0
            return result
        pu._conform_table = _counted_conform
        try:
            probe("merge start")
            _sort_s1_shards(shard_paths, out_path, schema)
            probe("merge done", total_rows)
        finally:
            pu._conform_table = _orig_conform
        print(f"  _conform_table calls: {emit_calls[0]:,}  "
              f"avg rows/call: {emit_rows[0]//max(emit_calls[0],1):,}  "
              f"total conform time: {t_conform[0]:.2f}s")
        # Also count heap loop iterations by patching the loop counter via _sort_s1_shards source
        # (done via emit count — each emit = one loop iteration in straddle path or whole-block path)

        # --- Verify output --------------------------------------------------
        import pyarrow.parquet as pq
        out_pf = pq.ParquetFile(out_path)
        out_rows = out_pf.metadata.num_rows
        out_rgs  = out_pf.metadata.num_row_groups
        print(f"\nOutput: {out_rows:,} rows  {out_rgs} row-groups  "
              f"(expected {total_rows:,}{'  OK' if out_rows == total_rows else '  MISMATCH'})")

    print_report(total_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-way merge performance harness")
    parser.add_argument("--n-pixels",        type=int,   default=200_000,
                        help="Unique pixels across all shards (default: 200000)")
    parser.add_argument("--n-shards",        type=int,   default=4,
                        help="Number of sorted shards to merge (default: 4)")
    parser.add_argument("--n-dates",         type=int,   default=40,
                        help="Observations per pixel (default: 40)")
    parser.add_argument("--mode",            type=str,   default="both",
                        choices=["sequential", "interleaved", "both"],
                        help="Pixel layout: sequential=non-overlapping northings "
                             "(fast path), interleaved=shared northings (straddle "
                             "path), both=run both (default: both)")
    parser.add_argument("--row-group-size",  type=int,   default=5_000_000,
                        help="Parquet row group size (default: 5000000)")
    parser.add_argument("--scale",           type=float, default=None,
                        help="Scale relative to real job: 1.0 ≈ 2M pixels × 40 dates "
                             "× 4 shards = 80M rows. Overrides --n-pixels.")
    args = parser.parse_args()

    if args.scale is not None:
        args.n_pixels = max(1_000, int(2_000_000 * args.scale))

    modes = ["sequential", "interleaved"] if args.mode == "both" else [args.mode]
    for mode in modes:
        _probes.clear()
        _t0 = time.perf_counter()
        run_harness(
            n_pixels=args.n_pixels,
            n_shards=args.n_shards,
            n_dates=args.n_dates,
            mode=mode,
            row_group_size=args.row_group_size,
        )
        print()
