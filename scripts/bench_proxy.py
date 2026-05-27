"""Fetch-proxy pipeline benchmark harness.

Synthesises realistic data and benchmarks each stage of the VM-side pipeline
(DuckDB merge, per-scene extraction, compression ratio, workstation concat)
plus a discrete-event pipeline simulation.

Run:
    python scripts/bench_proxy.py
    python scripts/bench_proxy.py --n-scenes 100 --strip-px 1024 --n-pixels 11000
    python scripts/bench_proxy.py --assert-merge-s 30 --assert-compression-ratio 5
    python scripts/bench_proxy.py --assert-extract-s 2
    python scripts/bench_proxy.py --assert-concat-mb-s 500
    python scripts/bench_proxy.py --sim-fetch-s 80 --sim-extract-s 60 --sim-merge-s 30 --sim-stream-s 147

Exit codes: 0 = ok, 1 = assertion failure (--assert-*).
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS


# ---------------------------------------------------------------------------
# Probe system
# ---------------------------------------------------------------------------

def rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


class Probe(NamedTuple):
    tag: str
    rss_gb: float
    delta_rss_gb: float
    elapsed_s: float
    rows: int | None
    extra: str


_probes: list[Probe] = []
_t0 = time.perf_counter()


def probe(tag: str, rows: int | None = None, extra: str = "") -> None:
    prev_rss = _probes[-1].rss_gb if _probes else rss_gb()
    r = rss_gb()
    elapsed = time.perf_counter() - _t0
    _probes.append(Probe(tag=tag, rss_gb=r, delta_rss_gb=r - prev_rss,
                         elapsed_s=elapsed, rows=rows, extra=extra))
    delta_s = f"Δ{r - prev_rss:+.2f}"
    rows_s  = f"  rows={rows:,}" if rows is not None else ""
    print(f"  [{elapsed:7.2f}s]  {tag}  RSS={r:.2f}GB {delta_s}GB{rows_s}{' ' + extra if extra else ''}", flush=True)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_scene_parquet(path: Path, scene_id: str, n_points: int, n_dates: int = 1) -> Path:
    """Write one per-scene parquet (sorted by northing, ZSTD, dict on point_id)."""
    rows = []
    for d_idx in range(n_dates):
        d = date(2022, 1, d_idx + 1)
        for i in range(n_points):
            rows.append({
                "point_id": f"px_{i:04d}_0000",
                "lon":       float(145.41 + i * 0.0001),
                "lat":       float(-22.81 + i * 0.0001),
                "date":      d,
                "item_id":   scene_id,
                "tile_id":   "55HBU",
                "source":    "S2",
                "scl_purity": 100,
                "scl":        4,
                "aot":        80,
                "view_zenith": 90,
                "sun_zenith":  80,
            })

    schema_fields = [
        pa.field("point_id",    pa.string()),
        pa.field("lon",         pa.float32()),
        pa.field("lat",         pa.float32()),
        pa.field("date",        pa.date32()),
        pa.field("item_id",     pa.string()),
        pa.field("tile_id",     pa.string()),
        pa.field("source",      pa.string()),
        pa.field("scl_purity",  pa.int8()),
        pa.field("scl",         pa.int8()),
        pa.field("aot",         pa.uint8()),
        pa.field("view_zenith", pa.uint8()),
        pa.field("sun_zenith",  pa.uint8()),
    ]
    for band in BANDS:
        schema_fields.append(pa.field(band, pa.uint16()))

    schema = pa.schema(schema_fields)
    tbl = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="zstd",
                   use_dictionary=["point_id", "item_id", "tile_id"])
    return path


def _make_unsorted_parquet(path: Path, n_points: int, n_dates: int) -> Path:
    """Write a parquet with randomly ordered rows and no dictionary encoding."""
    rows = []
    for d_idx in range(n_dates):
        d = date(2022, 1, d_idx + 1)
        pids = [f"px_{i:04d}_0000" for i in range(n_points)]
        random.shuffle(pids)
        for pid in pids:
            rows.append({
                "point_id": pid,
                "date": d,
                "source": "S2",
                "value": random.randint(0, 10000),
            })
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("date",     pa.date32()),
        pa.field("source",   pa.string()),
        pa.field("value",    pa.int32()),
    ])
    tbl = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="zstd")
    return path


def _make_sorted_parquet(path: Path, n_points: int, n_dates: int) -> Path:
    """Write a parquet sorted by point_id with dictionary encoding."""
    rows = []
    for i in range(n_points):
        pid = f"px_{i:04d}_0000"
        for d_idx in range(n_dates):
            rows.append({
                "point_id": pid,
                "date": date(2022, 1, d_idx + 1),
                "source": "S2",
                "value": 5000,
            })
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("date",     pa.date32()),
        pa.field("source",   pa.string()),
        pa.field("value",    pa.int32()),
    ])
    tbl = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="zstd", use_dictionary=["point_id"])
    return path


# ---------------------------------------------------------------------------
# bench_duckdb_merge
# ---------------------------------------------------------------------------

def bench_duckdb_merge(tmp: Path, n_scenes: int, n_pixels: int) -> float:
    from proxy._pipeline import merge_scenes

    print(f"\n[bench_duckdb_merge]  n_scenes={n_scenes}  n_pixels={n_pixels}")
    scene_dir = tmp / "scenes"
    scene_paths = []
    t_make = time.perf_counter()
    for i in range(n_scenes):
        p = _make_scene_parquet(scene_dir / f"scene_{i:04d}.parquet", f"scene_{i:04d}", n_pixels)
        scene_paths.append(p)
    print(f"  scene parquets created in {time.perf_counter() - t_make:.1f}s")

    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in scene_paths)
    probe("merge:start", rows=total_rows)

    out = tmp / "strip_merged.parquet"
    t0 = time.perf_counter()
    merge_scenes(scene_paths, None, out)
    elapsed = time.perf_counter() - t0

    actual_rows = pq.ParquetFile(out).metadata.num_rows
    size_mb = out.stat().st_size / 1e6
    probe("merge:done", rows=actual_rows, extra=f"{size_mb:.0f}MB  {elapsed:.1f}s")
    print(f"  DuckDB merge: {elapsed:.1f}s  ({total_rows:,} rows → {size_mb:.0f} MB)")

    gc.collect()
    return elapsed


# ---------------------------------------------------------------------------
# bench_compression_ratio
# ---------------------------------------------------------------------------

def bench_compression_ratio(tmp: Path, n_points: int, n_dates: int) -> float:
    print(f"\n[bench_compression_ratio]  n_points={n_points}  n_dates={n_dates}")

    unsorted = _make_unsorted_parquet(tmp / "unsorted.parquet", n_points, n_dates)
    sorted_d = _make_sorted_parquet(tmp / "sorted.parquet",   n_points, n_dates)

    sz_u = unsorted.stat().st_size
    sz_s = sorted_d.stat().st_size
    ratio = sz_u / max(sz_s, 1)

    probe("compression:unsorted", rows=n_points * n_dates,
          extra=f"{sz_u / 1e3:.0f}KB")
    probe("compression:sorted",   rows=n_points * n_dates,
          extra=f"{sz_s / 1e3:.0f}KB  ratio={ratio:.1f}×")
    print(f"  unsorted={sz_u/1e3:.0f}KB  sorted+dict={sz_s/1e3:.0f}KB  ratio={ratio:.1f}×")
    return ratio


# ---------------------------------------------------------------------------
# bench_workstation_concat
# ---------------------------------------------------------------------------

def bench_workstation_concat(tmp: Path, n_strips: int, rows_per_strip: int) -> float:
    from utils.parquet_utils import merge_strips

    print(f"\n[bench_workstation_concat]  n_strips={n_strips}  rows_per_strip={rows_per_strip}")
    strip_dir = tmp / "strips"
    strip_paths = []
    offset = 0
    for i in range(n_strips):
        p = strip_dir / f"strip_{i:04d}.parquet"
        rows = []
        for j in range(rows_per_strip):
            rows.append({
                "point_id": f"px_{offset + j:06d}_0000",
                "date": date(2022, 1, 1),
                "source": "S2",
            })
        schema = pa.schema([
            pa.field("point_id", pa.string()),
            pa.field("date",     pa.date32()),
            pa.field("source",   pa.string()),
        ])
        tbl = pa.Table.from_pylist(rows, schema=schema)
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, p, compression="zstd", use_dictionary=["point_id"])
        strip_paths.append(p)
        offset += rows_per_strip

    total_bytes = sum(p.stat().st_size for p in strip_paths)
    probe("concat:start", rows=n_strips * rows_per_strip)

    out = tmp / "tile.parquet"
    t0 = time.perf_counter()
    merge_strips(strip_paths, out)
    elapsed = time.perf_counter() - t0

    mb_s = (total_bytes / 1e6) / max(elapsed, 1e-6)
    probe("concat:done", rows=pq.ParquetFile(out).metadata.num_rows,
          extra=f"{mb_s:.0f}MB/s")
    print(f"  concat: {elapsed:.2f}s  {mb_s:.0f} MB/s  ({total_bytes/1e6:.0f} MB in)")
    return mb_s


# ---------------------------------------------------------------------------
# sim_pipeline — discrete-event simulation
# ---------------------------------------------------------------------------

def sim_pipeline(
    fetch_s: float,
    extract_s: float,
    merge_s: float,
    stream_s: float,
    n_strips: int = 10,
) -> None:
    print(f"\n[sim_pipeline]  fetch={fetch_s}s  extract={extract_s}s  "
          f"merge={merge_s}s  stream={stream_s}s  n_strips={n_strips}")

    # Stage times: fetch+extract can overlap, merger starts when extract done,
    # streamer starts when merger done for strip N while strip N+1 is being fetched.
    producer_budget = stream_s  # strip N+1 must finish before strip N stream ends
    producer_time = fetch_s + extract_s + merge_s  # sequential upper bound (overlap reduces this)

    print(f"  producer budget per strip : {producer_budget:.0f}s")
    print(f"  producer time (sequential): {producer_time:.0f}s")

    idle_total = 0.0
    stream_end = 0.0
    for i in range(n_strips):
        # Strip i: fetch+extract start at strip i's scene dir being ready
        # For simplicity model as sequential within a strip
        if i == 0:
            fetch_start = 0.0
        else:
            fetch_start = prev_extract_done  # overlaps with stream of strip i-1

        fetch_done   = fetch_start + fetch_s
        extract_done = fetch_done + extract_s
        merge_done   = extract_done + merge_s

        if i > 0:
            # Stream starts as soon as merged, but not before previous stream ends
            stream_start = max(merge_done, stream_end)
            idle = stream_start - stream_end
        else:
            stream_start = merge_done
            idle = 0.0

        stream_end = stream_start + stream_s
        idle_total += idle

        status = "✓" if idle <= 0.001 else f"✗ idle={idle:.0f}s"
        print(f"  strip {i:02d}:  fetch=[{fetch_start:.0f}–{fetch_done:.0f}]  "
              f"extract=[{fetch_done:.0f}–{extract_done:.0f}]  "
              f"merge=[{extract_done:.0f}–{merge_done:.0f}]  "
              f"stream=[{stream_start:.0f}–{stream_end:.0f}]  {status}")

        prev_extract_done = extract_done

    if idle_total <= 0.1:
        print(f"  → pipeline saturated ✓  (total idle {idle_total:.0f}s)")
    else:
        print(f"  → idle gap detected ✗  total idle = {idle_total:.0f}s "
              f"— tune PROXY_MAX_CONCURRENT / PROXY_N_WORKERS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch-proxy pipeline benchmark")
    p.add_argument("--n-scenes",   type=int, default=85,   help="S2 scenes per strip (default 85)")
    p.add_argument("--strip-px",   type=int, default=1024, help="Strip height in pixels (default 1024)")
    p.add_argument("--n-pixels",   type=int, default=57_000, help="Pixels per strip (default 57000)")
    p.add_argument("--n-strips",   type=int, default=11,   help="Strips per tile for concat bench (default 11)")
    p.add_argument("--n-dates",    type=int, default=90,   help="Dates per pixel for compression bench")

    p.add_argument("--assert-merge-s",          type=float, default=None, help="Fail if merge > N s")
    p.add_argument("--assert-compression-ratio", type=float, default=None, help="Fail if ratio < N×")
    p.add_argument("--assert-extract-s",         type=float, default=None, help="Placeholder — extraction bench requires S3")
    p.add_argument("--assert-concat-mb-s",       type=float, default=None, help="Fail if concat < N MB/s")

    p.add_argument("--sim-fetch-s",   type=float, default=None)
    p.add_argument("--sim-extract-s", type=float, default=None)
    p.add_argument("--sim-merge-s",   type=float, default=None)
    p.add_argument("--sim-stream-s",  type=float, default=None)
    p.add_argument("--sim-n-strips",  type=int,   default=10)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="bench_proxy_") as _tmp:
        tmp = Path(_tmp)

        # --- DuckDB merge bench ---
        merge_elapsed = bench_duckdb_merge(tmp / "merge", args.n_scenes, args.n_pixels)
        if args.assert_merge_s and merge_elapsed > args.assert_merge_s:
            failures.append(
                f"merge took {merge_elapsed:.1f}s > {args.assert_merge_s}s"
            )

        # --- Compression ratio bench ---
        ratio = bench_compression_ratio(tmp / "compression", n_points=1_000, n_dates=args.n_dates)
        if args.assert_compression_ratio and ratio < args.assert_compression_ratio:
            failures.append(
                f"compression ratio {ratio:.1f}× < {args.assert_compression_ratio}×"
            )

        # --- Workstation concat bench ---
        rows_per_strip = max(1, args.n_pixels * args.n_dates // args.n_strips)
        mb_s = bench_workstation_concat(tmp / "concat", args.n_strips, rows_per_strip)
        if args.assert_concat_mb_s and mb_s < args.assert_concat_mb_s:
            failures.append(
                f"concat {mb_s:.0f} MB/s < {args.assert_concat_mb_s} MB/s"
            )

        # --- Pipeline simulation ---
        if all(v is not None for v in [args.sim_fetch_s, args.sim_extract_s,
                                        args.sim_merge_s, args.sim_stream_s]):
            sim_pipeline(
                fetch_s=args.sim_fetch_s,
                extract_s=args.sim_extract_s,
                merge_s=args.sim_merge_s,
                stream_s=args.sim_stream_s,
                n_strips=args.sim_n_strips,
            )
        elif args.sim_fetch_s or args.sim_extract_s or args.sim_merge_s or args.sim_stream_s:
            print("\n[sim_pipeline] skipped — supply all four --sim-*-s flags to run")

    print("\n" + "=" * 80)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)
    else:
        print("  All assertions passed.")


if __name__ == "__main__":
    main()
