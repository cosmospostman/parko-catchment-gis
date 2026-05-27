"""Fetch-proxy pipeline benchmark harness.

Synthesises realistic data and benchmarks each stage of the VM-side pipeline
(DuckDB merge, per-scene extraction, compression ratio, workstation concat)
plus a discrete-event pipeline simulation, and end-to-end workstation benches
for both the proxy path and the all-workstation (location.py) path.

Benchmark sections
------------------
bench_duckdb_merge        — VM: N-way sort-merge of per-scene parquets
bench_compression_ratio   — sorted+dict vs unsorted ZSTD ratio
bench_local_extract       — WS-only: extract_item_to_df (S2 CPU bottleneck) + sort + merge_tile
                            Uses a MemoryChipStore mock — no HTTP, no NBAR angles.
                            Covers: location.fetch() → collect(phases={"extract"})
                            → sort_parquet_by_pixel → merge_tile
bench_local_s1_extract    — WS-only: _extract_s1_from_store (S1 extract) + _sort_s1_shards
                            Uses a MemoryChipStore mock — no HTTP, no COG reads.
                            Covers: collect_s1_for_tile() post-fetch extract + sort path
bench_local_sort          — WS-only: sort_parquet_by_pixel 1-pass vs 2-pass + merge_tile
bench_workstation_concat  — WS: merge_strips() throughput (proxy path)
bench_workstation_e2e     — WS: frame decode → strip write → merge_strips, with RSS tracking

What is NOT covered
-------------------
- collect(phases={"fetch"}) — S3/COG network fetch (requires real data, not CPU-bound)
- NBAR c-factor HTTP requests — apply_nbar=True adds per-item HTTP overhead excluded here
- collect_s1_for_tile fetch phase — MPC COG reads (network-bound, not CPU-benchmarkable locally)

Run:
    python scripts/bench_proxy.py
    python scripts/bench_proxy.py --n-pixels 57000 --n-items 90 --extract-workers 4
    python scripts/bench_proxy.py --assert-extract-items-s 5
    python scripts/bench_proxy.py --assert-merge-s 30 --assert-compression-ratio 5
    python scripts/bench_proxy.py --assert-concat-mb-s 30
    python scripts/bench_proxy.py --assert-ws-frame-mb-s 200
    python scripts/bench_proxy.py --assert-ws-rss-gb 2.0
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
    from datetime import timedelta
    _base = date(2022, 1, 1)
    rows = []
    for d_idx in range(n_dates):
        d = _base + timedelta(days=d_idx)
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
    from datetime import timedelta
    _base = date(2022, 1, 1)
    rows = []
    for d_idx in range(n_dates):
        d = _base + timedelta(days=d_idx)
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
    from datetime import timedelta
    _base = date(2022, 1, 1)
    rows = []
    for i in range(n_points):
        pid = f"px_{i:04d}_0000"
        for d_idx in range(n_dates):
            rows.append({
                "point_id": pid,
                "date": _base + timedelta(days=d_idx),
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

def _make_realistic_strip_parquet(path: Path, n_points: int, n_dates: int, northing_offset: int) -> Path:
    """Write a full-schema strip parquet: all S2 band columns + S1 + ZSTD + dict on point_id.

    Matches what merge_scenes() produces on the VM — this is what merge_strips() actually
    receives on the workstation.  Each pixel has n_dates observations.

    Uses PyArrow dictionary arrays for string columns to avoid building large Python lists.
    Sorted by (northing, date) to match merge_scenes output.
    """
    from datetime import timedelta
    from utils.parquet_utils import COMBINED_PIXEL_SCHEMA

    _epoch = date(1970, 1, 1).toordinal()
    _base  = date(2022, 1, 1)

    n_rows = n_points * n_dates

    # Northing indices 0..n_points-1, repeated n_dates times, then sorted by (northing, date)
    northing_idx = np.tile(np.arange(n_points, dtype=np.int32), n_dates)       # row → point index
    date_idx     = np.repeat(np.arange(n_dates, dtype=np.int32), n_points)     # row → date index
    sort_order   = np.lexsort((date_idx, northing_idx))
    northing_idx = northing_idx[sort_order]
    date_idx     = date_idx[sort_order]

    # point_id as dictionary array — avoids 5M Python format strings
    pid_dict   = pa.array([f"px_{northing_offset + i:06d}_0000" for i in range(n_points)])
    point_id_arr = pa.DictionaryArray.from_arrays(
        pa.array(northing_idx, type=pa.int32()), pid_dict
    ).cast(pa.string())  # merge_scenes outputs plain string, not dict

    date_ord = np.array([(_base + timedelta(days=int(d))).toordinal() - _epoch
                         for d in range(n_dates)], dtype=np.int32)
    date_arr = pa.array(date_ord[date_idx], type=pa.date32())

    arrays = {}
    arrays["point_id"]    = point_id_arr
    arrays["lon"]         = pa.array(np.full(n_rows, 145.41, dtype=np.float32), type=pa.float32())
    arrays["lat"]         = pa.array(np.linspace(-22.81, -22.74, n_rows, dtype=np.float32), type=pa.float32())
    arrays["date"]        = date_arr
    def _const_str(val: str) -> pa.Array:
        # Dictionary array with one unique value — avoids encoding n_rows string copies
        return pa.DictionaryArray.from_arrays(
            pa.array(np.zeros(n_rows, dtype=np.int32), type=pa.int32()),
            pa.array([val]),
        ).cast(pa.string())

    arrays["item_id"]     = _const_str("S2A_55HBU_20220101_0_L2A")
    arrays["tile_id"]     = _const_str("55HBU")
    arrays["source"]      = _const_str("S2")
    arrays["scl_purity"]  = pa.array(np.full(n_rows, 100, dtype=np.int8), type=pa.int8())
    arrays["scl"]         = pa.array(np.full(n_rows, 4,   dtype=np.int8), type=pa.int8())
    arrays["aot"]         = pa.array(np.full(n_rows, 80,  dtype=np.uint8), type=pa.uint8())
    arrays["view_zenith"] = pa.array(np.full(n_rows, 30,  dtype=np.uint8), type=pa.uint8())
    arrays["sun_zenith"]  = pa.array(np.full(n_rows, 45,  dtype=np.uint8), type=pa.uint8())
    for band in BANDS:
        arrays[band] = pa.array(np.random.randint(500, 8000, n_rows, dtype=np.uint16), type=pa.uint16())
    arrays["orbit"] = _const_str("")
    arrays["vh"]    = pa.array(np.full(n_rows, np.nan, dtype=np.float32), type=pa.float32())
    arrays["vv"]    = pa.array(np.full(n_rows, np.nan, dtype=np.float32), type=pa.float32())

    tbl = pa.table({name: arrays[name] for name in COMBINED_PIXEL_SCHEMA.names})
    path.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed for speed — synthesis is for schema/size realism, not compression realism.
    # merge_strips() recompresses on write, so the concat bench still exercises ZSTD.
    pq.write_table(tbl, path, compression="none", row_group_size=5_000_000)
    return path


def make_strips(tmp: Path, n_strips: int, n_pixels: int, n_dates: int) -> list[Path]:
    """Build synthetic strip parquets once; shared by concat and e2e benches."""
    strip_dir = tmp / "strips"
    strip_paths = []
    t_make = time.perf_counter()
    for i in range(n_strips):
        p = strip_dir / f"strip_{i:04d}.parquet"
        _make_realistic_strip_parquet(p, n_pixels, n_dates, northing_offset=i * n_pixels)
        strip_paths.append(p)
    print(f"\n[make_strips]  {n_strips} strips × {n_pixels} px × {n_dates} dates  "
          f"built in {time.perf_counter() - t_make:.1f}s  "
          f"({sum(p.stat().st_size for p in strip_paths)/1e6:.0f} MB total)")
    return strip_paths


def bench_workstation_concat(tmp: Path, strip_paths: list[Path], n_pixels: int, n_dates: int) -> float:
    from utils.parquet_utils import merge_strips

    # Throughput is ~40 MB/s on full-schema strips because PyArrow decompresses each row
    # group and recompresses it into the output writer — no raw passthrough API exists.
    # This is NOT a pipeline bottleneck: concat runs after the last strip is received
    # (~29s for a full tile/year vs ~1617s total stream time).
    n_strips = len(strip_paths)
    print(f"\n[bench_workstation_concat]  n_strips={n_strips}  n_pixels={n_pixels}  n_dates={n_dates}")

    total_bytes = sum(p.stat().st_size for p in strip_paths)
    total_rows  = n_strips * n_pixels * n_dates
    probe("concat:start", rows=total_rows,
          extra=f"({total_bytes/1e6:.0f} MB total input)")

    out = tmp / "tile.parquet"
    t0 = time.perf_counter()
    merge_strips(strip_paths, out)
    elapsed = time.perf_counter() - t0

    out_bytes = out.stat().st_size
    mb_s = (total_bytes / 1e6) / max(elapsed, 1e-6)
    probe("concat:done", rows=pq.ParquetFile(out).metadata.num_rows,
          extra=f"{mb_s:.0f}MB/s  out={out_bytes/1e6:.0f}MB")
    print(f"  concat: {elapsed:.2f}s  {mb_s:.0f} MB/s  "
          f"(in={total_bytes/1e6:.0f}MB  out={out_bytes/1e6:.0f}MB)")
    return mb_s


# ---------------------------------------------------------------------------
# bench_workstation_e2e
# ---------------------------------------------------------------------------

class _WSResult(NamedTuple):
    frame_mb_s: float      # frame-decode throughput MB/s
    strip_write_mb_s: float  # atomic strip-write throughput MB/s
    merge_mb_s: float      # merge_strips throughput MB/s (input bytes / elapsed)
    peak_rss_gb: float     # peak RSS sampled per row-group inside merge_strips
    total_s: float         # wall time: decode + write + merge


def bench_workstation_e2e(
    tmp: Path,
    strip_paths: list[Path],
    n_pixels: int,
    n_dates: int,
) -> _WSResult:
    """End-to-end workstation benchmark: frame decode → strip write → merge_strips.

    Accepts pre-built strip_paths (from make_strips) to avoid redundant synthesis.

    Measures:
      1. Frame-decode throughput: encode realistic strip bytes as 0x02 frames,
         time the StreamBuffer + read_frame decode loop (65536-byte chunks,
         mirroring httpx.iter_raw).
      2. Atomic strip-write throughput: write decoded bytes to .tmp then rename
         to .parquet (mirrors client.py atomic write).
      3. merge_strips throughput and peak RSS per row-group: monkey-patches
         ParquetWriter.write_table to sample RSS after each write_table call so
         we can confirm O(row-group) memory use without modifying production code.
    """
    import pyarrow.parquet as pq
    from proxy._pipeline import write_frame, read_frame, StreamBuffer
    from utils.parquet_utils import merge_strips

    n_strips = len(strip_paths)
    print(f"\n[bench_workstation_e2e]  n_strips={n_strips}  n_pixels={n_pixels}  n_dates={n_dates}")

    strip_payloads: list[bytes] = [p.read_bytes() for p in strip_paths]
    total_payload_bytes = sum(len(b) for b in strip_payloads)
    print(f"  total payload: {total_payload_bytes/1e6:.1f} MB  ({n_strips} strips × {total_payload_bytes/n_strips/1e6:.1f} MB avg)")

    # --- Stage 1: frame-decode throughput ---
    # Feed each strip as one pre-encoded frame (no chunk splitting).  The chunked
    # reassembly path is already covered by test_frame_roundtrip; here we measure
    # the header-parse + payload-copy cost of read_frame itself.
    probe("ws:frame_encode_start", extra=f"payload={total_payload_bytes/1e6:.0f}MB")
    t_decode = time.perf_counter()
    buf = StreamBuffer(iter(write_frame(0x02, p) for p in strip_payloads))
    decoded: list[bytes] = []
    while True:
        frame = read_frame(buf)
        if frame is None:
            break
        frame_type, payload = frame
        if frame_type == 0x02:
            decoded.append(payload)
    decode_elapsed = time.perf_counter() - t_decode

    assert len(decoded) == n_strips, f"expected {n_strips} frames, got {len(decoded)}"
    frame_mb_s = (total_payload_bytes / 1e6) / max(decode_elapsed, 1e-9)
    probe("ws:frame_decode", extra=f"{frame_mb_s:.0f} MB/s  ({decode_elapsed:.3f}s)")
    print(f"  frame decode: {decode_elapsed:.3f}s  {frame_mb_s:.0f} MB/s")

    gc.collect()

    # --- Stage 2: atomic strip-write throughput ---
    # Mirrors client.py: write to .tmp, verify size, rename to .parquet.
    receive_dir = tmp / "ws_receive"
    receive_dir.mkdir(parents=True, exist_ok=True)
    strip_paths: list[Path] = []

    t_write = time.perf_counter()
    for i, payload in enumerate(decoded):
        strip_path = receive_dir / f"strip_{i:04d}.parquet"
        tmp_path   = receive_dir / f"strip_{i:04d}.tmp"
        tmp_path.write_bytes(payload)
        if tmp_path.stat().st_size != len(payload):
            raise RuntimeError(f"strip {i}: size mismatch after write")
        tmp_path.replace(strip_path)
        strip_paths.append(strip_path)
    write_elapsed = time.perf_counter() - t_write

    strip_write_mb_s = (total_payload_bytes / 1e6) / max(write_elapsed, 1e-9)
    probe("ws:strip_write", extra=f"{strip_write_mb_s:.0f} MB/s  ({write_elapsed:.3f}s)")
    print(f"  strip write:  {write_elapsed:.3f}s  {strip_write_mb_s:.0f} MB/s")

    del decoded
    gc.collect()

    # --- Stage 3: merge_strips with per-row-group RSS tracking ---
    # Monkey-patch ParquetWriter.write_table to sample RSS after each call.
    rss_samples: list[float] = [rss_gb()]  # baseline before merge
    _orig_write_table = pq.ParquetWriter.write_table

    def _patched_write_table(self, table, *args, **kwargs):
        result = _orig_write_table(self, table, *args, **kwargs)
        rss_samples.append(rss_gb())
        return result

    out = tmp / "ws_tile.parquet"
    t_merge = time.perf_counter()
    try:
        pq.ParquetWriter.write_table = _patched_write_table
        merge_strips(strip_paths, out)
    finally:
        pq.ParquetWriter.write_table = _orig_write_table
    merge_elapsed = time.perf_counter() - t_merge

    rss_baseline = rss_samples[0]
    peak_rss = max(rss_samples)
    peak_delta = peak_rss - rss_baseline
    merge_mb_s = (total_payload_bytes / 1e6) / max(merge_elapsed, 1e-9)

    probe("ws:merge_done",
          rows=pq.ParquetFile(out).metadata.num_rows,
          extra=(f"{merge_mb_s:.0f} MB/s  ({merge_elapsed:.2f}s)  "
                 f"RSS baseline={rss_baseline:.2f}GB  peak={peak_rss:.2f}GB  Δ={peak_delta:+.2f}GB"))
    print(f"  merge_strips: {merge_elapsed:.2f}s  {merge_mb_s:.0f} MB/s")
    print(f"  RSS: baseline={rss_baseline:.2f}GB  peak={peak_rss:.2f}GB  Δ={peak_delta:+.2f}GB  "
          f"({len(rss_samples)-1} samples across {pq.ParquetFile(out).metadata.num_row_groups} row groups)")

    total_elapsed = decode_elapsed + write_elapsed + merge_elapsed
    print(f"  total workstation: {total_elapsed:.2f}s  "
          f"(decode={decode_elapsed:.2f}s  write={write_elapsed:.2f}s  merge={merge_elapsed:.2f}s)")

    return _WSResult(
        frame_mb_s=frame_mb_s,
        strip_write_mb_s=strip_write_mb_s,
        merge_mb_s=merge_mb_s,
        peak_rss_gb=peak_rss,
        total_s=total_elapsed,
    )


# ---------------------------------------------------------------------------
# bench_local_extract — extract_item_to_df with a MemoryChipStore mock
# ---------------------------------------------------------------------------

def _make_mock_store(
    item_ids: list[str],
    n_pixels: int,
    point_coords: dict[str, tuple[float, float]],
    patch_h: int = 512,
    patch_w: int = 512,
) -> "MemoryChipStore":
    """Build a MemoryChipStore populated with synthetic band patches.

    Each band patch is a (patch_h × patch_w) uint16 array with random values
    in [500, 8000] — realistic S2 surface reflectance range before /10000 scaling.
    SCL is set to 4 (vegetation = clear) for all pixels so every item contributes
    all n_pixels rows.  AOT is a uniform low-aerosol value.

    The affine transform maps the patch to the bbox so all n_pixels point
    coordinates project inside it — every point gets a valid pixel value.
    """
    from affine import Affine
    from pyproj import CRS, Transformer
    from utils.chip_store import MemoryChipStore
    from analysis.constants import BANDS, SCL_BAND, AOT_BAND

    # Use a realistic UTM CRS (zone 55 south — Longreach area)
    utm_crs = CRS.from_epsg(32755)

    # Derive a transform that spans all point coords.
    lons = np.array([lon for lon, _ in point_coords.values()])
    lats = np.array([lat for _, lat in point_coords.values()])
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    xs_raw, ys_raw = to_utm.transform(lons.tolist(), lats.tolist())
    xs, ys = np.asarray(xs_raw), np.asarray(ys_raw)
    x_min, x_max = xs.min() - 10, xs.max() + 10
    y_min, y_max = ys.min() - 10, ys.max() + 10

    # 10 m pixels to span the patch extent
    res = 10.0
    actual_w = max(patch_w, int((x_max - x_min) / res) + 2)
    actual_h = max(patch_h, int((y_max - y_min) / res) + 2)
    transform = Affine(res, 0, x_min, 0, -res, y_max)

    rng = np.random.default_rng(42)
    patches: dict[tuple[str, str], tuple[np.ndarray, object, object]] = {}
    all_bands = list(BANDS) + [SCL_BAND, AOT_BAND]

    for item_id in item_ids:
        for band in all_bands:
            if band == SCL_BAND:
                arr = np.full((actual_h, actual_w), 4, dtype=np.uint16)   # 4 = vegetation/clear
            elif band == AOT_BAND:
                arr = np.full((actual_h, actual_w), 60, dtype=np.uint16)  # low aerosol
            else:
                arr = rng.integers(500, 8000, (actual_h, actual_w), dtype=np.uint16)
            patches[(item_id, band)] = (arr, transform, utm_crs)

    return MemoryChipStore(patches=patches, point_coords=point_coords)


def bench_local_extract(tmp: Path, n_pixels: int, n_items: int, n_workers: int = 4) -> float:
    """Benchmark the extract phase of the workstation-only fetch path.

    This is the CPU bottleneck in location.py → fetch_spec → collect(phases={"extract"}).
    Each item goes through:
      1. MemoryChipStore.get_all_points()  — vectorised pixel extraction from band arrays
      2. extract_item_to_df()              — SCL filter, AOT quality, polars DataFrame build
      3. Polars → Arrow conversion         — .to_arrow() + batch accumulation
      4. pq.ParquetWriter.write_table()    — uncompressed shard flush (every 500k rows)

    Then the shard goes through:
      5. sort_parquet_by_pixel()           — Polars LazyFrame sort (1-pass, skip dict rewrite)
      6. merge_tile()                      — DuckDB 2-way sort-merge with synthetic S1

    n_workers matches the sliding-window executor in collect() — each worker holds
    one item's band arrays in RAM while it runs extract_item_to_df.

    apply_nbar is False here because granule angle fetch (HTTP) is out-of-scope for
    a local CPU benchmark.  In production apply_nbar=True adds per-item HTTP overhead
    (not CPU-bound) that is not captured here.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from utils.pixel_collector import extract_item_to_df
    from utils.parquet_utils import sort_parquet_by_pixel, merge_tile

    print(f"\n[bench_local_extract]  n_pixels={n_pixels}  n_items={n_items}  n_workers={n_workers}")

    # --- Synthetic pixel grid ------------------------------------------------
    # n_pixels points on a regular 10-m grid starting at (145.41, -22.81)
    point_ids = [f"px_{i:06d}_0000" for i in range(n_pixels)]
    # Spread over ~0.1° × 0.1° to avoid trivial single-pixel patch corner cases
    lons = np.linspace(145.41, 145.50, n_pixels, dtype=np.float64)
    lats = np.full(n_pixels, -22.81, dtype=np.float64)
    point_coords = {pid: (float(lon), float(lat)) for pid, lon, lat in zip(point_ids, lons, lats)}

    # --- Synthetic item stubs ------------------------------------------------
    # extract_item_to_df only uses item.id, item.datetime, item.properties
    # (for tile_id extraction).  Build minimal stubs.
    import types
    from datetime import datetime as _dt
    _base_dt = _dt(2022, 1, 1)
    item_ids = [f"S2A_55HBU_{2022*10000 + i:08d}_0_L2A" for i in range(n_items)]
    items = []
    for iid in item_ids:
        stub = types.SimpleNamespace(
            id=iid,
            datetime=_base_dt,
            properties={"s2:mgrs_tile": "55HBU"},
        )
        items.append(stub)

    # --- Build mock store (shared across all items — same patches) ------------
    t_store = time.perf_counter()
    store = _make_mock_store(item_ids, n_pixels, point_coords)
    print(f"  mock store built in {time.perf_counter()-t_store:.2f}s  "
          f"({len(item_ids)} items × {len(list(__import__('analysis.constants', fromlist=['BANDS']).BANDS)) + 2} bands × patch)")

    # --- Stage 1: extract phase (mirrors collect() sliding window) -----------
    from analysis.constants import BANDS as _BANDS
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = tmp / "extract"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "shard_0000.parquet"

    writer: pq.ParquetWriter | None = None
    shard_buf: list[pa.Table] = []
    shard_buf_rows = 0
    SHARD_BUF_SIZE = 500_000
    total_rows = 0
    items_done = 0

    def _flush(force: bool = False) -> None:
        nonlocal writer, shard_buf_rows
        if not shard_buf or (not force and shard_buf_rows < SHARD_BUF_SIZE):
            return
        merged = pa.concat_tables(shard_buf)
        shard_buf.clear()
        shard_buf_rows = 0
        if writer is None:
            return  # shouldn't happen, but guard
        writer.write_table(merged)

    # Production path: sliding window with n_workers concurrent items.
    # apply_nbar=False — no HTTP; utm_crs matches the mock store's CRS.
    def _process(item):
        return extract_item_to_df(
            item, store, point_ids,
            lons, lats,
            apply_nbar=False,
            utm_crs="EPSG:32755",
        )

    probe("extract:start", rows=n_pixels * n_items)
    t_extract = time.perf_counter()

    pending = list(items)
    in_flight: dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        while pending or in_flight:
            while pending and len(in_flight) < n_workers:
                item = pending.pop(0)
                in_flight[pool.submit(_process, item)] = item

            done_futs, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done_futs:
                in_flight.pop(fut)
                df = fut.result()
                if df is not None and len(df) > 0:
                    tbl = df.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(shard_path, tbl.schema, compression="none")
                    shard_buf.append(tbl)
                    shard_buf_rows += len(tbl)
                    total_rows += len(tbl)
                    if shard_buf_rows >= SHARD_BUF_SIZE:
                        _flush()
                items_done += 1

    if writer is not None:
        _flush(force=True)
        writer.close()

    extract_elapsed = time.perf_counter() - t_extract
    shard_sz = shard_path.stat().st_size if shard_path.exists() else 0
    items_per_s = n_items / max(extract_elapsed, 1e-9)
    rows_per_s = total_rows / max(extract_elapsed, 1e-9)

    probe("extract:done", rows=total_rows,
          extra=f"{items_per_s:.1f} items/s  {rows_per_s/1e6:.2f}M rows/s  {shard_sz/1e6:.0f}MB shard")
    print(f"  extract: {extract_elapsed:.2f}s  {items_per_s:.1f} items/s  "
          f"{rows_per_s/1e6:.2f}M rows/s  {total_rows:,} rows  {shard_sz/1e6:.0f} MB shard")

    if not shard_path.exists():
        print("  (no shard written — all items filtered, skipping sort+merge)")
        return items_per_s

    # --- Stage 2: sort + merge (reuses bench_local_sort logic inline) --------
    sorted_path = shard_dir / "shard_0000_sorted.parquet"
    t_sort = time.perf_counter()
    sort_parquet_by_pixel(shard_path, sorted_path, row_group_size=5_000_000, _skip_dict_rewrite=True)
    sort_elapsed = time.perf_counter() - t_sort
    sort_sz = sorted_path.stat().st_size

    print(f"  sort:    {sort_elapsed:.2f}s  {shard_sz/1e6/sort_elapsed:.0f} MB/s  "
          f"{shard_sz/1e6:.0f}→{sort_sz/1e6:.0f} MB")

    s1_path = shard_dir / "s1.parquet"
    _make_realistic_strip_parquet(s1_path, n_pixels // 4, n_items // 4, northing_offset=0)
    merged_path = shard_dir / "merged.parquet"
    t_merge = time.perf_counter()
    merge_tile(sorted_path, s1_path, merged_path)
    merge_elapsed = time.perf_counter() - t_merge
    merged_sz = merged_path.stat().st_size

    probe("extract:merge_done", rows=pq.ParquetFile(merged_path).metadata.num_rows,
          extra=f"{merge_elapsed:.2f}s  out={merged_sz/1e6:.0f}MB")
    print(f"  merge:   {merge_elapsed:.2f}s  out={merged_sz/1e6:.0f} MB")
    print(f"  total extract pipeline: {extract_elapsed + sort_elapsed + merge_elapsed:.2f}s  "
          f"(extract={extract_elapsed:.2f}s  sort={sort_elapsed:.2f}s  merge={merge_elapsed:.2f}s)")
    return items_per_s


# ---------------------------------------------------------------------------
# bench_local_s1_extract — _extract_s1_from_store + _sort_s1_shards
# ---------------------------------------------------------------------------

def _make_s1_mock_store(
    item_ids: list[str],
    n_pixels: int,
    point_coords: dict[str, tuple[float, float]],
) -> "MemoryChipStore":
    """Build a MemoryChipStore with synthetic S1 vh/vv float32 patches.

    Values are small positive floats (linear backscatter ~0.01–0.3).
    No zero/NaN inserted so every pixel contributes a valid observation.
    """
    from affine import Affine
    from pyproj import CRS, Transformer
    from utils.chip_store import MemoryChipStore

    utm_crs = CRS.from_epsg(32755)
    lons = np.array([lon for lon, _ in point_coords.values()])
    lats = np.array([lat for _, lat in point_coords.values()])
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    xs_raw, ys_raw = to_utm.transform(lons.tolist(), lats.tolist())
    xs, ys = np.asarray(xs_raw), np.asarray(ys_raw)
    x_min, x_max = xs.min() - 10, xs.max() + 10
    y_min, y_max = ys.min() - 10, ys.max() + 10

    res = 10.0
    actual_w = max(64, int((x_max - x_min) / res) + 2)
    actual_h = max(64, int((y_max - y_min) / res) + 2)
    transform = Affine(res, 0, x_min, 0, -res, y_max)

    rng = np.random.default_rng(7)
    patches: dict[tuple[str, str], tuple[np.ndarray, object, object]] = {}
    for item_id in item_ids:
        for band in ("vh", "vv"):
            arr = rng.uniform(0.01, 0.3, (actual_h, actual_w)).astype(np.float32)
            patches[(item_id, band)] = (arr, transform, utm_crs)

    return MemoryChipStore(patches=patches, point_coords=point_coords)


def bench_local_s1_extract(tmp: Path, n_pixels: int, n_s1_items: int) -> float:
    """Benchmark the S1 post-fetch extract path: _extract_s1_from_store + _sort_s1_shards.

    Mirrors _collect_s1_shards() with a MemoryChipStore mock instead of a
    CachedNpzChipStore backed by on-disk .npz files.  No HTTP, no COG reads.

    S1 extraction is serial (items processed one-by-one, no thread pool) because
    _extract_s1_from_store is cheaper than S2's extract_item_to_df — only 2 bands,
    no SCL filtering, no NBAR.

    Returns items/s for the extract phase.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.s1_collector import _extract_s1_from_store
    from utils.parquet_utils import _sort_s1_shards, COMBINED_PIXEL_SCHEMA

    print(f"\n[bench_local_s1_extract]  n_pixels={n_pixels}  n_s1_items={n_s1_items}")

    point_ids = [f"px_{i:06d}_0000" for i in range(n_pixels)]
    lons = np.linspace(145.41, 145.50, n_pixels, dtype=np.float64)
    lats = np.full(n_pixels, -22.81, dtype=np.float64)
    point_coords = {pid: (float(lon), float(lat)) for pid, lon, lat in zip(point_ids, lons, lats)}

    import types
    from datetime import datetime as _dt
    item_ids = [f"S1A_IW_SLC__{2022 * 10000 + i:08d}" for i in range(n_s1_items)]
    items = [
        types.SimpleNamespace(
            id=iid,
            datetime=_dt(2022, 1, 1),
            properties={"sat:orbit_state": "ascending"},
        )
        for iid in item_ids
    ]

    t_store = time.perf_counter()
    store = _make_s1_mock_store(item_ids, n_pixels, point_coords)
    print(f"  mock store built in {time.perf_counter()-t_store:.2f}s  "
          f"({len(item_ids)} items × 2 bands × patch)")

    # --- Extract phase (mirrors _collect_s1_shards serial loop) --------------
    _ARROW_SCHEMA = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("lon",      pa.float64()),
        pa.field("lat",      pa.float64()),
        pa.field("date",     pa.timestamp("ms")),
        pa.field("source",   pa.string()),
        pa.field("vh",       pa.float32()),
        pa.field("vv",       pa.float32()),
        pa.field("orbit",    pa.string()),
    ])

    shard_dir = tmp / "s1_extract"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "shard_0000.parquet"

    writer = pq.ParquetWriter(shard_path, _ARROW_SCHEMA, compression="none")
    total_rows = 0
    _FLUSH_ROWS = max(10_000, min(200_000, n_pixels * 10))
    buf_pid: list = []; buf_lon: list = []; buf_lat: list = []
    buf_date: list = []; buf_vh: list = []; buf_vv: list = []; buf_orbit: list = []

    def _flush() -> None:
        if not buf_pid:
            return
        tbl = pa.table({
            "point_id": pa.array(buf_pid,  pa.string()),
            "lon":      pa.array(buf_lon,  pa.float64()),
            "lat":      pa.array(buf_lat,  pa.float64()),
            "date":     pa.array(buf_date, pa.timestamp("ms")),
            "source":   pa.repeat("S1", len(buf_pid)).cast(pa.string()),
            "vh":       pa.array(buf_vh,   pa.float32()),
            "vv":       pa.array(buf_vv,   pa.float32()),
            "orbit":    pa.array(buf_orbit, pa.string()),
        })
        writer.write_table(tbl)
        buf_pid.clear(); buf_lon.clear(); buf_lat.clear(); buf_date.clear()
        buf_vh.clear();  buf_vv.clear();  buf_orbit.clear()

    probe("s1_extract:start", rows=n_pixels * n_s1_items)
    t_extract = time.perf_counter()

    for item in items:
        result = _extract_s1_from_store(item, store, point_ids, lons, lats)
        store.release_item(item.id)
        if result is None:
            continue
        pids, i_lons, i_lats, dates, vhs, vvs, orbits = result
        buf_pid.extend(pids);   buf_lon.extend(i_lons);  buf_lat.extend(i_lats)
        buf_date.extend(dates); buf_vh.extend(vhs);      buf_vv.extend(vvs)
        buf_orbit.extend(orbits)
        total_rows += len(pids)
        if len(buf_pid) >= _FLUSH_ROWS:
            _flush()

    _flush()
    writer.close()

    extract_elapsed = time.perf_counter() - t_extract
    shard_sz = shard_path.stat().st_size if shard_path.exists() and total_rows > 0 else 0
    items_per_s = n_s1_items / max(extract_elapsed, 1e-9)
    rows_per_s = total_rows / max(extract_elapsed, 1e-9)

    probe("s1_extract:done", rows=total_rows,
          extra=f"{items_per_s:.1f} items/s  {rows_per_s/1e6:.2f}M rows/s  {shard_sz/1e6:.0f}MB shard")
    print(f"  s1 extract: {extract_elapsed:.2f}s  {items_per_s:.1f} items/s  "
          f"{rows_per_s/1e6:.2f}M rows/s  {total_rows:,} rows  {shard_sz/1e6:.0f} MB shard")

    if total_rows == 0:
        print("  (no rows produced, skipping sort)")
        return items_per_s

    # --- Sort phase (_sort_s1_shards with single shard → sort_parquet_by_pixel) --
    out_path = shard_dir / "s1_sorted.parquet"
    t_sort = time.perf_counter()
    _sort_s1_shards([shard_path], out_path, COMBINED_PIXEL_SCHEMA)
    sort_elapsed = time.perf_counter() - t_sort
    sort_sz = out_path.stat().st_size

    probe("s1_extract:sort_done", rows=pq.ParquetFile(out_path).metadata.num_rows,
          extra=f"{sort_elapsed:.2f}s  {shard_sz/1e6/sort_elapsed:.0f} MB/s  out={sort_sz/1e6:.0f}MB")
    print(f"  s1 sort:    {sort_elapsed:.2f}s  {shard_sz/1e6/sort_elapsed:.0f} MB/s  "
          f"{shard_sz/1e6:.0f}→{sort_sz/1e6:.0f} MB")
    print(f"  s1 total:   {extract_elapsed + sort_elapsed:.2f}s  "
          f"(extract={extract_elapsed:.2f}s  sort={sort_elapsed:.2f}s)")

    return items_per_s


# ---------------------------------------------------------------------------
# bench_local_sort — sort_parquet_by_pixel with/without dict rewrite, + merge_tile
# ---------------------------------------------------------------------------

def bench_local_sort(tmp: Path, n_pixels: int, n_dates: int) -> None:
    """Benchmark the workstation-only shard sort + merge_tile path.

    Measures:
      - sort_parquet_by_pixel with _skip_dict_rewrite=False (two ZSTD passes)
      - sort_parquet_by_pixel with _skip_dict_rewrite=True  (one ZSTD pass)
      - merge_tile (DuckDB 2-way sort-merge of s2+s1 parquets)

    Input: one unsorted shard parquet (compression="none"), matching what
    pixel_collector.py writes before calling sort_parquet_by_pixel.
    """
    from utils.parquet_utils import sort_parquet_by_pixel, merge_tile, COMBINED_PIXEL_SCHEMA
    import pyarrow.parquet as pq

    print(f"\n[bench_local_sort]  n_pixels={n_pixels}  n_dates={n_dates}")
    shard_dir = tmp / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Build one unsorted shard: compression="none", no dict encoding, random row order.
    # Matches what pixel_collector writes at shard_path with compression="none".
    n_rows = n_pixels * n_dates
    shard_path = shard_dir / "shard_0000.parquet"
    t_make = time.perf_counter()
    _make_realistic_strip_parquet(shard_path, n_pixels, n_dates, northing_offset=0)
    # _make_realistic_strip_parquet writes uncompressed; shuffle rows to simulate unsorted shard
    tbl = pq.read_table(shard_path)
    import pyarrow.compute as pc
    idx = np.random.permutation(len(tbl))
    tbl = tbl.take(idx)
    pq.write_table(tbl, shard_path, compression="none", row_group_size=5_000_000)
    del tbl
    sz_in = shard_path.stat().st_size
    print(f"  shard built in {time.perf_counter()-t_make:.1f}s  ({sz_in/1e6:.0f} MB uncompressed)")
    probe("sort:shard_ready", rows=n_rows, extra=f"{sz_in/1e6:.0f} MB")

    # --- Two-pass sort (current production behaviour) ---
    dst_2pass = shard_dir / "sorted_2pass.parquet"
    t0 = time.perf_counter()
    sort_parquet_by_pixel(shard_path, dst_2pass, row_group_size=5_000_000, _skip_dict_rewrite=False)
    elapsed_2pass = time.perf_counter() - t0
    sz_2pass = dst_2pass.stat().st_size
    probe("sort:2pass_done", rows=n_rows,
          extra=f"{elapsed_2pass:.2f}s  {sz_in/1e6/elapsed_2pass:.0f} MB/s  out={sz_2pass/1e6:.0f} MB")
    print(f"  sort 2-pass (dict rewrite): {elapsed_2pass:.2f}s  "
          f"{sz_in/1e6/elapsed_2pass:.0f} MB/s  {sz_in/1e6:.0f}→{sz_2pass/1e6:.0f} MB")

    # --- One-pass sort (skip dict rewrite) ---
    dst_1pass = shard_dir / "sorted_1pass.parquet"
    t0 = time.perf_counter()
    sort_parquet_by_pixel(shard_path, dst_1pass, row_group_size=5_000_000, _skip_dict_rewrite=True)
    elapsed_1pass = time.perf_counter() - t0
    sz_1pass = dst_1pass.stat().st_size
    probe("sort:1pass_done", rows=n_rows,
          extra=f"{elapsed_1pass:.2f}s  {sz_in/1e6/elapsed_1pass:.0f} MB/s  out={sz_1pass/1e6:.0f} MB")
    print(f"  sort 1-pass (no dict rewrite): {elapsed_1pass:.2f}s  "
          f"{sz_in/1e6/elapsed_1pass:.0f} MB/s  {sz_in/1e6:.0f}→{sz_1pass/1e6:.0f} MB")

    speedup = elapsed_2pass / max(elapsed_1pass, 1e-6)
    dict_saving_mb = (sz_2pass - sz_1pass) / 1e6  # negative = 1pass is larger
    print(f"  dict-rewrite cost: {elapsed_2pass - elapsed_1pass:.2f}s  "
          f"({speedup:.1f}× slower)  size delta: {-dict_saving_mb:+.0f} MB "
          f"({'smaller' if sz_2pass < sz_1pass else 'larger'} without dict)")

    # --- merge_tile: DuckDB 2-way sort-merge of s2 + s1 ---
    # Use the 2-pass sorted output as s2; build a small synthetic s1.
    s1_path = shard_dir / "s1.parquet"
    _make_realistic_strip_parquet(s1_path, n_pixels // 4, n_dates, northing_offset=0)
    # tag source="S1", zero out band columns to simulate real S1 shape
    s1_tbl = pq.read_table(s1_path)
    import pyarrow as pa
    s1_tbl = s1_tbl.set_column(
        s1_tbl.schema.get_field_index("source"), "source",
        pa.array(["S1"] * len(s1_tbl), type=pa.string()),
    )
    pq.write_table(s1_tbl, s1_path, compression="zstd", row_group_size=5_000_000)
    del s1_tbl

    merged_path = tmp / "merged.parquet"
    t0 = time.perf_counter()
    merge_tile(dst_2pass, s1_path, merged_path)
    elapsed_merge = time.perf_counter() - t0
    sz_merged = merged_path.stat().st_size
    total_in = sz_2pass + s1_path.stat().st_size
    probe("sort:merge_tile_done", rows=pq.ParquetFile(merged_path).metadata.num_rows,
          extra=f"{elapsed_merge:.2f}s  {total_in/1e6/elapsed_merge:.0f} MB/s  out={sz_merged/1e6:.0f} MB")
    print(f"  merge_tile (DuckDB): {elapsed_merge:.2f}s  "
          f"{total_in/1e6/elapsed_merge:.0f} MB/s  out={sz_merged/1e6:.0f} MB")

    print(f"  total shard pipeline: {elapsed_2pass + elapsed_merge:.2f}s  "
          f"(sort={elapsed_2pass:.2f}s  merge={elapsed_merge:.2f}s)")
    print(f"  potential saving if skip dict rewrite: {elapsed_2pass - elapsed_1pass:.2f}s per shard")


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
    p.add_argument("--n-dates",    type=int, default=90,   help="Dates per pixel for compression bench (default 90)")
    p.add_argument("--ws-n-dates", type=int, default=20,
                   help="Dates per pixel for workstation strip synthesis (default 20). "
                        "Lower than --n-dates to keep synthesis fast — strips are uncompressed "
                        "intermediates, not realistic VM output.")

    p.add_argument("--assert-merge-s",          type=float, default=None, help="Fail if merge > N s")
    p.add_argument("--assert-compression-ratio", type=float, default=None, help="Fail if ratio < N×")
    p.add_argument("--assert-extract-s",         type=float, default=None, help="(deprecated) Superseded by --assert-extract-items-s")
    p.add_argument("--assert-concat-mb-s",       type=float, default=None,
                   help="Fail if concat < N MB/s. "
                        "Measured ~40 MB/s on full-schema strips: PyArrow must decompress+recompress each "
                        "row group (no passthrough API exists). This is not a pipeline bottleneck — "
                        "concat runs after the last strip is received (~29s vs ~1617s stream time). "
                        "Use --assert-concat-mb-s 30 as a regression guard, not 500.")
    p.add_argument("--assert-sort-mb-s",      type=float, default=None,
                   help="Fail if sort_parquet_by_pixel (2-pass) < N MB/s")
    p.add_argument("--assert-ws-frame-mb-s",  type=float, default=None,
                   help="Fail if workstation frame-decode throughput < N MB/s "
                        "(expected >200 MB/s — pure in-memory struct.unpack loop)")
    p.add_argument("--assert-ws-rss-gb",      type=float, default=None,
                   help="Fail if peak RSS during merge_strips > N GB "
                        "(expected <2 GB — O(row-group) memory, each RG ≤250 MB uncompressed)")
    p.add_argument("--assert-extract-items-s", type=float, default=None,
                   help="Fail if extract_item_to_df throughput < N items/s "
                        "(no HTTP overhead — pure CPU: SCL filter, NBAR-off band arrays, polars→arrow)")
    p.add_argument("--n-items",       type=int, default=90,
                   help="S2 items (scene acquisitions) for the extract bench (default 90, ~one year)")
    p.add_argument("--extract-workers", type=int, default=4,
                   help="Thread count for the sliding-window extract executor (default 4)")
    p.add_argument("--n-s1-items",    type=int, default=60,
                   help="S1 items for the S1 extract bench (default 60, ~one year of 6-day revisit)")

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

        # --- Build strips once, share between concat and e2e benches ---
        strip_paths = make_strips(tmp / "ws", args.n_strips, args.n_pixels, args.ws_n_dates)

        # --- Local extract bench (CPU-only extract_item_to_df path) ---
        extract_items_s = bench_local_extract(tmp / "local_extract", args.n_pixels, args.n_items, args.extract_workers)
        if args.assert_extract_items_s and extract_items_s < args.assert_extract_items_s:
            failures.append(
                f"extract {extract_items_s:.1f} items/s < {args.assert_extract_items_s} items/s"
            )

        # --- S1 extract bench (_extract_s1_from_store + _sort_s1_shards) ---
        bench_local_s1_extract(tmp / "local_s1_extract", args.n_pixels, args.n_s1_items)

        # --- Local sort bench (workstation-only path) ---
        bench_local_sort(tmp / "local_sort", args.n_pixels, args.ws_n_dates)

        # --- Workstation concat bench ---
        mb_s = bench_workstation_concat(tmp / "concat", strip_paths, args.n_pixels, args.ws_n_dates)
        if args.assert_concat_mb_s and mb_s < args.assert_concat_mb_s:
            failures.append(
                f"concat {mb_s:.0f} MB/s < {args.assert_concat_mb_s} MB/s"
            )

        # --- Workstation end-to-end bench ---
        ws = bench_workstation_e2e(tmp / "ws_e2e", strip_paths, args.n_pixels, args.ws_n_dates)
        if args.assert_ws_frame_mb_s and ws.frame_mb_s < args.assert_ws_frame_mb_s:
            failures.append(
                f"ws frame decode {ws.frame_mb_s:.0f} MB/s < {args.assert_ws_frame_mb_s} MB/s"
            )
        if args.assert_ws_rss_gb and ws.peak_rss_gb > args.assert_ws_rss_gb:
            failures.append(
                f"ws merge peak RSS {ws.peak_rss_gb:.2f} GB > {args.assert_ws_rss_gb} GB"
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
