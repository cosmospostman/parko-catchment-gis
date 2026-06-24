"""TAM scoring pipeline benchmark harness.

Synthesises a realistic pixel parquet at configurable scale, runs each stage of
score_pixels_chunked (pre-passes + inference), and records RSS (GB) and wall
time (s) at each probe point in a single merged table.

Run:
    python scripts/bench_score.py
    python scripts/bench_score.py --n-pixels 20000 --n-obs-per-pixel 30 --n-years 1
    python scripts/bench_score.py --pixel-zscore --band-summaries
    python scripts/bench_score.py --d-model 256 --n-layers 3 --d-ff 1024   # v10 size
    python scripts/bench_score.py --assert-wall-s 60 --assert-pixels-per-sec 5000
    python scripts/bench_score.py --n-pixels 50000 --n-years 2 --d-model 256 --n-layers 3 --d-ff 1024 --pixel-zscore --band-summaries

    # Gate fast-path: compares no-gate vs gate=0.5 throughput
    python scripts/bench_score.py --bench-gate --mixed --pixel-zscore

    # Pre-pass RSS: probes each pre-pass separately and measures post-free RSS delta
    python scripts/bench_score.py --bench-prepass --mixed --pixel-zscore

    # Streaming tile-year path: runs score_tile_year (production path) and tracks
    # RSS per writer flush to verify memory stays flat (no full accumulation)
    python scripts/bench_score.py --bench-tile-year --mixed --pixel-zscore

    # Real-data mode: benchmark ONE chunkstore chunk (1024x1024 px) on the exact
    # production path (load_tam -> score_tile_year), capturing the prep_wait /
    # gpu_starvation / transfer breakdown into a machine-readable run.json.
    python scripts/bench_score.py --real \
        --checkpoint outputs/models/tam-v10 --location mitchell \
        --pixel-dir /mnt/gis-archive/chunkstore --years 2025 --tile-id 54LWH \
        --label baseline --out-json bench_runs.jsonl
    # ...change n_prep_workers and compare:
    python scripts/bench_score.py --real ... --n-prep-workers 10 \
        --label workers10 --out-json bench_runs.jsonl
    python scripts/bench_score.py --compare bench_runs.jsonl   # side-by-side delta table

Exit codes: 0 = ok, 1 = assertion failure (--assert-*), 2 = timeout.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Probe system — unified RSS + wall-time table
# ---------------------------------------------------------------------------

def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


class Probe(NamedTuple):
    tag: str
    rss_gb: float
    delta_rss_gb: float
    frame_gb: float | None
    elapsed_s: float
    rows: int | None


_probes: list[Probe] = []
_t0 = time.perf_counter()


def probe(tag: str, df: pl.DataFrame | None = None, rows: int | None = None, extra: str = "") -> None:
    prev_rss = _probes[-1].rss_gb if _probes else rss_gb()
    r = rss_gb()
    frame_gb = df.estimated_size() / 1e9 if df is not None else None
    if rows is None and df is not None:
        rows = len(df)
    _probes.append(Probe(
        tag=tag,
        rss_gb=r,
        delta_rss_gb=r - prev_rss,
        frame_gb=frame_gb,
        elapsed_s=time.perf_counter() - _t0,
        rows=rows,
    ))
    delta_s = f"{'':>1}{'Δ':>1}{_probes[-1].delta_rss_gb:+.2f}"
    frame_s = f"  frame={frame_gb:.2f}GB" if frame_gb is not None else ""
    rows_s  = f"  rows={rows:,}" if rows is not None else ""
    extra_s = f"  {extra}" if extra else ""
    print(f"  [{_probes[-1].elapsed_s:7.2f}s]  {tag}  RSS={r:.2f}GB{delta_s}GB{frame_s}{rows_s}{extra_s}", flush=True)


# ---------------------------------------------------------------------------
# Breakdown capture — parse score_pixels_chunked's logger.info breakdown lines
# ---------------------------------------------------------------------------

# Matches the per-thread budget lines emitted at the end of score_pixels_chunked
# (tam/core/score.py). Kept loose (float capture only) so format tweaks like
# added percentages don't silently break parsing — we only read the seconds.
# The GPU-thread line is the headline: score + gpu_starvation = gpu_wall, and
# duty = score/gpu_wall is the "is the GPU the bottleneck and is it fed?" number.
_RE_GPU = re.compile(
    r"GPU thread budget — wall ([\d.]+)s: score ([\d.]+)s .*?"
    r"gpu_starvation ([\d.]+)s .*?duty=([\d.]+)% +(\d+) GPU calls .*?total run ([\d.]+)s"
)
_RE_XFER = re.compile(
    r"Transfer thread budget — wall ([\d.]+)s: bgt ([\d.]+)s .*?h2d ([\d.]+)s .*?"
    r"push_wait ([\d.]+)s .*?get_wait ([\d.]+)s"
)
_RE_MAIN = re.compile(
    r"Main thread budget .*? wall ([\d.]+)s: prep_wait ([\d.]+)s .*?"
    r"merge ([\d.]+)s .*?xfer_backpressure ([\d.]+)s .*?other ([\d.]+)s"
)
_RE_BATCH = re.compile(
    r"GPU forward per-batch \(ms\): mean=([\d.]+)\s+median=([\d.]+)\s+"
    r"p95=([\d.]+)\s+max=([\d.]+)\s+min=([\d.]+)\s+n=(\d+)"
)


class BreakdownCapture(logging.Handler):
    """Attach to the 'tam.core.score' logger to extract pipeline breakdown stats.

    score_pixels_chunked only *logs* its per-thread timing budgets; it doesn't
    return them. Rather than refactor the production signature, we parse the
    summary lines back into a dict so the bench can fold them into run.json. If a
    tile emits multiple budgets (multi-year), the last one wins — real-data mode
    scores a single chunk/year so there is exactly one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stats: dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if (m := _RE_GPU.search(msg)):
            g = [float(x) for x in m.groups()]
            self.stats.update(
                gpu_wall_s=g[0], score_s=g[1], gpu_starvation_s=g[2],
                gpu_duty_pct=g[3], n_gpu_calls=g[4], total_s=g[5],
            )
        elif (m := _RE_MAIN.search(msg)):
            g = [float(x) for x in m.groups()]
            self.stats.update(
                main_wall_s=g[0], prep_wait_s=g[1], merge_s=g[2],
                xfer_backpressure_s=g[3], other_s=g[4],
            )
        elif (m := _RE_XFER.search(msg)):
            g = [float(x) for x in m.groups()]
            self.stats.update(
                xfer_wall_s=g[0], xfer_bgt_s=g[1], xfer_h2d_s=g[2],
                xfer_push_s=g[3], xfer_get_s=g[4],
            )
        elif (m := _RE_BATCH.search(msg)):
            g = [float(x) for x in m.groups()]
            self.stats.update(
                batch_ms_mean=g[0], batch_ms_median=g[1], batch_ms_p95=g[2],
                batch_ms_max=g[3], batch_ms_min=g[4], n_batches=g[5],
            )


def print_report(
    assert_wall_s: float | None,
    assert_pixels_per_sec: int | None,
    n_pixels: int,
    n_pixel_years: int,
) -> None:
    W = 90
    print("\n" + "=" * W)
    print(f"  {'Stage':<40}  {'RSS GB':>7}  {'Δ RSS':>7}  {'Frame GB':>9}  {'Elapsed s':>9}  {'Rows':>12}")
    print("-" * W)
    for p in _probes:
        frame_s = f"{p.frame_gb:9.2f}" if p.frame_gb is not None else f"{'':>9}"
        rows_s  = f"{p.rows:12,}" if p.rows is not None else f"{'':12}"
        delta_s = f"{p.delta_rss_gb:+7.2f}"
        print(f"  {p.tag:<40}  {p.rss_gb:7.2f}  {delta_s}  {frame_s}  {p.elapsed_s:9.1f}  {rows_s}")
    print("=" * W)

    peak_rss   = max(p.rss_gb for p in _probes)
    total_wall = _probes[-1].elapsed_s
    print(f"\nPeak RSS: {peak_rss:.2f} GB   Total wall: {total_wall:.1f} s")

    # Per-stage deltas for score_chunked
    score_idx = next((i for i, p in enumerate(_probes) if p.tag == "score_chunked"), None)
    if score_idx is not None and score_idx > 0:
        score_delta = _probes[score_idx].elapsed_s - _probes[score_idx - 1].elapsed_s
        pps = n_pixels / score_delta if score_delta > 0 else float("inf")
        pyps = n_pixel_years / score_delta if score_delta > 0 else float("inf")
        print(f"\nThroughput (score_chunked stage only):")
        print(f"  elapsed:          {score_delta:.2f} s")
        print(f"  pixels/sec:       {pps:,.0f}")
        print(f"  pixel-years/sec:  {pyps:,.0f}")
    else:
        score_delta = None
        pps = None

    ok = True
    if assert_wall_s is not None and score_delta is not None:
        if score_delta > assert_wall_s:
            print(f"FAIL: score_chunked {score_delta:.2f} s > limit {assert_wall_s:.2f} s")
            ok = False
        else:
            print(f"PASS: score_chunked {score_delta:.2f} s <= limit {assert_wall_s:.2f} s")
    if assert_pixels_per_sec is not None and pps is not None:
        if pps < assert_pixels_per_sec:
            print(f"FAIL: throughput {pps:,.0f} px/s < limit {assert_pixels_per_sec:,} px/s")
            ok = False
        else:
            print(f"PASS: throughput {pps:,.0f} px/s >= limit {assert_pixels_per_sec:,} px/s")

    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic parquet builder
# ---------------------------------------------------------------------------

def make_synthetic_parquet(
    tmp_dir: Path,
    n_pixels: int,
    n_obs_per_pixel: int,
    n_years: int,
    seed: int,
    mixed: bool = False,
    n_s1_obs_per_pixel: int = 12,
) -> tuple[Path, int, int]:
    """Build a pixel-sorted parquet and write to tmp_dir/bench_pixels.parquet.

    When mixed=True, interleaves S2 rows (with spectral bands + scl_purity) and
    S1 rows (with s1_vh, s1_vv in dB, source="S1") in the same file, matching
    the real mixed-mode parquet schema produced by pixel_collector.

    Returns (path, actual_n_pixels, n_pixel_years).
    """
    rng = np.random.default_rng(seed)
    n_regions = 20
    pids_per_region = max(1, n_pixels // n_regions)

    point_ids: list[str] = []
    for reg in range(n_regions):
        cls = "presence" if reg < n_regions // 2 else "absence"
        for i in range(pids_per_region):
            point_ids.append(f"region_{reg}_{cls}_{i}_0_0")

    n_px = len(point_ids)
    s2_band_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    base_year = 2023 - n_years
    epoch_offset = datetime.date(1970, 1, 1).toordinal()
    jan1_unix = np.array([
        datetime.date(base_year + y, 1, 1).toordinal() - epoch_offset
        for y in range(n_years)
    ], dtype=np.int32)

    def _make_dates(n_obs: int, n_px: int) -> np.ndarray:
        doys = rng.integers(1, 365, size=(n_px, n_years, n_obs)).astype(np.int32)
        doys.sort(axis=2)
        jan1_grid = np.broadcast_to(jan1_unix[None, :, None], (n_px, n_years, n_obs)).copy()
        return (jan1_grid + doys - 1).flatten().astype(np.int32)

    # --- S2 rows ---
    n_s2_total = n_px * n_obs_per_pixel * n_years
    s2_pids    = np.repeat(point_ids, n_obs_per_pixel * n_years)
    s2_dates   = _make_dates(n_obs_per_pixel, n_px)
    s2_data: dict[str, object] = {
        "point_id":   pl.Series(s2_pids),
        "date":       pl.Series(s2_dates, dtype=pl.Date),
        "source":     pl.Series(["S2"] * n_s2_total),
        "scl_purity": pl.Series(rng.uniform(0.6, 1.0, size=n_s2_total).astype(np.float32)),
        **{b: pl.Series(rng.uniform(0.01, 0.5, size=n_s2_total).astype(np.float32))
           for b in s2_band_names},
    }
    if mixed:
        # S2 rows need placeholder vh/vv so schema matches S1 rows after concat
        s2_data["vh"] = pl.Series(np.zeros(n_s2_total, dtype=np.float32))
        s2_data["vv"] = pl.Series(np.zeros(n_s2_total, dtype=np.float32))

    df_s2 = pl.DataFrame(s2_data)

    if mixed:
        # --- S1 rows: vh/vv as raw linear power (same schema as real pixel parquets) ---
        # _compute_pixel_s1_stats_mixed reads "vh"/"vv" columns and converts to dB internally.
        n_s1_total = n_px * n_s1_obs_per_pixel * n_years
        s1_pids    = np.repeat(point_ids, n_s1_obs_per_pixel * n_years)
        s1_dates   = _make_dates(n_s1_obs_per_pixel, n_px)
        s1_data: dict[str, object] = {
            "point_id":   pl.Series(s1_pids),
            "date":       pl.Series(s1_dates, dtype=pl.Date),
            "source":     pl.Series(["S1"] * n_s1_total),
            "scl_purity": pl.Series(np.ones(n_s1_total, dtype=np.float32)),
            **{b: pl.Series(np.zeros(n_s1_total, dtype=np.float32)) for b in s2_band_names},
            "vh": pl.Series(rng.uniform(0.001, 0.1, size=n_s1_total).astype(np.float32)),
            "vv": pl.Series(rng.uniform(0.002, 0.15, size=n_s1_total).astype(np.float32)),
        }
        df = pl.concat([df_s2, pl.DataFrame(s1_data)]).sort(["point_id", "date"])
    else:
        df = df_s2.sort(["point_id", "date"])

    path = tmp_dir / "bench_pixels.parquet"
    df.write_parquet(path, row_group_size=16384, compression="snappy")

    return path, n_px, n_px * n_years


# ---------------------------------------------------------------------------
# Warmup helper
# ---------------------------------------------------------------------------

def run_warmup(device: str) -> None:
    """JIT-compile numba kernels and warm the torch device.

    Only calls _numba_warmup() — no scoring pass, so cost is constant
    regardless of dataset size or model size.
    """
    import torch
    from tam.core._preprocess_numba import warmup as _numba_warmup

    _numba_warmup()
    # Touch the device so CUDA context is initialised before the timed run.
    if device != "cpu":
        _ = torch.zeros(1, device=device)


# ---------------------------------------------------------------------------
# bench_prepass — per-pass RSS and wall-time breakdown
# ---------------------------------------------------------------------------

def bench_prepass(
    model: "TAMClassifier",
    year_parquets: list,
    s2_cols: list[str],
    s1_cols: list[str],
    mixed: bool,
) -> None:
    """Probe each pre-pass (S2 zscore, S1 zscore) for RSS and wall time.

    Runs the pre-passes sequentially (not concurrent) so RSS deltas are clean,
    then checks what RSS is reclaimed after del + gc.collect().
    """
    from tam.core.score import (
        _compute_s2_pixel_zscore_stats,
        _compute_pixel_s1_stats_mixed,
        _ZscoreArrays,
    )

    print("\n[bench_prepass]")
    probe("prepass:start")

    # S2 zscore
    t0 = time.perf_counter()
    raw_s2 = _compute_s2_pixel_zscore_stats(
        year_parquets=year_parquets,
        feature_cols=s2_cols,
        scl_purity_min=0.5,
    )
    elapsed_s2 = time.perf_counter() - t0
    n_s2_px = len(raw_s2[0])
    probe("prepass:s2_zscore_done", rows=n_s2_px,
          extra=f"{elapsed_s2:.2f}s  {n_s2_px:,} px × {len(s2_cols)} features")
    print(f"  S2 zscore: {elapsed_s2:.2f}s  {n_s2_px:,} pixels")

    # Convert to _ZscoreArrays to match production (allocates two numpy matrices)
    t0 = time.perf_counter()
    s2_za = _ZscoreArrays(*raw_s2, n_feat=len(s2_cols))
    elapsed_convert = time.perf_counter() - t0
    probe("prepass:s2_zscore_arrays", rows=n_s2_px,
          extra=f"convert={elapsed_convert:.3f}s")
    del raw_s2   # dicts are freed; arrays survive in s2_za
    gc.collect()
    probe("prepass:s2_dict_freed")
    print(f"  S2 ZscoreArrays built in {elapsed_convert:.3f}s  (dicts freed)")

    # S1 zscore (mixed mode only)
    s1_za = None
    if mixed and s1_cols:
        t0 = time.perf_counter()
        raw_s1 = _compute_pixel_s1_stats_mixed(
            year_parquets=year_parquets,
            s1_feature_cols=s1_cols,
        )
        elapsed_s1 = time.perf_counter() - t0
        n_s1_px = len(raw_s1[0])
        probe("prepass:s1_zscore_done", rows=n_s1_px,
              extra=f"{elapsed_s1:.2f}s  {n_s1_px:,} px × {len(s1_cols)} features")
        print(f"  S1 zscore: {elapsed_s1:.2f}s  {n_s1_px:,} pixels")

        s1_za = _ZscoreArrays(*raw_s1, n_feat=len(s1_cols))
        del raw_s1
        gc.collect()
        probe("prepass:s1_dict_freed")

    # Peak
    s2_bytes = s2_za._means.nbytes + s2_za._stds.nbytes
    print(f"  ZscoreArrays heap: S2={s2_bytes/1e6:.1f} MB", end="")
    if s1_za is not None:
        s1_bytes = s1_za._means.nbytes + s1_za._stds.nbytes
        print(f"  S1={s1_bytes/1e6:.1f} MB", end="")
    print()

    probe("prepass:done")
    pre_rss   = _probes[0].rss_gb  # baseline before any probes
    peak_rss  = max(p.rss_gb for p in _probes)
    print(f"  Peak RSS during pre-passes: {peak_rss:.2f} GB  (baseline {pre_rss:.2f} GB  Δ{peak_rss - pre_rss:+.2f} GB)")


# ---------------------------------------------------------------------------
# bench_gate — compare no-gate vs gate=0.5 throughput on same data
# ---------------------------------------------------------------------------

def bench_gate(
    parquet: Path,
    model: "TAMClassifier",
    band_mean: "np.ndarray",
    band_std:  "np.ndarray",
    s2_cols: list[str],
    s1_cols: list[str],
    pixel_zscore_stats,
    s1_zscore_stats,
    batch_size: int,
    n_prep_workers: int,
    device: str,
    n_pixels: int,
    end_year: int,
    mixed: bool,
    gate_threshold: float = 0.5,
    T_gate: int = 8,
) -> None:
    """Compare no-gate vs gate fast-path throughput on the same synthetic parquet.

    Reports for both runs:
      - Wall time for score_pixels_chunked
      - Pixels/sec
      - (gate run) fraction cut by gate, effective full-T px/s
    """
    from tam.core.score import score_pixels_chunked

    print(f"\n[bench_gate]  gate_threshold={gate_threshold}  T_gate={T_gate}  n_pixels={n_pixels:,}")

    # No-gate baseline
    probe("gate:no_gate_start")
    t0 = time.perf_counter()
    scores_ng = score_pixels_chunked(
        parquet=parquet, model=model,
        band_mean=band_mean, band_std=band_std,
        scl_purity_min=0.5, min_obs_per_year=8,
        batch_size=batch_size, n_prep_workers=n_prep_workers,
        device=device, end_year=end_year, decay=0.7,
        n_total_pixels=n_pixels,
        mixed=mixed, pixel_zscore=(pixel_zscore_stats is not None),
        pixel_zscore_stats=pixel_zscore_stats,
        s1_zscore_stats=s1_zscore_stats,
        s2_feature_cols=s2_cols,
        s1_feature_cols=s1_cols if mixed else None,
        gate_threshold=0.0,
    )
    elapsed_ng = time.perf_counter() - t0
    pps_ng = n_pixels / elapsed_ng
    probe("gate:no_gate_done", rows=len(scores_ng), extra=f"{elapsed_ng:.1f}s  {pps_ng:,.0f} px/s")
    print(f"  no-gate:  {elapsed_ng:.1f}s  {pps_ng:,.0f} px/s  ({len(scores_ng):,} scored px)")
    del scores_ng
    gc.collect()

    # Gate run — capture gate stats via log interception
    import logging as _logging
    gate_stats_msg: list[str] = []
    class _GateCap(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            if "Gate (" in record.getMessage():
                gate_stats_msg.append(record.getMessage())
    _cap = _GateCap()
    logging.getLogger("tam.core.score").addHandler(_cap)

    probe("gate:gate_start")
    t0 = time.perf_counter()
    scores_g = score_pixels_chunked(
        parquet=parquet, model=model,
        band_mean=band_mean, band_std=band_std,
        scl_purity_min=0.5, min_obs_per_year=8,
        batch_size=batch_size, n_prep_workers=n_prep_workers,
        device=device, end_year=end_year, decay=0.7,
        n_total_pixels=n_pixels,
        mixed=mixed, pixel_zscore=(pixel_zscore_stats is not None),
        pixel_zscore_stats=pixel_zscore_stats,
        s1_zscore_stats=s1_zscore_stats,
        s2_feature_cols=s2_cols,
        s1_feature_cols=s1_cols if mixed else None,
        gate_threshold=gate_threshold,
        T_gate=T_gate,
    )
    elapsed_g = time.perf_counter() - t0
    logging.getLogger("tam.core.score").removeHandler(_cap)

    pps_g = n_pixels / elapsed_g
    speedup = elapsed_ng / max(elapsed_g, 1e-6)
    probe("gate:gate_done", rows=len(scores_g), extra=f"{elapsed_g:.1f}s  {pps_g:,.0f} px/s  {speedup:.2f}× speedup")
    print(f"  gate:     {elapsed_g:.1f}s  {pps_g:,.0f} px/s  {speedup:.2f}× speedup vs no-gate")
    for msg in gate_stats_msg:
        print(f"  {msg}")
    del scores_g
    gc.collect()


# ---------------------------------------------------------------------------
# bench_tile_year_streaming — score_tile_year RSS profile
# ---------------------------------------------------------------------------

def bench_tile_year_streaming(
    parquet: Path,
    model: "TAMClassifier",
    band_mean: "np.ndarray",
    band_std:  "np.ndarray",
    s2_cols: list[str],
    s1_cols: list[str],
    batch_size: int,
    n_prep_workers: int,
    device: str,
    n_pixels: int,
    end_year: int,
    mixed: bool,
    tmp_dir: Path,
) -> None:
    """Benchmark score_pixels_chunked in streaming mode (out_writer path) and profile RSS.

    The production scoring path (score_tile_year) uses out_writer to stream
    inference results directly to a ParquetWriter in a background thread, so
    memory is bounded by the flush interval rather than accumulating all results.

    This bench exercises that exact path and verifies that RSS stays flat
    throughout inference by monkey-patching the writer flush to sample RSS
    after each row-group write — mirroring the technique in bench_fetch.py.

    Key numbers reported:
      - Total wall time including pre-passes
      - Pixels/sec for the scoring stage
      - RSS at start / peak / end (Δ should be small — only pre-pass stats + model)
      - Number of writer flushes and RSS range during flushing
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tam.core.score import (
        score_pixels_chunked,
        _compute_s2_pixel_zscore_stats,
        _compute_pixel_s1_stats_mixed,
        _ZscoreArrays,
        _get_staging_schema,
        _STAGING_WRITE_OPTS,
    )

    print(f"\n[bench_tile_year_streaming]  n_pixels={n_pixels:,}  end_year={end_year}")
    staging_dir = tmp_dir / "staging_bench"
    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / f"bench_{end_year}.parquet"
    tmp_path = out_path.with_suffix(".tmp.parquet")

    year_parquets = [(end_year, parquet)]

    # --- Pre-passes (mirrors score_tile_year) ---
    probe("tile_year:start")
    from concurrent.futures import ThreadPoolExecutor as _TPE
    t_pre = time.perf_counter()
    _n_pre = 2 if mixed else 1
    with _TPE(max_workers=_n_pre) as _ex:
        _f_s2 = _ex.submit(
            _compute_s2_pixel_zscore_stats,
            year_parquets=year_parquets, feature_cols=s2_cols, scl_purity_min=0.5,
        )
        _f_s1 = _ex.submit(
            _compute_pixel_s1_stats_mixed,
            year_parquets=year_parquets, s1_feature_cols=s1_cols,
        ) if mixed else None
    _raw_s2 = _f_s2.result()
    _pz = _ZscoreArrays(*_raw_s2, n_feat=len(s2_cols))
    _s1z = _ZscoreArrays(*_f_s1.result(), n_feat=len(s1_cols)) if _f_s1 is not None else None
    elapsed_pre = time.perf_counter() - t_pre
    probe("tile_year:prepass_done", extra=f"{elapsed_pre:.2f}s  {len(_raw_s2[0]):,} px")
    print(f"  pre-passes: {elapsed_pre:.2f}s  {len(_raw_s2[0]):,} pixels")

    # Monkey-patch ParquetWriter.write_table to sample RSS per flush
    rss_samples: list[float] = []
    flush_count = [0]
    _orig_write_table = pq.ParquetWriter.write_table

    def _patched_write_table(self, table, *args, **kwargs):
        result = _orig_write_table(self, table, *args, **kwargs)
        rss_samples.append(rss_gb())
        flush_count[0] += 1
        return result

    probe("tile_year:score_start")
    t0 = time.perf_counter()
    try:
        pq.ParquetWriter.write_table = _patched_write_table
        with pq.ParquetWriter(tmp_path, schema=_get_staging_schema(), **_STAGING_WRITE_OPTS) as staging_writer:
            score_pixels_chunked(
                parquet=parquet, model=model,
                band_mean=band_mean, band_std=band_std,
                scl_purity_min=0.5, min_obs_per_year=8,
                batch_size=batch_size, n_prep_workers=n_prep_workers,
                device=device, end_year=end_year, decay=0.0,
                n_total_pixels=n_pixels,
                mixed=mixed, pixel_zscore=True,
                pixel_zscore_stats=_pz,
                s1_zscore_stats=_s1z,
                s2_feature_cols=s2_cols,
                s1_feature_cols=s1_cols if mixed else None,
                out_writer=staging_writer,
                write_flush_rows=200_000,
            )
        tmp_path.rename(out_path)
    finally:
        pq.ParquetWriter.write_table = _orig_write_table
    elapsed = time.perf_counter() - t0

    # Read output to check row count
    out_rows = pq.ParquetFile(out_path).metadata.num_rows if out_path.exists() else 0
    out_bytes = out_path.stat().st_size if out_path.exists() else 0
    pps = n_pixels / max(elapsed, 1e-6)

    probe("tile_year:done", rows=out_rows,
          extra=f"{elapsed:.1f}s  {pps:,.0f} px/s  out={out_bytes/1e6:.0f}MB")
    print(f"  score_tile_year: {elapsed:.1f}s  {pps:,.0f} px/s  "
          f"({out_rows:,} rows → {out_bytes/1e6:.0f} MB)")

    if rss_samples:
        rss_baseline = _probes[-3].rss_gb if len(_probes) >= 3 else rss_gb()
        peak_rss  = max(rss_samples)
        final_rss = rss_samples[-1]
        rss_range = max(rss_samples) - min(rss_samples)
        print(f"  RSS: baseline={rss_baseline:.2f}GB  peak={peak_rss:.2f}GB  final={final_rss:.2f}GB")
        print(f"  Writer flushes: {flush_count[0]}  RSS range during flush: {rss_range:.3f} GB")
        print(f"  {'PASS' if rss_range < 0.5 else 'WARN'}: "
              f"streaming RSS variation {'<' if rss_range < 0.5 else '>='} 0.5 GB threshold")


# ---------------------------------------------------------------------------
# Main benchmark run
# ---------------------------------------------------------------------------

def run_bench(args: argparse.Namespace, tmp_dir: Path) -> None:
    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    probe("baseline")

    # 1. Synthetic data
    print(f"\nbench_score: n_pixels={args.n_pixels:,}  n_obs/px/yr={args.n_obs_per_pixel}"
          f"  n_years={args.n_years}  batch_size={args.batch_size}"
          f"  n_prep_workers={args.n_prep_workers}  device={args.device}"
          f"  d_model={args.d_model}  n_layers={args.n_layers}  d_ff={args.d_ff}"
          f"  mixed={args.mixed}\n")

    parquet, n_pixels, n_pixel_years = make_synthetic_parquet(
        tmp_dir,
        n_pixels=args.n_pixels,
        n_obs_per_pixel=args.n_obs_per_pixel,
        n_years=args.n_years,
        seed=args.seed,
        mixed=args.mixed,
        n_s1_obs_per_pixel=args.n_s1_obs_per_pixel,
    )
    n_synth_rows = n_pixels * args.n_obs_per_pixel * args.n_years
    if args.mixed:
        n_synth_rows += n_pixels * args.n_s1_obs_per_pixel * args.n_years
    probe("synth_gen", rows=n_synth_rows)
    probe("parquet_write")

    # 2. Deferred tam.* imports (after parquet write so RSS delta is clean)
    from tam.core.model import TAMClassifier
    from tam.core.dataset import ALL_FEATURE_COLS, V10_S1_FEATURE_COLS
    from tam.core.score import (
        score_pixels_chunked,
        _compute_s2_pixel_zscore_stats,
        _compute_pixel_s1_stats_mixed,
    )

    # 3. Model init — mixed mode uses S2 + S1 bands
    s2_cols = list(ALL_FEATURE_COLS)
    s1_cols = list(V10_S1_FEATURE_COLS) if args.mixed else []
    n_bands = len(s2_cols) + len(s1_cols)
    band_mean = np.zeros(n_bands, dtype=np.float32)
    band_std  = np.ones(n_bands,  dtype=np.float32)
    model = TAMClassifier(
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        n_bands=n_bands,
    )
    model._max_seq_len = args.max_seq_len  # mirrors what load_tam sets from tam_config.json
    model.to(args.device)
    model.eval()
    probe("model_init", rows=sum(p.numel() for p in model.parameters()))

    # 4. Warmup (JIT + optional CUDA context)
    if not args.skip_warmup:
        run_warmup(args.device)
        gc.collect()
        probe("warmup")

    end_year = 2022 + args.n_years - 1  # last synthetic year
    year_parquets = [(end_year, parquet)]

    # 5. Pre-passes (run in parallel when --pixel-zscore is set)
    pixel_zscore_stats = None
    s1_zscore_stats    = None
    if args.pixel_zscore:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _workers = 2 + (1 if args.mixed else 0)  # S2 zscore + S1 zscore (mixed) + no band summaries here
        with _TPE(max_workers=_workers) as ex:
            f_s2 = ex.submit(
                _compute_s2_pixel_zscore_stats,
                year_parquets=year_parquets,
                feature_cols=s2_cols,
                scl_purity_min=0.5,
            )
            f_s1 = ex.submit(
                _compute_pixel_s1_stats_mixed,
                year_parquets=year_parquets,
                s1_feature_cols=s1_cols,
            ) if args.mixed else None
        pixel_zscore_stats = f_s2.result()
        probe("prepass_s2_zscore", rows=len(pixel_zscore_stats[0]))
        if f_s1 is not None:
            s1_zscore_stats = f_s1.result()
            probe("prepass_s1_zscore", rows=len(s1_zscore_stats[0]))

    # 7. Score (main timed stage)
    n_feat = len(s2_cols)
    scores_df = score_pixels_chunked(
        parquet=parquet,
        model=model,
        band_mean=band_mean,
        band_std=band_std,
        scl_purity_min=0.5,
        min_obs_per_year=8,
        batch_size=args.batch_size,
        n_prep_workers=args.n_prep_workers,
        device=args.device,
        end_year=end_year,
        decay=0.7,
        n_total_pixels=n_pixels,
        pixel_zscore=args.pixel_zscore,
        pixel_zscore_stats=pixel_zscore_stats,
        s1_zscore_stats=s1_zscore_stats,
        annual_feat_mean=np.zeros(n_feat * 3, dtype=np.float32) if args.band_summaries else None,
        annual_feat_std=np.ones(n_feat * 3,  dtype=np.float32) if args.band_summaries else None,
        mixed=args.mixed,
        s2_feature_cols=s2_cols,
        s1_feature_cols=s1_cols if args.mixed else None,
    )
    probe("score_chunked", df=scores_df, rows=len(scores_df))
    probe("done")

    print_report(
        assert_wall_s=args.assert_wall_s,
        assert_pixels_per_sec=args.assert_pixels_per_sec,
        n_pixels=n_pixels,
        n_pixel_years=n_pixel_years,
    )

    # 8. Optional focused benches (run after main bench so model+data are warm)
    if args.bench_prepass:
        bench_prepass(
            model=model,
            year_parquets=year_parquets,
            s2_cols=s2_cols,
            s1_cols=s1_cols,
            mixed=args.mixed,
        )

    if args.bench_gate:
        bench_gate(
            parquet=parquet,
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            s2_cols=s2_cols,
            s1_cols=s1_cols,
            pixel_zscore_stats=pixel_zscore_stats,
            s1_zscore_stats=s1_zscore_stats,
            batch_size=args.batch_size,
            n_prep_workers=args.n_prep_workers,
            device=args.device,
            n_pixels=n_pixels,
            end_year=end_year,
            mixed=args.mixed,
            gate_threshold=args.gate_threshold,
            T_gate=args.t_gate,
        )

    if args.bench_tile_year:
        bench_tile_year_streaming(
            parquet=parquet,
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            s2_cols=s2_cols,
            s1_cols=s1_cols,
            batch_size=args.batch_size,
            n_prep_workers=args.n_prep_workers,
            device=args.device,
            n_pixels=n_pixels,
            end_year=end_year,
            mixed=args.mixed,
            tmp_dir=tmp_dir,
        )


# ---------------------------------------------------------------------------
# Real-data mode — benchmark ONE chunkstore chunk on the production path
# ---------------------------------------------------------------------------

def _resolve_chunks(args: argparse.Namespace) -> tuple[str, int, list[Path]]:
    """Resolve (tile_id, year, [chunk_paths]) for the real-data bench.

    Either --chunk-file gives one path directly, or --location/--pixel-dir/
    --tile-id/--years select the first --n-chunks chunks (row-major) for that
    tile-year. Multiple chunks are wrapped in a single ChunkPixelSource so the
    bench exercises the production cross-chunk streaming path.
    """
    from tam.pipeline import _CHUNK_PAT, _chunk_key_stem, _tile_id_from_stem

    n_chunks = max(1, getattr(args, "n_chunks", 1))

    if args.chunk_file:
        chunk = Path(args.chunk_file)
        if not chunk.is_file():
            sys.exit(f"--chunk-file not found: {chunk}")
        return _tile_id_from_stem(chunk.stem), (args.years[0] if args.years else 0), [chunk]

    from utils.location import get as get_location

    if not args.location:
        sys.exit("real mode needs --chunk-file or (--location and --tile-id)")
    loc = get_location(args.location)
    pixel_dir = Path(args.pixel_dir) if args.pixel_dir else None
    tile_paths_by_year = loc.parquet_tile_paths(base_dir=pixel_dir)
    if args.years:
        tile_paths_by_year = {y: ps for y, ps in tile_paths_by_year.items() if y in args.years}
    if not tile_paths_by_year:
        sys.exit(f"No parquets found for {args.location} years={args.years}")
    year = max(tile_paths_by_year)
    tid_filter = set(args.tile_id or [])
    chunks = [
        p for p in tile_paths_by_year[year]
        if _CHUNK_PAT.search(p.stem) and (not tid_filter or _tile_id_from_stem(p.stem) in tid_filter)
    ]
    if not chunks:
        sys.exit(f"No chunk files for tile(s) {sorted(tid_filter) or 'ANY'} in year {year}")
    chunks.sort(key=lambda p: _chunk_key_stem(p.stem))
    selected = chunks[:n_chunks]
    if len(selected) < n_chunks:
        print(f"WARN: requested {n_chunks} chunks but only {len(selected)} available for tile")
    return _tile_id_from_stem(selected[0].stem), year, selected


def run_real(args: argparse.Namespace, tmp_dir: Path) -> None:
    """Score one real chunkstore chunk via score_tile_year, capture breakdown → run.json."""
    import torch
    from tam.core.pixel_source import ChunkPixelSource
    from tam.core.score import score_tile_year
    from tam.core.train import load_tam

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    tile_id, year, chunk_paths = _resolve_chunks(args)
    chunk_mb = sum(p.stat().st_size for p in chunk_paths) / 1e6
    n_chunks = len(chunk_paths)
    chunk_desc = (chunk_paths[0].name if n_chunks == 1
                  else f"{n_chunks} chunks {chunk_paths[0].name}..{chunk_paths[-1].name}")
    print(f"\nbench_score --real: {chunk_desc} ({chunk_mb:.0f} MB total)  "
          f"tile={tile_id}  year={year}  device={args.device}\n"
          f"  checkpoint={args.checkpoint}  batch_size={args.batch_size}  "
          f"n_prep_workers={args.n_prep_workers}  label={args.label or '-'}\n", flush=True)
    probe("baseline")

    # --- Load checkpoint exactly as tam.pipeline._cmd_score does ---
    checkpoint_dir = Path(args.checkpoint)
    model, band_mean, band_std, annual_feat_mean, annual_feat_std = load_tam(
        checkpoint_dir, device=args.device
    )
    with open(checkpoint_dir / "tam_config.json") as fh:
        cfg = json.load(fh)
    use_s1 = cfg.get("use_s1", False)
    s1_only = model.n_bands == 4
    mixed = (not s1_only) and bool(use_s1)
    s2_cols = cfg.get("feature_cols", None)
    s1_cols = cfg.get("s1_feature_cols", None) or (["s1_vh", "s1_vv"] if mixed else None)
    s1_despeckle_window = cfg.get("s1_despeckle_window", 0)
    probe("load_tam", rows=sum(p.numel() for p in model.parameters()),
          extra=f"mixed={mixed} s1_only={s1_only}")

    # Warm numba + CUDA so JIT cost isn't charged to the timed run.
    if not args.skip_warmup:
        run_warmup(args.device)
        gc.collect()
        probe("warmup")

    # --- Capture the breakdown log lines emitted inside score_pixels_chunked ---
    # The breakdown is logged at INFO; the bench never calls basicConfig, so the
    # logger's effective level defaults to WARNING and would drop INFO records
    # before they reach our handler. Force INFO on this logger so capture works
    # without spamming the rest of the bench output with score's progress logs.
    score_logger = logging.getLogger("tam.core.score")
    score_logger.setLevel(logging.INFO)
    cap = BreakdownCapture()
    score_logger.addHandler(cap)

    source = ChunkPixelSource(chunk_paths)   # N chunks as one stream, no geometry filter
    staging_dir = tmp_dir / "staging_real"

    probe("score_start")
    t0 = time.perf_counter()
    try:
        out_path = score_tile_year(
            source=source,
            tile_id=tile_id,
            year=year,
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            staging_dir=staging_dir,
            scl_purity_min=args.scl_purity,
            batch_size=args.batch_size,
            n_prep_workers=args.n_prep_workers,
            device=args.device,
            s1_only=s1_only,
            mixed=mixed,
            s1_despeckle_window=s1_despeckle_window,
            s2_feature_cols=s2_cols if mixed else None,
            s1_feature_cols=s1_cols if mixed else None,
            annual_feat_mean=annual_feat_mean,
            annual_feat_std=annual_feat_std,
        )
    finally:
        score_logger.removeHandler(cap)
    elapsed = time.perf_counter() - t0

    import pyarrow.parquet as pq
    out_rows = pq.ParquetFile(out_path).metadata.num_rows if out_path.exists() else 0
    probe("score_done", rows=out_rows, extra=f"{elapsed:.1f}s")
    probe("done")

    # n_scored pixels = output rows (one row per scored pixel-window).
    pps = out_rows / elapsed if elapsed > 0 else 0.0
    stats = cap.stats
    if not stats:
        print("\nWARN: no breakdown line captured — log format may have changed "
              "(check _RE_GPU in bench_score.py).")

    run = {
        "label": args.label,
        "tile_id": tile_id,
        "year": year,
        "n_chunks": n_chunks,
        "chunk": chunk_desc,
        "chunk_mb": round(chunk_mb, 1),
        "checkpoint": str(checkpoint_dir),
        "device": args.device,
        "batch_size": args.batch_size,
        "n_prep_workers": args.n_prep_workers,
        "scored_rows": out_rows,
        "wall_s": round(elapsed, 2),
        "px_per_s": round(pps, 0),
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        **{k: round(v, 2) for k, v in stats.items()},
    }

    # --- Report ---
    W = 60
    print("\n" + "=" * W)
    print(f"  REAL-DATA BENCH  [{args.label or 'run'}]")
    print("-" * W)
    print(f"  chunks:           {n_chunks}  ({chunk_mb:.0f} MB total)")
    print(f"  scored rows:      {out_rows:,}")
    print(f"  wall:             {elapsed:.1f} s")
    print(f"  throughput:       {pps:,.0f} px/s")
    if stats:
        # GPU thread is the pacing stage: score + gpu_starvation = gpu_wall.
        gpu_wall = stats.get("gpu_wall_s", 0.0) or 1.0
        duty = stats.get("gpu_duty_pct", 100.0 * stats.get("score_s", 0) / gpu_wall)
        print(f"  GPU duty cycle:   {duty:5.1f}%   <- headline (score / gpu_wall)")
        print(f"    score (work):   {stats.get('score_s', 0):7.1f}s")
        print(f"    gpu_starvation: {stats.get('gpu_starvation_s', 0):7.1f}s   <- GPU idle/unfed")
        xw = stats.get("xfer_wall_s", 0.0) or 1.0
        print(f"  transfer (wall {xw:.1f}s):  bgt {stats.get('xfer_bgt_s', 0):.1f}s  "
              f"h2d {stats.get('xfer_h2d_s', 0):.1f}s  push {stats.get('xfer_push_s', 0):.1f}s  "
              f"get_wait {stats.get('xfer_get_s', 0):.1f}s")
        print(f"  main thread (overlaps GPU, not lost time):  "
              f"prep_wait {stats.get('prep_wait_s', 0):.1f}s  "
              f"xfer_backp {stats.get('xfer_backpressure_s', 0):.1f}s")
        if "batch_ms_p95" in stats:
            print(f"  per-batch ms:     mean={stats['batch_ms_mean']:.1f} "
                  f"p95={stats['batch_ms_p95']:.1f} max={stats['batch_ms_max']:.1f} "
                  f"n={int(stats.get('n_batches', 0))}")
    print("=" * W)

    if args.out_json:
        with open(args.out_json, "a") as fh:
            fh.write(json.dumps(run) + "\n")
        print(f"\nAppended run to {args.out_json}")
    else:
        print("\n(no --out-json given; run not persisted for comparison)")


def run_compare(path: Path) -> None:
    """Print a side-by-side delta table from a JSONL of real-data runs."""
    runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not runs:
        sys.exit(f"No runs in {path}")
    rows = [
        ("label",          lambda r: str(r.get("label") or "-")),
        ("chunks",         lambda r: str(r.get("n_chunks", 1))),
        ("workers",        lambda r: str(r.get("n_prep_workers", "-"))),
        ("batch",          lambda r: str(r.get("batch_size", "-"))),
        ("px/s",           lambda r: f"{r.get('px_per_s', 0):,.0f}"),
        ("wall s",         lambda r: f"{r.get('wall_s', 0):.1f}"),
        ("gpu_duty %",     lambda r: f"{r.get('gpu_duty_pct', 0):.1f}"),
        ("score s",        lambda r: f"{r.get('score_s', 0):.1f}"),
        ("gpu_starv s",    lambda r: f"{r.get('gpu_starvation_s', 0):.1f}"),
        ("xfer h2d s",     lambda r: f"{r.get('xfer_h2d_s', 0):.1f}"),
        ("xfer get s",     lambda r: f"{r.get('xfer_get_s', 0):.1f}"),
        ("batch p95 ms",   lambda r: f"{r.get('batch_ms_p95', 0):.1f}"),
    ]
    label_w = max(12, max(len(str(r.get("label") or "-")) for r in runs) + 1)
    head = f"  {'metric':<14}" + "".join(f"{f'run{i}':>{label_w}}" for i in range(len(runs)))
    print("\n" + "=" * len(head))
    print(head)
    print("-" * len(head))
    for name, fn in rows:
        line = f"  {name:<14}" + "".join(f"{fn(r):>{label_w}}" for r in runs)
        print(line)
    print("=" * len(head))
    # px/s delta vs first run
    base = runs[0].get("px_per_s", 0) or 1
    print("\n  px/s vs run0:")
    for i, r in enumerate(runs):
        d = r.get("px_per_s", 0)
        print(f"    run{i} [{r.get('label') or '-'}]: {d:,.0f} px/s  ({100*(d-base)/base:+.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # flush probes immediately in subprocess
    parser = argparse.ArgumentParser(description="TAM scoring pipeline benchmark")
    parser.add_argument("--n-pixels",              type=int,   default=50_000,
                        help="Unique pixels in synthetic data")
    parser.add_argument("--n-obs-per-pixel",        type=int,   default=50,
                        help="S2 observations per pixel per year")
    parser.add_argument("--n-years",               type=int,   default=2,
                        help="Number of synthetic years")
    parser.add_argument("--batch-size",            type=int,   default=4096)
    parser.add_argument("--n-prep-workers",        type=int,   default=5)
    parser.add_argument("--pixel-zscore",          action="store_true", default=False)
    parser.add_argument("--band-summaries",        action="store_true", default=False)
    parser.add_argument("--mixed",                 action="store_true", default=False,
                        help="Synthesise mixed S2+S1 parquet (production mode)")
    parser.add_argument("--n-s1-obs-per-pixel",    type=int,   default=12,
                        help="S1 observations per pixel per year (mixed mode only)")
    parser.add_argument("--skip-warmup",           action="store_true", default=False)
    parser.add_argument("--assert-wall-s",         type=float, default=None,
                        help="Assert score_chunked elapsed < N s")
    parser.add_argument("--assert-pixels-per-sec", type=int,   default=None,
                        help="Assert throughput > N pixels/sec")
    parser.add_argument("--timeout",               type=int,   default=300)
    parser.add_argument("--device",                default=None,
                        help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("--d-model",               type=int,   default=64,
                        help="TAMClassifier d_model (v10=256)")
    parser.add_argument("--n-layers",              type=int,   default=2,
                        help="TAMClassifier n_layers (v10=3)")
    parser.add_argument("--d-ff",                  type=int,   default=128,
                        help="TAMClassifier d_ff (v10=1024)")
    parser.add_argument("--max-seq-len",           type=int,   default=128,
                        help="Inference sequence length cap (v10 trained at 128)")
    parser.add_argument("--seed",                  type=int,   default=42)
    # Focused bench flags (run after main bench; combine with --mixed --pixel-zscore)
    parser.add_argument("--bench-prepass",   action="store_true", default=False,
                        help="Probe each pre-pass (S2/S1 zscore) separately for RSS and wall time")
    parser.add_argument("--bench-gate",      action="store_true", default=False,
                        help="Compare no-gate vs gate fast-path throughput (gate_threshold=0 vs --gate-threshold)")
    parser.add_argument("--bench-tile-year", action="store_true", default=False,
                        help="Run score_tile_year (production streaming path) and track RSS per writer flush")
    parser.add_argument("--gate-threshold",  type=float, default=0.5,
                        help="Gate threshold for --bench-gate (default: 0.5)")
    parser.add_argument("--t-gate",          type=int,   default=8,
                        help="Gate sequence length for --bench-gate (default: 8)")
    # --- Real-data mode: benchmark one chunkstore chunk on the production path ---
    parser.add_argument("--real",            action="store_true", default=False,
                        help="Benchmark ONE real chunkstore chunk via score_tile_year")
    parser.add_argument("--checkpoint",      default=None,
                        help="[real] Checkpoint dir, e.g. outputs/models/tam-v10")
    parser.add_argument("--location",        default=None,
                        help="[real] Location id (e.g. mitchell) to resolve chunks")
    parser.add_argument("--pixel-dir",       default=None,
                        help="[real] Chunkstore base dir (e.g. /mnt/gis-archive/chunkstore)")
    parser.add_argument("--tile-id",         nargs="+", default=None, metavar="TILE_ID",
                        help="[real] MGRS tile(s); first matching chunk is benched")
    parser.add_argument("--chunk-file",      default=None,
                        help="[real] Explicit chunk parquet path (overrides --location/--tile-id)")
    parser.add_argument("--n-chunks",        type=int, default=1,
                        help="[real] Score the first N chunks of the tile as one stream "
                             "(default 1; raise to probe cross-chunk gpu_starvation)")
    parser.add_argument("--years",           type=int, nargs="+", default=None, metavar="YEAR",
                        help="[real] Year(s); the max matching year is scored")
    parser.add_argument("--scl-purity",      type=float, default=0.5,
                        help="[real] scl_purity_min passed to score_tile_year")
    parser.add_argument("--label",           default=None,
                        help="[real] Tag this run in the JSON output for comparison")
    parser.add_argument("--out-json",        default=None,
                        help="[real] Append run record (JSONL) for --compare")
    parser.add_argument("--compare",         default=None, metavar="JSONL",
                        help="Print side-by-side delta table from a runs JSONL and exit")
    args = parser.parse_args()

    if args.compare:
        run_compare(Path(args.compare))
        sys.exit(0)

    def _timeout_handler(signum, frame):  # noqa: ARG001
        print(f"\nTIMEOUT: exceeded {args.timeout}s")
        sys.exit(2)

    signal.signal(signal.SIGALRM, _timeout_handler)
    # Real-data runs include checkpoint load + IO from chunkstore; give them room.
    signal.alarm(max(args.timeout, 1800) if args.real else args.timeout)

    with tempfile.TemporaryDirectory(prefix="bench_score_") as _tmp:
        if args.real:
            run_real(args, Path(_tmp))
        else:
            run_bench(args, Path(_tmp))

    signal.alarm(0)
