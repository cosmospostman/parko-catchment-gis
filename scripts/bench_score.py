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

Exit codes: 0 = ok, 1 = assertion failure (--assert-*), 2 = timeout.
"""

from __future__ import annotations

import argparse
import datetime
import gc
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


def probe(tag: str, df: pl.DataFrame | None = None, rows: int | None = None) -> None:
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
    print(f"  [{_probes[-1].elapsed_s:7.2f}s]  {tag}  RSS={r:.2f}GB{delta_s}GB{frame_s}{rows_s}", flush=True)


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
        _compute_band_summaries_from_parquets,
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

    # 6. Pre-pass: band summaries
    band_summaries_dict = None
    if args.band_summaries:
        n_feat = len(s2_cols)
        band_summaries_dict = _compute_band_summaries_from_parquets(
            year_parquets=year_parquets,
            feature_cols=s2_cols,
            scl_purity_min=0.5,
            global_feat_mean=np.zeros(n_feat * 3, dtype=np.float32),
            global_feat_std=np.ones(n_feat * 3,  dtype=np.float32),
        )
        probe("prepass_band_summaries", rows=len(band_summaries_dict))

    # 7. Score (main timed stage)
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
        band_summaries=band_summaries_dict,
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
    parser.add_argument("--n-prep-workers",        type=int,   default=4)
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
    args = parser.parse_args()

    def _timeout_handler(signum, frame):  # noqa: ARG001
        print(f"\nTIMEOUT: exceeded {args.timeout}s")
        sys.exit(2)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout)

    with tempfile.TemporaryDirectory(prefix="bench_score_") as _tmp:
        run_bench(args, Path(_tmp))

    signal.alarm(0)
