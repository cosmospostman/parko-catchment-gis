"""Memory profile test for the TAM training pipeline.

Run:
    python scripts/profile_memory.py

Probes RSS at every major stage of _cmd_train and train_tam using a small
synthetic pixel_df (configurable scale). The test:

1. Builds a synthetic pixel_df matching the v10 schema (S2 + S1 rows).
2. Runs each stage of the data-processing pipeline one by one.
3. Records RSS and estimated Polars frame sizes before/after each operation.
4. Prints a summary table and asserts that no stage exceeds a user-configured
   peak RSS threshold (default: unlimited — use --assert-rss-gb to set one).

The synthetic frame uses the same column types and approximate row-count ratios
as v10. Default scale: 5M S2 rows + 1M S1 rows (~700 MB estimated) so the test
runs quickly. Use --scale 1.0 to approximate the real 145M-row dataset.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# RSS helpers
# ---------------------------------------------------------------------------

def rss_gb() -> float:
    """Current process RSS in GB (reads /proc/self/status)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def estimated_gb(df: pl.DataFrame) -> float:
    return df.estimated_size() / 1e9


# ---------------------------------------------------------------------------
# Probe recording
# ---------------------------------------------------------------------------

class Probe(NamedTuple):
    tag: str
    rss_gb: float
    frame_gb: float | None   # None when frame has been deleted
    elapsed_s: float


_probes: list[Probe] = []
_t0 = time.perf_counter()


def probe(tag: str, df: pl.DataFrame | None = None) -> None:
    _probes.append(Probe(
        tag=tag,
        rss_gb=rss_gb(),
        frame_gb=estimated_gb(df) if df is not None else None,
        elapsed_s=time.perf_counter() - _t0,
    ))


def print_report(assert_rss_gb: float | None = None) -> None:
    print("\n" + "=" * 78)
    print(f"{'Stage':<40}  {'RSS GB':>8}  {'Frame GB':>9}  {'Elapsed s':>9}")
    print("-" * 78)
    for p in _probes:
        frame_s = f"{p.frame_gb:9.2f}" if p.frame_gb is not None else f"{'(freed)':>9}"
        print(f"{p.tag:<40}  {p.rss_gb:8.2f}  {frame_s}  {p.elapsed_s:9.1f}")
    print("=" * 78)

    max_rss = max(p.rss_gb for p in _probes)
    print(f"\nPeak RSS: {max_rss:.2f} GB")

    if assert_rss_gb is not None:
        if max_rss > assert_rss_gb:
            print(f"\nFAIL: peak RSS {max_rss:.2f} GB exceeds limit {assert_rss_gb:.2f} GB")
            sys.exit(1)
        else:
            print(f"PASS: peak RSS {max_rss:.2f} GB <= limit {assert_rss_gb:.2f} GB")


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def make_pixel_df(
    n_pixels: int = 50_000,
    n_s2_obs_per_pixel: int = 100,
    n_s1_obs_per_pixel: int = 20,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a synthetic pixel_df matching the v10 schema.

    n_pixels × n_s2_obs_per_pixel S2 rows + n_pixels × n_s1_obs_per_pixel S1 rows.
    At the defaults (50k pixels, 100 S2 obs, 20 S1 obs) = 6M rows, ~1 GB.
    """
    rng = np.random.default_rng(seed)

    # Point IDs: half presence, half absence, spread across ~20 "regions"
    n_regions = 20
    pids_per_region = n_pixels // n_regions
    point_ids: list[str] = []
    labels_out: dict[str, float] = {}
    for reg in range(n_regions):
        cls = "presence" if reg < n_regions // 2 else "absence"
        lbl = 1.0 if cls == "presence" else 0.0
        for i in range(pids_per_region):
            pid = f"region_{reg}_{cls}_{i}_0_0"
            point_ids.append(pid)
            labels_out[pid] = lbl

    n_px = len(point_ids)

    def _make_rows(n_obs: int, source: str) -> pl.DataFrame:
        total = n_px * n_obs
        pids_rep = np.repeat(point_ids, n_obs)
        years = rng.integers(2020, 2024, size=total).astype(np.int32)
        doys = rng.integers(1, 366, size=total).astype(np.int32)
        lons = rng.uniform(140.0, 148.0, size=n_px).repeat(n_obs).astype(np.float32)
        lats = rng.uniform(-22.0, -18.0, size=n_px).repeat(n_obs).astype(np.float32)
        scl_purity = rng.uniform(0.3, 1.0, size=total).astype(np.float32)
        scl = rng.choice([4, 5, 6], size=total).astype(np.int8)

        # Build date as a Date column from year+doy for TAMDataset compatibility
        import datetime
        dates_list = [
            datetime.date(int(y), 1, 1) + datetime.timedelta(days=int(d) - 1)
            for y, d in zip(years, doys)
        ]
        data: dict[str, pl.Series] = {
            "point_id":   pl.Series(pids_rep),
            "lon":        pl.Series(lons),
            "lat":        pl.Series(lats),
            "year":       pl.Series(years),
            "doy":        pl.Series(doys),
            "date":       pl.Series(dates_list).cast(pl.Date),
            "scl_purity": pl.Series(scl_purity),
            "scl":        pl.Series(scl),
            "source":     pl.Series([source] * total),
        }

        if source == "S2":
            for band in ["B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12"]:
                data[band] = pl.Series(rng.uniform(0.01, 0.5, size=total).astype(np.float32))
            # S1 cols absent in S2 rows
            data["vh"] = pl.Series(np.full(total, None, dtype=object)).cast(pl.Float32)
            data["vv"] = pl.Series(np.full(total, None, dtype=object)).cast(pl.Float32)
        else:  # S1
            for band in ["B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12"]:
                data[band] = pl.Series(np.full(total, None, dtype=object)).cast(pl.Float32)
            data["vh"] = pl.Series(rng.uniform(0.001, 0.1, size=total).astype(np.float32))
            data["vv"] = pl.Series(rng.uniform(0.001, 0.1, size=total).astype(np.float32))

        return pl.DataFrame(data)

    s2 = _make_rows(n_s2_obs_per_pixel, "S2")
    s1 = _make_rows(n_s1_obs_per_pixel, "S1")
    df = pl.concat([s2, s1])
    del s2, s1
    gc.collect()
    return df, labels_out, point_ids


# ---------------------------------------------------------------------------
# Stage-by-stage pipeline replay
# ---------------------------------------------------------------------------

def run_profile(
    n_pixels: int,
    n_s2_obs: int,
    n_s1_obs: int,
    assert_rss_gb: float | None,
    pixel_zscore: bool,
    use_s1: bool,
) -> None:
    from tam.core.config import TAMConfig
    from tam.core.dataset import V9_FEATURE_COLS, V10_FEATURE_COLS, V10_S1_FEATURE_COLS, TAMDataset
    from tam.core.train import _compute_band_summaries, train_tam

    print(f"\nBuilding synthetic pixel_df: {n_pixels:,} pixels "
          f"× {n_s2_obs} S2 obs + {n_s1_obs} S1 obs per pixel ...")
    probe("baseline (no data)")

    pixel_df, labels, point_ids = make_pixel_df(n_pixels, n_s2_obs, n_s1_obs)
    probe("after make_pixel_df", pixel_df)
    print(f"  rows={len(pixel_df):,}  cols={pixel_df.width}  "
          f"estimated={estimated_gb(pixel_df):.2f} GB  RSS={rss_gb():.2f} GB")

    # --- Band summaries (per-tile, like pipeline.py) -----------------------
    print("\nStage: compute band summaries ...")
    band_summaries = _compute_band_summaries(pixel_df, V9_FEATURE_COLS)
    probe("after band summaries", pixel_df)
    print(f"  band_summaries: {len(band_summaries):,} pixels × {band_summaries.width} cols  "
          f"estimated={estimated_gb(band_summaries):.3f} GB")

    # --- Pixel zscore (per-tile, like pipeline.py) -------------------------
    # Import the actual pipeline function so we test the real implementation.
    if pixel_zscore:
        print("\nStage: pixel zscore (via pipeline._apply_pixel_zscore) ...")
        # _apply_pixel_zscore is a nested function in _cmd_train, so we replicate
        # its import pattern — import the module and call the helper directly.
        import importlib, sys as _sys
        _pip_mod = importlib.import_module("tam.pipeline")

        # Manually replicate _apply_pixel_zscore logic since it's a nested function.
        # This is a direct copy of the CURRENT implementation in pipeline.py so that
        # the test always exercises the same code path.
        zscore_cols = [c for c in V10_FEATURE_COLS if c in pixel_df.columns]
        if zscore_cols:
            has_source = "source" in pixel_df.columns
            s2_mask_arr = (pixel_df["source"] == "S2").to_numpy() if has_source else np.ones(len(pixel_df), dtype=bool)
            sort_idx = np.argsort(pixel_df["point_id"].to_numpy(), kind="stable")
            inv_idx = np.empty_like(sort_idx)
            inv_idx[sort_idx] = np.arange(len(pixel_df))
            pid_sorted = pixel_df["point_id"].to_numpy()[sort_idx]
            s2_mask_sorted = s2_mask_arr[sort_idx]
            boundaries = np.nonzero(pid_sorted[1:] != pid_sorted[:-1])[0] + 1

            s2_idx_sorted = np.where(s2_mask_sorted)[0]
            s2_pid_sorted = pid_sorted[s2_idx_sorted]
            s2_bounds = np.nonzero(s2_pid_sorted[1:] != s2_pid_sorted[:-1])[0] + 1 if len(s2_pid_sorted) > 1 else np.array([], dtype=np.int64)
            s2_starts = np.concatenate([[0], s2_bounds])
            s2_ns = np.diff(np.concatenate([s2_starts, [len(s2_pid_sorted)]]))

            updated_cols: dict = {}
            for c in zscore_cols:
                arr = pixel_df[c].to_numpy().astype(np.float32, copy=False)
                arr_sorted = arr[sort_idx].astype(np.float64)
                s2_vals = arr_sorted[s2_idx_sorted]
                group_sums = np.add.reduceat(s2_vals, s2_starts)
                group_means = group_sums / s2_ns
                mean_per_s2 = np.repeat(group_means, s2_ns)
                diffs_sq = (s2_vals - mean_per_s2) ** 2
                group_var_sums = np.add.reduceat(diffs_sq, s2_starts)
                group_stds = np.sqrt(group_var_sums / np.maximum(s2_ns - 1, 1))
                group_stds = np.maximum(group_stds, 1e-6)
                std_per_s2 = np.repeat(group_stds, s2_ns)
                normed_s2 = (s2_vals - mean_per_s2) / std_per_s2
                arr_sorted[s2_idx_sorted] = normed_s2
                updated_cols[c] = arr_sorted[inv_idx].astype(np.float32)
            pixel_df = pixel_df.with_columns([pl.Series(c, updated_cols[c]) for c in zscore_cols])
            del updated_cols
            gc.collect()

        probe("after pixel zscore (numpy/pipeline impl)", pixel_df)
        print(f"  after zscore: estimated={estimated_gb(pixel_df):.2f} GB  RSS={rss_gb():.2f} GB")

    # --- Column trim (mirrors train_tam early trim) ------------------------
    print("\nStage: column trim ...")
    _feature_cols_base = set(V10_FEATURE_COLS)
    _active_s1_cols = set(V10_S1_FEATURE_COLS) if use_s1 else set()
    _keep_cols = {"point_id", "date", "year", "doy", "scl_purity", "scl", "source"} | \
                 _feature_cols_base | _active_s1_cols | {"vh", "vv"}
    pixel_df = pixel_df.select([c for c in pixel_df.columns if c in _keep_cols])
    _str_cols = [c for c in ("point_id", "source") if c in pixel_df.columns]
    if _str_cols:
        pixel_df = pixel_df.with_columns([pl.col(c).cast(pl.Categorical) for c in _str_cols])
    gc.collect()
    probe("after column trim + categorical cast", pixel_df)
    print(f"  estimated={estimated_gb(pixel_df):.2f} GB  RSS={rss_gb():.2f} GB")

    # --- Refcount free verification: does del actually free RSS? -------------
    # Allocate a temporary frame, measure RSS before and after del with refcount=1.
    print("\nRefcount free check ...")
    _tmp = pl.DataFrame({"x": np.arange(5_000_000, dtype=np.float32)})   # ~20 MB
    rss_before = rss_gb()
    del _tmp
    gc.collect()
    rss_after = rss_gb()
    freed = rss_before - rss_after
    print(f"  Allocated 20 MB, del+gc freed {freed*1e3:.0f} MB of RSS "
          f"({'OK — del actually frees' if freed > 0.01 else 'WARNING — RSS did not drop; malloc arena waste'})")

    # --- SCL=6 exclusion --------------------------------------------------
    if "scl" in pixel_df.columns and "source" in pixel_df.columns:
        n_before = len(pixel_df)
        pixel_df = pixel_df.filter(
            ~((pl.col("source") == "S2") & (pl.col("scl") == 6))
        )
        pixel_df = pixel_df.drop("scl")
        print(f"\nStage: SCL=6 exclusion — removed {n_before - len(pixel_df):,} rows")
    gc.collect()
    probe("after SCL=6 exclusion", pixel_df)

    # --- Train/val split --------------------------------------------------
    print("\nStage: train/val split ...")
    all_pids = list(labels.keys())
    rng_split = np.random.default_rng(0)
    val_mask = rng_split.random(len(all_pids)) < 0.2
    val_pids   = {p for p, v in zip(all_pids, val_mask) if v}
    train_pids = {p for p, v in zip(all_pids, val_mask) if not v}

    probe("before split (pixel_df live)", pixel_df)

    train_pixel_df = pixel_df.filter(pl.col("point_id").is_in(train_pids))
    val_pixel_df   = pixel_df.filter(pl.col("point_id").is_in(val_pids))
    probe("after filter (train+val+original all live)", pixel_df)
    print(f"  pixel_df={estimated_gb(pixel_df):.2f} GB  "
          f"train={estimated_gb(train_pixel_df):.2f} GB  "
          f"val={estimated_gb(val_pixel_df):.2f} GB  "
          f"RSS={rss_gb():.2f} GB")

    del pixel_df
    gc.collect()
    probe("after del pixel_df (outer ref gone)", train_pixel_df)
    print(f"  train={estimated_gb(train_pixel_df):.2f} GB  "
          f"val={estimated_gb(val_pixel_df):.2f} GB  RSS={rss_gb():.2f} GB")

    # --- TAMDataset construction -------------------------------------------
    print("\nStage: TAMDataset(train) ...")

    # Build pixel-year labels
    py_labels_train: dict[tuple[str, int], float] = {}
    for pid in train_pids:
        for yr in [2020, 2021, 2022, 2023]:
            py_labels_train[(pid, yr)] = labels[pid]
    py_labels_val: dict[tuple[str, int], float] = {}
    for pid in val_pids:
        for yr in [2020, 2021, 2022, 2023]:
            py_labels_val[(pid, yr)] = labels[pid]

    cfg = TAMConfig(
        n_epochs=1,
        patience=1,
        use_s1="mixed" if use_s1 else False,
        pixel_zscore=False,   # already applied above
        use_band_summaries=True,
        doy_density_norm=False,
        feature_cols_override=tuple(V10_FEATURE_COLS),
        s1_feature_cols=tuple(V10_S1_FEATURE_COLS) if use_s1 else (),
        n_bands=len(V10_FEATURE_COLS) + (len(V10_S1_FEATURE_COLS) if use_s1 else 0),
        d_model=256,
        n_layers=3,
        dropout=0.5,
        n_annual_features=0,
    )

    probe("before TAMDataset train", train_pixel_df)
    t_ds_start = time.perf_counter()
    try:
        train_ds = TAMDataset(
            train_pixel_df, py_labels_train,
            annual_features_df=band_summaries,
            use_s1="mixed" if use_s1 else False,
            pixel_zscore=False,
            feature_cols_override=list(V10_FEATURE_COLS),
            s1_feature_cols_override=list(V10_S1_FEATURE_COLS) if use_s1 else None,
        )
    except Exception as exc:
        print(f"  TAMDataset(train) raised: {exc}")
        probe("TAMDataset(train) FAILED", train_pixel_df)
        print_report(assert_rss_gb)
        return

    probe("after TAMDataset train", train_pixel_df)
    print(f"  train_ds windows={len(train_ds)}  "
          f"elapsed={time.perf_counter() - t_ds_start:.1f}s  RSS={rss_gb():.2f} GB")

    band_mean, band_std = train_ds.band_stats
    del train_pixel_df
    gc.collect()
    probe("after del train_pixel_df", None)
    print(f"  RSS after freeing train_pixel_df: {rss_gb():.2f} GB")

    print("\nStage: TAMDataset(val) ...")
    probe("before TAMDataset val", val_pixel_df)
    try:
        val_ds = TAMDataset(
            val_pixel_df, py_labels_val,
            band_mean=band_mean, band_std=band_std,
            annual_features_df=band_summaries,
            annual_feat_mean=train_ds.annual_feat_mean,
            annual_feat_std=train_ds.annual_feat_std,
            use_s1="mixed" if use_s1 else False,
            pixel_zscore=False,
            feature_cols_override=list(V10_FEATURE_COLS),
            s1_feature_cols_override=list(V10_S1_FEATURE_COLS) if use_s1 else None,
        )
    except Exception as exc:
        print(f"  TAMDataset(val) raised: {exc}")
        probe("TAMDataset(val) FAILED", val_pixel_df)
        print_report(assert_rss_gb)
        return

    probe("after TAMDataset val", val_pixel_df)
    print(f"  val_ds windows={len(val_ds)}  RSS={rss_gb():.2f} GB")

    del val_pixel_df, band_summaries
    gc.collect()
    probe("after del val_pixel_df + band_summaries", None)
    print(f"  RSS after freeing val frames: {rss_gb():.2f} GB")

    # --- Model init -------------------------------------------------------
    import torch
    from tam.core.model import TAMClassifier

    n_annual = 0
    if train_ds.annual_feat_mean is not None:
        n_annual = len(train_ds.annual_feat_mean)
    cfg.n_annual_features = n_annual

    print(f"\nStage: model init (d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_bands={cfg.n_bands}, n_annual={n_annual}) ...")
    probe("before model init", None)
    model = TAMClassifier.from_config(cfg)
    model.to("cpu")
    probe("after model init", None)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}  RSS={rss_gb():.2f} GB")

    print_report(assert_rss_gb)

    # --- Scaling analysis -------------------------------------------------
    print("\nMemory scaling summary:")
    n_total_rows = n_pixels * (n_s2_obs + n_s1_obs)
    for p in _probes:
        if p.frame_gb is not None:
            bytes_per_row = p.frame_gb * 1e9 / n_total_rows
            print(f"  {p.tag:<40}  {bytes_per_row:.1f} bytes/row")


# ---------------------------------------------------------------------------
# Refcount verification
# ---------------------------------------------------------------------------

def verify_refcount_behaviour() -> None:
    """Verify that [frame].pop() does NOT free the outer reference.

    This is the confirmed Python semantics: the list is temporary, but after
    .pop() the function's local binding holds the only reference. However, the
    OUTER name is still alive throughout the call unless explicitly deleted.
    """
    import sys as _sys

    print("\n--- Refcount verification ---")
    a = pl.DataFrame({"x": np.arange(1_000_000, dtype=np.float32)})
    print(f"  refcount(a) before call site: {_sys.getrefcount(a)}")

    def check_refcount(df: pl.DataFrame) -> int:
        cnt = _sys.getrefcount(df)
        del df
        return cnt

    # Plain pass: refcount inside = 3 (a + arg + getrefcount frame)
    rc_plain = check_refcount(df=a)
    print(f"  refcount inside fn (plain pass): {rc_plain}")

    # [a].pop() pass: still 3 inside — outer 'a' is still alive
    rc_pop = check_refcount(df=[a].pop())
    print(f"  refcount inside fn ([a].pop()): {rc_pop}")
    print(f"  'a' still accessible after call: {a is not None}  (len={len(a):,})")

    # Correct approach: del a BEFORE the call
    b = pl.DataFrame({"x": np.arange(1_000_000, dtype=np.float32)})
    b_ref = b  # simulate pipeline holding both names
    del b
    # Now b_ref is the only name; pass it and del inside
    rc_del = check_refcount(df=b_ref)
    print(f"  refcount inside fn (del outer first): {rc_del}")
    del b_ref

    print()
    print("  Conclusion: [frame].pop() does NOT drop the outer reference.")
    print("  Correct fix: del pixel_df in the caller before or after the call,")
    print("  or restructure so the caller never holds the name after passing it.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAM training pipeline memory profiler")
    parser.add_argument("--n-pixels",      type=int,   default=50_000,
                        help="Number of unique pixels (default: 50000)")
    parser.add_argument("--n-s2-obs",      type=int,   default=100,
                        help="S2 observations per pixel (default: 100)")
    parser.add_argument("--n-s1-obs",      type=int,   default=20,
                        help="S1 observations per pixel (default: 20)")
    parser.add_argument("--scale",         type=float, default=None,
                        help="Scale factor relative to real v10 dataset "
                             "(145M S2 rows, 25M S1 rows across 100k pixels). "
                             "Overrides --n-pixels/--n-s2-obs/--n-s1-obs.")
    parser.add_argument("--assert-rss-gb", type=float, default=None,
                        help="Fail if peak RSS exceeds this value (GB)")
    parser.add_argument("--pixel-zscore",  action="store_true", default=False,
                        help="Enable pixel zscore stage (mirrors v10 pipeline.py)")
    parser.add_argument("--use-s1",        action="store_true", default=False,
                        help="Enable S1 mixed mode (mirrors v10)")
    parser.add_argument("--verify-refcount", action="store_true", default=False,
                        help="Run refcount verification and exit")
    args = parser.parse_args()

    if args.verify_refcount:
        verify_refcount_behaviour()
        sys.exit(0)

    if args.scale is not None:
        # Real v10: ~100k pixels × 1450 S2 obs + 250 S1 obs
        args.n_pixels  = max(100, int(100_000 * args.scale))
        args.n_s2_obs  = max(10,  int(1_450   * args.scale))
        args.n_s1_obs  = max(5,   int(250     * args.scale))

    run_profile(
        n_pixels=args.n_pixels,
        n_s2_obs=args.n_s2_obs,
        n_s1_obs=args.n_s1_obs,
        assert_rss_gb=args.assert_rss_gb,
        pixel_zscore=args.pixel_zscore,
        use_s1=args.use_s1,
    )
