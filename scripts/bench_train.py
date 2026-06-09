"""TAM training pipeline profiler and performance harness.

Synthesises a realistic pixel_df + labels + pixel_coords at configurable scale,
replays every named stage of train_tam + TAMDataset.__init__ in order, and records
RSS (GB) and wall time (s) at each probe point in a single merged table.

Run:
    python scripts/bench_train.py
    python scripts/bench_train.py --use-s1 --presence-filter
    python scripts/bench_train.py --scale 0.05          # ~5% of real v10
    python scripts/bench_train.py --assert-rss-gb 4.0 --assert-wall-s 60
    python scripts/bench_train.py --scale 1.0 --timeout 120

Exit codes: 0 = ok, 1 = assertion failure (--assert-*), 2 = timeout.
"""

from __future__ import annotations

import argparse
import gc
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
    print(f"  [{_probes[-1].elapsed_s:7.2f}s]  {tag}  RSS={r:.2f}GB{delta_s}GB{frame_s}{rows_s}")


def print_report(assert_rss_gb: float | None = None, assert_wall_s: float | None = None) -> None:
    W = 90
    print("\n" + "=" * W)
    print(f"  {'Stage':<40}  {'RSS GB':>7}  {'Δ RSS':>7}  {'Frame GB':>9}  {'Elapsed s':>9}  {'Rows':>12}")
    print("-" * W)
    for p in _probes:
        frame_s = f"{p.frame_gb:9.2f}" if p.frame_gb is not None else f"{'(freed)':>9}"
        rows_s  = f"{p.rows:12,}" if p.rows is not None else f"{'':12}"
        delta_s = f"{p.delta_rss_gb:+7.2f}"
        print(f"  {p.tag:<40}  {p.rss_gb:7.2f}  {delta_s}  {frame_s}  {p.elapsed_s:9.1f}  {rows_s}")
    print("=" * W)

    peak_rss   = max(p.rss_gb  for p in _probes)
    total_wall = _probes[-1].elapsed_s
    print(f"\nPeak RSS: {peak_rss:.2f} GB   Total wall: {total_wall:.1f} s")

    ok = True
    if assert_rss_gb is not None:
        if peak_rss > assert_rss_gb:
            print(f"FAIL: peak RSS {peak_rss:.2f} GB > limit {assert_rss_gb:.2f} GB")
            ok = False
        else:
            print(f"PASS: peak RSS {peak_rss:.2f} GB <= limit {assert_rss_gb:.2f} GB")
    if assert_wall_s is not None:
        if total_wall > assert_wall_s:
            print(f"FAIL: wall time {total_wall:.1f} s > limit {assert_wall_s:.1f} s")
            ok = False
        else:
            print(f"PASS: wall time {total_wall:.1f} s <= limit {assert_wall_s:.1f} s")

    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_pixels: int = 5_000,
    n_s2_obs: int = 80,
    n_s1_obs: int = 15,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, float], pl.DataFrame]:
    """Build synthetic pixel_df + labels + pixel_coords matching v10 schema.

    Point IDs use format region_{reg}_{cls}_{i}_0_0 so that _site_class() in
    train.py parses them correctly. No spectral index columns are included —
    they are computed downstream exactly as in the real pipeline.
    """
    import datetime

    rng = np.random.default_rng(seed)
    n_regions = 20
    pids_per_region = max(1, n_pixels // n_regions)

    point_ids: list[str] = []
    labels: dict[str, float] = {}
    lons_px: list[float] = []
    lats_px: list[float] = []

    for reg in range(n_regions):
        cls = "presence" if reg < n_regions // 2 else "absence"
        lbl = 1.0 if cls == "presence" else 0.0
        for i in range(pids_per_region):
            pid = f"region_{reg}_{cls}_{i}_0_0"
            point_ids.append(pid)
            labels[pid] = lbl
            lons_px.append(float(rng.uniform(140.0, 148.0)))
            lats_px.append(float(rng.uniform(-22.0, -18.0)))

    n_px = len(point_ids)
    pixel_coords = pl.DataFrame({
        "point_id": point_ids,
        "lon": np.array(lons_px, dtype=np.float32),
        "lat": np.array(lats_px, dtype=np.float32),
    })

    band_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    def _make_rows(n_obs: int, source: str) -> pl.DataFrame:
        total = n_px * n_obs
        pids_rep = np.repeat(point_ids, n_obs)

        # Build dates spanning 2020-2023 without any Python datetime objects.
        # Compute year and doy from ordinals using integer arithmetic.
        base_ord = datetime.date(2020, 1, 1).toordinal()
        raw_doys = rng.integers(1, 365, size=(n_px, n_obs))
        raw_doys.sort(axis=1)
        yr_offsets = rng.integers(0, 4, size=(n_px, n_obs)) * 365
        ordinals = (base_ord + yr_offsets + raw_doys - 1).flatten().astype(np.int32)

        # Derive year from ordinal: use a reference epoch (days since 0001-01-01).
        # Python's date.toordinal() uses proleptic Gregorian; approximate year via
        # integer division then correct by checking Jan-1 of that year.
        # Faster: use numpy datetime64 arithmetic — ordinal → date64 → year/doy.
        epoch_offset = datetime.date(1970, 1, 1).toordinal()  # 719163
        unix_days = (ordinals - epoch_offset).astype("int64")
        dt64 = unix_days.astype("datetime64[D]")
        years = dt64.astype("datetime64[Y]").astype("int32") + 1970
        year_start = ((years - 1970).astype("int64")).view("datetime64[Y]").astype("datetime64[D]").astype("int64")
        doys = (unix_days - year_start + 1).astype(np.int32)

        scl_purity = rng.uniform(0.3, 1.0, size=total).astype(np.float32)
        scl        = rng.choice([4, 5, 6], size=total).astype(np.int8)

        data: dict[str, object] = {
            "point_id":   pl.Series(pids_rep),
            "lon":        pl.Series(np.repeat(np.array(lons_px, dtype=np.float32), n_obs)),
            "lat":        pl.Series(np.repeat(np.array(lats_px, dtype=np.float32), n_obs)),
            "date":       pl.Series(unix_days.astype(np.int32), dtype=pl.Date),
            "year":       pl.Series(years),
            "doy":        pl.Series(doys),
            "scl_purity": pl.Series(scl_purity),
            "scl":        pl.Series(scl),
            "source":     pl.Series(np.full(total, source)),
        }

        if source == "S2":
            for band in band_names:
                data[band] = pl.Series(rng.uniform(0.01, 0.5, size=total).astype(np.float32))
            data["vh"] = pl.Series(np.full(total, None, dtype=object)).cast(pl.Float32)
            data["vv"] = pl.Series(np.full(total, None, dtype=object)).cast(pl.Float32)
        else:  # S1
            for band in band_names:
                data[band] = pl.Series(np.full(total, None, dtype=object)).cast(pl.Float32)
            # Realistic linear backscatter values (will be converted to dB downstream)
            data["vh"] = pl.Series(rng.uniform(0.001, 0.08, size=total).astype(np.float32))
            data["vv"] = pl.Series(rng.uniform(0.002, 0.12, size=total).astype(np.float32))

        return pl.DataFrame(data)

    s2 = _make_rows(n_s2_obs, "S2")
    s1 = _make_rows(n_s1_obs, "S1")
    pixel_df = pl.concat([s2, s1])
    del s2, s1
    gc.collect()

    return pixel_df, labels, pixel_coords


# ---------------------------------------------------------------------------
# Inline helpers (replicated from train.py internals)
# ---------------------------------------------------------------------------

def _broadcast_to_pixel_years(
    pixel_years: pl.DataFrame,
    lbl: dict[str, float],
) -> dict[tuple[str, int], float]:
    """Expand pixel-level label dict to (point_id, year) keys."""
    lbl_pids = set(lbl)
    return {
        (row[0], row[1]): lbl[row[0]]
        for row in pixel_years.filter(pl.col("point_id").is_in(lbl_pids)).iter_rows()
    }


def _build_presence_slim(
    pixel_df: pl.DataFrame,
    labels: dict[str, float],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Extract S1/S2 slim frames for presence filter (mirrors train.py logic)."""
    from tam.core.constants import DRY_DOY_MIN, DRY_DOY_MAX

    if "vh" not in pixel_df.columns or "source" not in pixel_df.columns:
        return None, None

    presence_pids = {pid for pid, lbl in labels.items() if lbl == 1.0}

    s1_slim = (
        pixel_df.lazy()
        .filter(
            (pl.col("source") == "S1") &
            pl.col("point_id").is_in(presence_pids) &
            pl.col("doy").is_between(DRY_DOY_MIN, DRY_DOY_MAX)
        )
        .select(["point_id", "year", "vh", "doy"])
        .collect()
    )

    ndvi_expr = (pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))
    s2_slim = (
        pixel_df.lazy()
        .filter(
            (pl.col("source") == "S2") &
            pl.col("point_id").is_in(presence_pids) &
            pl.col("doy").is_between(DRY_DOY_MIN, DRY_DOY_MAX)
        )
        .with_columns(ndvi_expr.alias("NDVI"))
        .select(["point_id", "year", "doy", "NDVI"])
        .collect()
    )

    return s1_slim, s2_slim


# ---------------------------------------------------------------------------
# Main profiling run
# ---------------------------------------------------------------------------

def run_profile(
    n_pixels: int,
    n_s2_obs: int,
    n_s1_obs: int,
    seed: int,
    use_s1: bool,
    pixel_zscore: bool,
    run_presence_filter: bool,
    dataset_subprocess: bool,
    n_shards: int,
    assert_rss_gb: float | None,
    assert_wall_s: float | None,
    out_dir: Path,
) -> None:
    from tam.core.config import TAMConfig
    from tam.core.dataset import TAMDataset, V10_FEATURE_COLS, V10_S1_FEATURE_COLS
    from tam.core.model import TAMClassifier
    from tam.core.train import (
        _apply_presence_filter,
        _build_dataset_subprocess,
        _build_dataset_sharded,
        _compute_band_stats_subprocess,
        _write_dataset_parquet,
        _compute_band_summaries,
        _site_class,
        spatial_split,
    )
    from tam.core.dataset import lin_to_db

    total_rows = n_pixels * (n_s2_obs + n_s1_obs)
    print(f"\nTAM training pipeline bench")
    print(f"  n_pixels={n_pixels:,}  n_s2_obs={n_s2_obs}  n_s1_obs={n_s1_obs}  "
          f"total_rows≈{total_rows:,}  use_s1={use_s1}  "
          f"pixel_zscore={pixel_zscore}  presence_filter={run_presence_filter}  "
          f"dataset_subprocess={dataset_subprocess}")
    print()

    probe("baseline")

    # --- Synthetic data ---------------------------------------------------
    print("Building synthetic data ...")
    pixel_df, labels, pixel_coords = make_synthetic_data(n_pixels, n_s2_obs, n_s1_obs, seed)
    probe("after make_pixel_df", pixel_df)

    # --- Label split -------------------------------------------------------
    train_labels, val_labels = spatial_split(labels, pixel_coords, val_frac=0.2)

    # --- Column trim -------------------------------------------------------
    _s1_raw = ["vh", "vv"] if use_s1 else []
    _feature_cols_base = set(V10_FEATURE_COLS)
    _active_s1_cols    = set(V10_S1_FEATURE_COLS) if use_s1 else set()
    _keep_cols = {"point_id", "date", "year", "doy", "scl_purity", "scl", "source"} | \
                 _feature_cols_base | _active_s1_cols | set(_s1_raw)
    pixel_df = pixel_df.select([c for c in pixel_df.columns if c in _keep_cols])
    _str_cols = [c for c in ("point_id", "source") if c in pixel_df.columns]
    if _str_cols:
        pixel_df = pixel_df.with_columns([pl.col(c).cast(pl.Categorical) for c in _str_cols])
    gc.collect()
    probe("after column trim + categorical cast", pixel_df)

    # --- SCL=6 exclusion --------------------------------------------------
    if "scl" in pixel_df.columns and "source" in pixel_df.columns:
        pixel_df = pixel_df.filter(~((pl.col("source") == "S2") & (pl.col("scl") == 6)))
    if "scl" in pixel_df.columns:
        pixel_df = pixel_df.drop("scl")
    gc.collect()
    probe("after SCL=6 exclusion", pixel_df)

    # --- Band summaries ---------------------------------------------------
    band_summaries = _compute_band_summaries(pixel_df, V10_FEATURE_COLS)
    gc.collect()
    probe("after band summaries", pixel_df, rows=len(band_summaries))

    # --- Presence slim extract --------------------------------------------
    s1_slim: pl.DataFrame | None = None
    s2_slim: pl.DataFrame | None = None
    if run_presence_filter:
        s1_slim, s2_slim = _build_presence_slim(pixel_df, labels)
    probe("after presence slim extract", pixel_df)

    # --- S1 drop ----------------------------------------------------------
    if "source" in pixel_df.columns and not use_s1:
        pixel_df = pixel_df.filter(pl.col("source") == "S2")
        gc.collect()
    probe("after S1 drop", pixel_df)

    # --- Pixel-year broadcast ---------------------------------------------
    labeled_pids = set(labels.keys())
    pixel_years = (
        pixel_df.filter(pl.col("point_id").is_in(labeled_pids))
        .select(["point_id", "year"])
        .unique()
    )
    train_py_labels = _broadcast_to_pixel_years(pixel_years, train_labels)
    val_py_labels   = _broadcast_to_pixel_years(pixel_years, val_labels)
    gc.collect()
    probe("after pixel-year broadcast", pixel_df,
          rows=len(train_py_labels) + len(val_py_labels))

    # --- Presence filter --------------------------------------------------
    noise_removed: dict = {}
    if run_presence_filter and s1_slim is not None and len(s1_slim) > 0:
        from tam.core.constants import DRY_DOY_MIN, DRY_DOY_MAX

        cfg_tmp = TAMConfig()
        pid_to_sc = {pid: _site_class(pid) for pid in labeled_pids}

        vh_lin   = s1_slim["vh"].cast(pl.Float32).to_numpy()
        vh_db    = lin_to_db(vh_lin)
        doy_vals = s1_slim["doy"].to_numpy()
        dry_mask = (doy_vals >= DRY_DOY_MIN) & (doy_vals <= DRY_DOY_MAX) & np.isfinite(vh_db)

        dry_s1 = pl.DataFrame({
            "point_id": s1_slim["point_id"].to_numpy()[dry_mask],
            "year":     s1_slim["year"].to_numpy()[dry_mask],
            "_vh_db":   vh_db[dry_mask].astype(np.float32),
        })
        _mean_vh = dry_s1.group_by(["point_id", "year"]).agg(pl.col("_vh_db").mean().alias("mean_vh"))
        mean_vh_dry_py = {(r[0], r[1]): r[2] for r in _mean_vh.iter_rows()}

        mean_ndvi_dry_py = None
        if s2_slim is not None and len(s2_slim) > 0:
            dry_s2 = s2_slim.filter(pl.col("NDVI").is_not_null())
            if len(dry_s2) > 0:
                _mean_ndvi = dry_s2.group_by(["point_id", "year"]).agg(
                    pl.col("NDVI").mean().alias("mean_ndvi")
                )
                mean_ndvi_dry_py = {(r[0], r[1]): r[2] for r in _mean_ndvi.iter_rows()}

        train_py_labels = _apply_presence_filter(train_py_labels, mean_vh_dry_py, cfg_tmp, pid_to_sc, noise_removed, mean_ndvi_dry_py)
        val_py_labels   = _apply_presence_filter(val_py_labels,   mean_vh_dry_py, cfg_tmp, pid_to_sc, noise_removed, mean_ndvi_dry_py)
        del s1_slim, s2_slim, dry_s1, _mean_vh
        gc.collect()
    probe("after presence filter", pixel_df)

    # --- Train/val split --------------------------------------------------
    train_pids_set = {k[0] for k in train_py_labels}
    val_pids_set   = {k[0] for k in val_py_labels}

    train_pixel_df = pixel_df.filter(pl.col("point_id").is_in(train_pids_set))
    val_pixel_df   = pixel_df.filter(pl.col("point_id").is_in(val_pids_set))
    del pixel_df
    gc.collect()
    probe("after split + del pixel_df", train_pixel_df,
          rows=len(train_pixel_df) + len(val_pixel_df))

    # --- TAMDataset construction ------------------------------------------
    cfg = TAMConfig(
        n_epochs=1,
        patience=1,
        use_s1="mixed" if use_s1 else False,
        pixel_zscore=pixel_zscore,
        feature_cols_override=tuple(V10_FEATURE_COLS),
        s1_feature_cols=tuple(V10_S1_FEATURE_COLS) if use_s1 else (),
        n_bands=len(V10_FEATURE_COLS) + (len(V10_S1_FEATURE_COLS) if use_s1 else 0),
        d_model=32,
        n_layers=1,
        dropout=0.3,
        n_annual_features=0,
    )
    _ds_kwargs = dict(
        annual_features_df=band_summaries,
        use_s1="mixed" if use_s1 else False,
        pixel_zscore=pixel_zscore,
        feature_cols_override=list(V10_FEATURE_COLS),
        s1_feature_cols_override=list(V10_S1_FEATURE_COLS) if use_s1 else None,
    )

    if dataset_subprocess:
        import tempfile
        _shm_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
        _ds_tmp = tempfile.TemporaryDirectory(prefix="bench_ds_", dir=_shm_base)
        _ds_tmp_path = Path(_ds_tmp.name)

        _train_parquet = _write_dataset_parquet(train_pixel_df, _ds_tmp_path, "train")
        del train_pixel_df
        gc.collect()
        probe("after write train parquet + free frame")

        # --- Band stats subprocess -------------------------------------------
        _use_s1_mode = "mixed" if use_s1 else False
        _npz = _ds_tmp_path / "band_stats"
        _gf_parquet = _ds_tmp_path / "annual_feat.parquet"
        band_summaries.write_parquet(str(_gf_parquet))
        probe("before band stats subprocess")
        _stats = _compute_band_stats_subprocess(
            parquet_path=_train_parquet,
            out_npz=_npz,
            use_s1=_use_s1_mode,
            feature_cols=list(V10_FEATURE_COLS) + (list(V10_S1_FEATURE_COLS) if use_s1 else []),
            s1_feature_cols=list(V10_S1_FEATURE_COLS) if use_s1 else [],
            scl_purity_min=0.5,
            s1_despeckle_window=0,
            annual_features_df_path=_gf_parquet,
        )
        band_mean = _stats["band_mean"]
        band_std  = _stats["band_std"]
        probe("after band stats subprocess")

        if n_shards > 1:
            probe(f"before TAMDataset(train) sharded n={n_shards}")
            _train_kwargs = dict(**_ds_kwargs, doy_jitter=0)
            train_ds = _build_dataset_sharded(
                parquet_path=_train_parquet,
                labels=train_py_labels,
                tmp_dir=_ds_tmp_path,
                name="train",
                kwargs=_train_kwargs,
                n_shards=n_shards,
                band_mean=band_mean,
                band_std=band_std,
                annual_feat_mean=_stats["annual_feat_mean"],
                annual_feat_std=_stats["annual_feat_std"],
            )
            probe(f"after TAMDataset(train) sharded n={n_shards}", rows=len(train_ds))
        else:
            probe("before TAMDataset(train) subprocess")
            train_ds = _build_dataset_subprocess(
                _train_parquet, train_py_labels, _ds_tmp_path, "train",
                kwargs=dict(**_ds_kwargs, doy_jitter=0,
                            band_mean=band_mean, band_std=band_std,
                            annual_feat_mean=_stats["annual_feat_mean"],
                            annual_feat_std=_stats["annual_feat_std"]),
            )
            probe("after TAMDataset(train) subprocess", rows=len(train_ds))

        _val_parquet = _write_dataset_parquet(val_pixel_df, _ds_tmp_path, "val")
        del val_pixel_df
        gc.collect()
        probe("after write val parquet + free frame")

        probe("before TAMDataset(val) subprocess")
        val_ds = _build_dataset_subprocess(
            _val_parquet, val_py_labels, _ds_tmp_path, "val",
            kwargs=dict(**_ds_kwargs,
                        band_mean=band_mean, band_std=band_std,
                        annual_feat_mean=train_ds.annual_feat_mean,
                        annual_feat_std=train_ds.annual_feat_std,
                        doy_jitter=0),
        )
        probe("after TAMDataset(val) subprocess", rows=len(val_ds))

        del band_summaries
        gc.collect()
        probe("after del band_summaries (subprocess)")

    else:
        def _ds_probe(tag: str) -> None:
            probe(tag)

        try:
            train_ds = TAMDataset(
                train_pixel_df, train_py_labels,
                _log_rss=_ds_probe, **_ds_kwargs,
            )
        except Exception as exc:
            print(f"  TAMDataset(train) raised: {exc}")
            probe("TAMDataset(train) FAILED", train_pixel_df)
            print_report(assert_rss_gb, assert_wall_s)
            return

        band_mean, band_std = train_ds.band_stats
        probe("TAMDataset(train) done", train_pixel_df, rows=len(train_ds))
        del train_pixel_df
        gc.collect()
        probe("after del train_pixel_df")

        try:
            val_ds = TAMDataset(
                val_pixel_df, val_py_labels,
                band_mean=band_mean, band_std=band_std,
                annual_feat_mean=train_ds.annual_feat_mean,
                annual_feat_std=train_ds.annual_feat_std,
                doy_jitter=0,
                _log_rss=_ds_probe,
                **_ds_kwargs,
            )
        except Exception as exc:
            print(f"  TAMDataset(val) raised: {exc}")
            probe("TAMDataset(val) FAILED", val_pixel_df)
            print_report(assert_rss_gb, assert_wall_s)
            return

        probe("TAMDataset(val) done", val_pixel_df, rows=len(val_ds))
        del val_pixel_df, band_summaries
        gc.collect()
        probe("after del val frames + band_summaries")

    # --- Model init -------------------------------------------------------
    import torch
    n_annual = int(train_ds._n_annual)
    cfg.n_annual_features = n_annual
    model = TAMClassifier.from_config(cfg)
    model.to("cpu")
    probe("after model init", rows=sum(p.numel() for p in model.parameters()))

    signal.alarm(0)  # cancel timeout
    print_report(assert_rss_gb, assert_wall_s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAM training pipeline profiler")
    parser.add_argument("--n-pixels",       type=int,   default=5_000,
                        help="Unique pixels (default: 5000)")
    parser.add_argument("--n-s2-obs",       type=int,   default=80,
                        help="S2 observations per pixel (default: 80)")
    parser.add_argument("--n-s1-obs",       type=int,   default=15,
                        help="S1 observations per pixel (default: 15)")
    parser.add_argument("--scale",          type=float, default=None,
                        help="Scale relative to real v10 (100k px × 1450 S2 + 250 S1). "
                             "Overrides --n-pixels/--n-s2-obs/--n-s1-obs.")
    parser.add_argument("--use-s1",         action="store_true", default=False,
                        help="Enable mixed S1+S2 mode")
    parser.add_argument("--pixel-zscore",   action="store_true", default=False,
                        help="Enable pixel zscore stage")
    parser.add_argument("--presence-filter", action="store_true", default=False,
                        help="Enable presence filter stage (requires S1 rows)")
    parser.add_argument("--dataset-subprocess", action="store_true", default=False,
                        help="Build TAMDataset in subprocess to reclaim jemalloc phantom arenas")
    parser.add_argument("--n-shards",          type=int,   default=1,
                        help="Number of dataset shards (>1 uses _build_dataset_sharded + band stats subprocess)")
    parser.add_argument("--timeout",        type=int,   default=120,
                        help="Abort with exit 2 after this many seconds (default: 120)")
    parser.add_argument("--assert-rss-gb",  type=float, default=None,
                        help="Exit 1 if peak RSS exceeds this (GB)")
    parser.add_argument("--assert-wall-s",  type=float, default=None,
                        help="Exit 1 if total wall time exceeds this (seconds)")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    if args.scale is not None:
        args.n_pixels = max(100,  int(100_000 * args.scale))
        args.n_s2_obs = max(10,   int(1_450   * args.scale))
        args.n_s1_obs = max(5,    int(250     * args.scale))

    def _timeout_handler(signum: int, frame: object) -> None:
        print(f"\nTIMEOUT: exceeded {args.timeout}s limit")
        sys.exit(2)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout)

    with tempfile.TemporaryDirectory(prefix="bench_train_") as _tmp:
        run_profile(
            n_pixels=args.n_pixels,
            n_s2_obs=args.n_s2_obs,
            n_s1_obs=args.n_s1_obs,
            seed=args.seed,
            use_s1=args.use_s1,
            pixel_zscore=args.pixel_zscore,
            run_presence_filter=args.presence_filter,
            dataset_subprocess=args.dataset_subprocess,
            n_shards=args.n_shards,
            assert_rss_gb=args.assert_rss_gb,
            assert_wall_s=args.assert_wall_s,
            out_dir=Path(_tmp),
        )
