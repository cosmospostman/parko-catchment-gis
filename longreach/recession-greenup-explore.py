"""longreach/recession-greenup-explore.py — Diagnostic exploration of
RecessionSensitivitySignal and GreenupTimingSignal at Longreach 8×8 km.

Seven diagnostic stages as specified in research/RECESSION_AND_GREENUP.md:

  Stage 1 — Raw material: NDVI waveform, observation density, smoothing sensitivity
  Stage 2 — Wet-season moisture proxy: NDWI inter-annual variation
  Stage 3 — Recession slopes: per-year class separation and sensitivity scatter
  Stage 4 — Peak DOY estimates: per-year consistency and peak-finding sanity check
  Stage 5 — Riparian proxy case: overlay on recession and greenup figures
  Stage 6 — Feature correlation with existing signals
  Stage 7 — Parameter sensitivity sweep

Class labels
------------
  Presence : top-10% prob_lr in the 8×8 km scene pixel ranking CSV
  Absence  : bottom-10% prob_lr
  Middle 80% excluded from class-labelled figures; retained in continuous-score figures

Riparian proxy
--------------
  Top-10% nir_mean (dry-season mean B08) of pixels that fall within the
  "extension" sub-bbox of the original Longreach location
  (the southern grassland region: lon 145.423948–145.424956, lat -22.767104–-22.764033).
  nir_mean is computed fresh from the raw observations using NirCvSignal.

Usage
-----
    python -m longreach."recession-greenup-explore"           # all stages
    python -m longreach."recession-greenup-explore" --stage 1 # single stage
    python -m longreach."recession-greenup-explore" --fast    # skip stage 7 sweep
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.location import get
from signals import NirCvSignal, QualityParams
from signals._shared import load_and_filter, annual_ndvi_curve, annual_ndvi_curve_chunked
from signals.recession import RecessionSensitivitySignal
from signals.greenup import GreenupTimingSignal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENE_LOC_ID = "longreach-8x8km"
RANKING_CSV  = PROJECT_ROOT / "outputs" / "longreach-8x8" / "longreach_8x8km_pixel_ranking.csv"
OUT_DIR      = PROJECT_ROOT / "research" / "recession-and-greenup" / "longreach-recession-greenup"

# Confirmed infestation sub-bbox from longreach.yaml — used as the Presence class.
# Everything else in the 8×8 km scene is treated as scene background (Absence).
# Assumption: scattered Parkinsonia in the broader scene will disappear into the average;
# the dense infestation strip should still stand out.
INFESTATION_BBOX = [145.423948, -22.764033, 145.424956, -22.761054]  # [lon_min, lat_min, lon_max, lat_max]

# Extension sub-bbox from longreach.yaml (the "grassland"/absence training region)
# used as the pool for deriving the riparian proxy.
EXT_BBOX = [145.423948, -22.767104, 145.424956, -22.764033]  # [lon_min, lat_min, lon_max, lat_max]

# Class colours consistent with existing longreach analyses
CLASS_COLOURS = {
    "Presence":  "#2ca02c",
    "Absence":   "#ff7f0e",
    "Riparian":  "#1f77b4",
}

PRESENCE_COLOUR  = CLASS_COLOURS["Presence"]
ABSENCE_COLOUR   = CLASS_COLOURS["Absence"]
RIPARIAN_COLOUR  = CLASS_COLOURS["Riparian"]

YEARS = list(range(2016, 2022))


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_ranking() -> pd.DataFrame:
    """Load the 8×8 km pixel ranking CSV and label classes by geometry.

    Presence  — pixels within the confirmed infestation sub-bbox (INFESTATION_BBOX).
    Absence   — all other pixels in the 8×8 km scene.

    The infestation strip is a dense, confirmed Parkinsonia patch. Scattered
    Parkinsonia elsewhere in the scene is expected but assumed to be diluted
    by the broader background, so the strip should still stand out.
    """
    ranking = pd.read_csv(RANKING_CSV)
    lon_min, lat_min, lon_max, lat_max = INFESTATION_BBOX
    in_strip = (
        (ranking["lon"] >= lon_min) & (ranking["lon"] <= lon_max) &
        (ranking["lat"] >= lat_min) & (ranking["lat"] <= lat_max)
    )
    ranking["class"] = "Absence"
    ranking.loc[in_strip, "class"] = "Presence"
    log(f"  Pixel ranking loaded: {len(ranking):,} total | "
        f"Presence (infestation bbox): {in_strip.sum():,} | "
        f"Absence (rest of scene): {(~in_strip).sum():,}")
    return ranking


def sorted_parquet_path(loc) -> Path:
    """Return the pixel-sorted parquet path (<id>-by-pixel.parquet)."""
    base = loc.parquet_path()
    return base.with_name(base.stem + "-by-pixel.parquet")


def load_raw_pixels(loc) -> pl.DataFrame:
    """Load the raw 8×8 km observation parquet, selecting only the columns
    consumed by this script (9 of 20) and downcasting float64 → float32.

    Columns kept:
      point_id, lon, lat, date, scl_purity — always needed
      B08  — NirCvSignal (riparian proxy) + NDVI curve
      B04  — NDVI curve
      B03, B11 — NDWI (recession signal, stage 2)

    Downcasting lon/lat/band columns from float64 to float32 halves their
    footprint (~8 GB → ~4 GB for the numeric columns).
    """
    _LOAD_COLS = [
        "point_id", "lon", "lat", "date", "scl_purity",
        "B03", "B04", "B08", "B11",
    ]
    path = loc.parquet_path()
    log(f"  Loading raw pixels from {path} ...")
    df = pl.read_parquet(path, columns=_LOAD_COLS).with_columns([
        pl.col("lon").cast(pl.Float32),
        pl.col("lat").cast(pl.Float32),
        pl.col("B03").cast(pl.Float32),
        pl.col("B04").cast(pl.Float32),
        pl.col("B08").cast(pl.Float32),
        pl.col("B11").cast(pl.Float32),
        pl.col("scl_purity").cast(pl.Float32),
    ])
    log(f"    {len(df):,} observations, {df['point_id'].n_unique():,} pixels")
    return df


def derive_riparian_proxy(raw_df: pl.DataFrame, loc) -> list[str]:
    """Return point_ids for the riparian proxy: top-10% nir_mean within the extension sub-bbox.

    Steps:
    1. Filter raw pixels to the extension bbox (in Polars, before any heavy compute).
    2. Compute dry-season mean B08 (nir_mean) per pixel via NirCvSignal on the small subset.
    3. Return top-10% point_ids by nir_mean.
    """
    ext_lon_min, ext_lat_min, ext_lon_max, ext_lat_max = EXT_BBOX

    # Pre-filter to extension bbox in Polars — avoids running NirCvSignal on 267M rows
    ext_df = raw_df.filter(
        (pl.col("lon") >= ext_lon_min) & (pl.col("lon") <= ext_lon_max) &
        (pl.col("lat") >= ext_lat_min) & (pl.col("lat") <= ext_lat_max)
    )
    n_ext_pixels = ext_df["point_id"].n_unique()
    log(f"  Extension bbox: {n_ext_pixels} pixels, {len(ext_df):,} observations")

    if n_ext_pixels == 0:
        log("  WARNING: No pixels found in extension bbox — riparian proxy will be empty.")
        return []

    nir_sig = NirCvSignal()
    nir_stats = nir_sig.compute(ext_df, loc)

    thresh = nir_stats["nir_mean"].quantile(0.90)
    rip_ids = nir_stats.loc[nir_stats["nir_mean"] >= thresh, "point_id"].tolist()
    log(f"  Riparian proxy: {len(rip_ids)} pixels (top-10% nir_mean in extension bbox, "
        f"threshold = {thresh:.1f})")
    return rip_ids


# ---------------------------------------------------------------------------
# Stage 1 — Raw material
# ---------------------------------------------------------------------------

def stage1(raw_df, ranking: pd.DataFrame, rip_ids: list[str], loc,
           df_filt: pl.DataFrame | None = None) -> None:
    log("\n=== Stage 1: Raw material — NDVI time series, observation density, smoothing ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pres_ids = ranking.loc[ranking["class"] == "Presence", "point_id"].tolist()
    # Cap absence to bottom-10% prob_lr to avoid materialising the full scene
    # (640K absence pixels × ~400 obs each ≈ 267M rows → OOM in to_pandas).
    abs_all  = ranking.loc[ranking["class"] == "Absence"].sort_values("prob_lr", ascending=True)
    abs_ids  = abs_all.iloc[:max(1, len(abs_all) // 10)]["point_id"].tolist()

    # Filtered observations with year/month — pre-filter to relevant point_ids
    # before converting to pandas to avoid materialising the full 267M-row frame.
    q = QualityParams()
    _filt = df_filt if df_filt is not None else load_and_filter(raw_df, q.scl_purity_min)
    all_ids = list(set(pres_ids) | set(abs_ids) | set(rip_ids))
    df = (
        _filt.filter(pl.col("point_id").is_in(all_ids))
        .to_pandas()
    )
    df["ndvi"] = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"])
    df["doy"]  = df["date"].dt.dayofyear

    # ------------------------------------------------------------------
    # Figure 1a — Mean NDVI time series by class (full 2016–2021)
    # ------------------------------------------------------------------
    log("  Figure 1a: mean NDVI time series by class")

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Longreach 8×8 km — Mean daily NDVI by class (2016–2021)", fontsize=12)

    class_spec = [
        ("Presence",  pres_ids,  PRESENCE_COLOUR),
        ("Absence",   abs_ids,   ABSENCE_COLOUR),
        ("Riparian",  rip_ids,   RIPARIAN_COLOUR),
    ]

    for label, ids, colour in class_spec:
        if not ids:
            continue
        sub = df[df["point_id"].isin(ids)].copy()

        # Per-year daily mean
        for yr in YEARS:
            yr_sub = sub[sub["date"].dt.year == yr].copy()
            if yr_sub.empty:
                continue
            daily = yr_sub.groupby("date")["ndvi"].mean()
            ax.plot(daily.index, daily.values, color=colour, alpha=0.18, linewidth=0.8)

        # 6-year mean
        daily_all = sub.groupby("date")["ndvi"].mean()
        ax.plot(daily_all.index, daily_all.values, color=colour, linewidth=1.8, label=label)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Mean NDVI", fontsize=10)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1a_ndvi_timeseries_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s1a_ndvi_timeseries_by_class.png")

    # ------------------------------------------------------------------
    # Figure 1b — Observation density calendar (clean acquisitions per 14-day bin)
    # ------------------------------------------------------------------
    log("  Figure 1b: observation density calendar")

    bins = list(range(1, 366, 14))
    bin_labels = [f"DOY {b}" for b in bins]

    fig, axes = plt.subplots(len(YEARS), 1, figsize=(12, 2.2 * len(YEARS)), sharex=True)
    fig.suptitle("Longreach — Clean acquisition count per 14-day DOY bin, by class", fontsize=11)

    for i, yr in enumerate(YEARS):
        ax = axes[i]
        yr_df = df[df["date"].dt.year == yr].copy()
        for label, ids, colour in class_spec:
            if not ids:
                continue
            sub = yr_df[yr_df["point_id"].isin(ids)].copy()
            # Count unique dates per DOY bin (one obs per pixel per date = 1 acquisition)
            sub["doy_bin"] = pd.cut(sub["doy"], bins=bins + [366], right=False, labels=bins)
            counts = sub.groupby(["doy_bin", "point_id"], observed=False)["date"].nunique().groupby("doy_bin").sum()
            ax.bar(counts.index.astype(float), counts.values,
                   width=12, alpha=0.5, color=colour, label=label if i == 0 else "_nolegend_")
        ax.set_ylabel(str(yr), fontsize=8, rotation=0, labelpad=30, va="center")
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("DOY", fontsize=9)
    handles = [mpatches.Patch(color=CLASS_COLOURS[l], label=l)
               for l, _, _ in class_spec if _]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1b_obs_density_calendar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s1b_obs_density_calendar.png")

    # ------------------------------------------------------------------
    # Figure 1c — Smoothing sensitivity: raw vs. smoothed for sample pixels
    # ------------------------------------------------------------------
    log("  Figure 1c: smoothing sensitivity (raw vs. smoothed at 15/30/45 days)")

    # Pick 3 presence and 3 absence sample pixels, stratified by prob_lr rank
    def sample_pixels(ids, n=3):
        sub = ranking[ranking["point_id"].isin(ids)].sort_values("prob_lr", ascending=False)
        idx = np.linspace(0, len(sub) - 1, n, dtype=int)
        return sub.iloc[idx]["point_id"].tolist()

    sample_pres = sample_pixels(pres_ids, 3)
    sample_abs  = sample_pixels(abs_ids, 3)
    sample_pids = sample_pres + sample_abs
    sample_labels = ["Presence"] * 3 + ["Absence"] * 3
    sample_colours = [PRESENCE_COLOUR] * 3 + [ABSENCE_COLOUR] * 3

    smooth_windows = [15, 30, 45]
    smooth_styles  = ["-", "--", ":"]
    smooth_widths  = [1.4, 1.2, 1.0]

    df_pl = _filt  # reuse already-filtered frame

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False, sharey=False)
    fig.suptitle("Longreach — Raw NDVI vs. rolling-median smoothing at 3 window widths\n"
                 "(dots = raw observations; lines = smoothed)", fontsize=11)

    for pi, (pid, lbl, col) in enumerate(zip(sample_pids, sample_labels, sample_colours)):
        ax = axes[pi // 3, pi % 3]
        pix = df_pl.filter(pl.col("point_id") == pid)
        pix = pix.with_columns([
            ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))).alias("ndvi"),
        ]).sort("date")
        pix_pd = pix.select(["date", "ndvi"]).to_pandas()
        ax.scatter(pix_pd["date"], pix_pd["ndvi"], s=6, color=col, alpha=0.5, zorder=3)

        for sw, ls, lw in zip(smooth_windows, smooth_styles, smooth_widths):
            window_rows = max(3, sw // 5)
            sm = pix.with_columns([
                pl.col("ndvi")
                  .rolling_median(window_size=window_rows, min_samples=1, center=True)
                  .alias("ndvi_sm"),
            ])
            sm_pd = sm.select(["date", "ndvi_sm"]).to_pandas()
            ax.plot(sm_pd["date"], sm_pd["ndvi_sm"],
                    color=col, linestyle=ls, linewidth=lw,
                    label=f"{sw}d", alpha=0.9)

        prob = ranking.loc[ranking["point_id"] == pid, "prob_lr"]
        prob_str = f"prob={prob.values[0]:.3f}" if not prob.empty else ""
        ax.set_title(f"{lbl} — {pid}\n{prob_str}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Date", fontsize=7)
        ax.set_ylabel("NDVI", fontsize=7)
        if pi == 0:
            ax.legend(title="smooth", fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1c_smoothing_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s1c_smoothing_sensitivity.png")


# ---------------------------------------------------------------------------
# Stage 2 — Wet-season moisture proxy
# ---------------------------------------------------------------------------

def stage2(raw_df, ranking: pd.DataFrame, df_filt: pl.DataFrame | None = None) -> None:
    log("\n=== Stage 2: Wet-season moisture proxy — NDWI inter-annual variation ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    q = QualityParams()
    _filt = df_filt if df_filt is not None else load_and_filter(raw_df, q.scl_purity_min)

    # Compute per-pixel per-year peak NDWI entirely in Polars — avoids
    # materialising the full 267M-row frame as pandas.
    wet_months = [12, 1, 2, 3]
    ndwi_peak = (
        _filt
        .filter(pl.col("month").is_in(wet_months))
        .with_columns([
            ((pl.col("B03") - pl.col("B11")) / (pl.col("B03") + pl.col("B11"))).alias("ndwi"),
            pl.when(pl.col("month") == 12)
              .then(pl.col("year") + 1)
              .otherwise(pl.col("year"))
              .alias("wet_year"),
        ])
        .filter(pl.col("wet_year").is_in(YEARS))
        .group_by(["point_id", "wet_year"])
        .agg(pl.col("ndwi").max().alias("ndwi_peak"))
        .to_pandas()
    )

    # ------------------------------------------------------------------
    # Figure 2a — Per-year scene-mean wet-season NDWI distribution (boxplot)
    # ------------------------------------------------------------------
    log("  Figure 2a: per-year NDWI peak distribution (scene-wide boxplot)")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Longreach 8×8 km — Peak wet-season NDWI per pixel, per year\n"
                 "(scene-wide; sufficient inter-annual range → reliable recession sensitivity)",
                 fontsize=10)

    data_by_year = [ndwi_peak[ndwi_peak["wet_year"] == yr]["ndwi_peak"].dropna().values
                    for yr in YEARS]
    bp = ax.boxplot(data_by_year, labels=YEARS, patch_artist=True, medianprops={"linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
    ax.set_xlabel("Wet season year (Dec Y-1 → Mar Y)", fontsize=10)
    ax.set_ylabel("Peak NDWI", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s2a_ndwi_peak_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s2a_ndwi_peak_boxplot.png")

    # ------------------------------------------------------------------
    # Figure 2b — NDWI peak maps per year (6 small panels)
    # ------------------------------------------------------------------
    log("  Figure 2b: NDWI peak spatial maps per year")

    # Merge with coords
    coords = ranking[["point_id", "lon", "lat"]].copy()
    ndwi_map = ndwi_peak.merge(coords, on="point_id", how="left")

    vmin = ndwi_peak["ndwi_peak"].quantile(0.02)
    vmax = ndwi_peak["ndwi_peak"].quantile(0.98)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Longreach 8×8 km — Peak wet-season NDWI per year", fontsize=11)

    for i, yr in enumerate(YEARS):
        ax = axes[i // 3, i % 3]
        yr_data = ndwi_map[ndwi_map["wet_year"] == yr]
        sc = ax.scatter(yr_data["lon"], yr_data["lat"],
                        c=yr_data["ndwi_peak"], cmap="Blues",
                        vmin=vmin, vmax=vmax, s=2, rasterized=True)
        ax.set_title(f"Wet season {yr}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s2b_ndwi_peak_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s2b_ndwi_peak_maps.png")

    # ------------------------------------------------------------------
    # Figure 2c — NDWI inter-annual range (histogram + spatial map)
    # ------------------------------------------------------------------
    log("  Figure 2c: NDWI inter-annual range per pixel")

    ndwi_range = (
        ndwi_peak.groupby("point_id")["ndwi_peak"]
        .agg(["max", "min"])
        .assign(ndwi_range=lambda d: d["max"] - d["min"])
        .reset_index()
    )
    ndwi_range = ndwi_range.merge(coords, on="point_id", how="left")

    low_range_frac = (ndwi_range["ndwi_range"] < 0.05).mean()
    log(f"    Pixels with NDWI range < 0.05: {low_range_frac:.1%} "
        f"(→ recession_sensitivity unreliable for these pixels)")

    fig, (ax_hist, ax_map) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Longreach — NDWI inter-annual range per pixel\n"
                 "(range < 0.05 → recession_sensitivity unreliable)", fontsize=10)

    ax_hist.hist(ndwi_range["ndwi_range"].dropna(), bins=40, color="#4878cf", edgecolor="none")
    ax_hist.axvline(0.05, color="red", linestyle="--", linewidth=1.2, label="0.05 threshold")
    ax_hist.set_xlabel("NDWI range (max − min across years)", fontsize=10)
    ax_hist.set_ylabel("Pixel count", fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.tick_params(labelsize=9)

    sc = ax_map.scatter(ndwi_range["lon"], ndwi_range["lat"],
                        c=ndwi_range["ndwi_range"], cmap="YlOrRd",
                        s=2, rasterized=True)
    ax_map.set_title("Spatial map of NDWI range", fontsize=9)
    ax_map.set_aspect("equal")
    ax_map.tick_params(labelsize=7)
    plt.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s2c_ndwi_range.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s2c_ndwi_range.png")


# ---------------------------------------------------------------------------
# Stage 3 — Recession slopes
# ---------------------------------------------------------------------------

def stage3(rec_signal: RecessionSensitivitySignal, raw_df,
           ranking: pd.DataFrame, rip_ids: list[str], loc,
           df_filt: pl.DataFrame | None = None,
           curve: "pl.DataFrame | Path | None" = None) -> pd.DataFrame:
    """Compute recession features and produce Stage 3 diagnostic figures.

    Returns the per-pixel recession stats DataFrame for reuse in later stages.
    """
    log("\n=== Stage 3: Recession slopes — per-year class separation and sensitivity ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pres_ids = ranking.loc[ranking["class"] == "Presence", "point_id"].tolist()
    abs_ids  = ranking.loc[ranking["class"] == "Absence",  "point_id"].tolist()

    # Compute per-pixel recession features
    p = rec_signal.params
    q = p.quality
    _df = df_filt if df_filt is not None else load_and_filter(raw_df, q.scl_purity_min)
    _curve = curve  # may be None; compute() handles it

    log("  Computing RecessionSensitivitySignal ...")
    rec_stats = rec_signal.compute(raw_df, loc, _df=_df, _curve=_curve)
    log(f"    {len(rec_stats):,} pixels with recession stats")

    # Also build per-pixel-year table for the scatter plots (reuse _df)
    df_pl_ndwi = _df.with_columns([
        ((pl.col("B03") - pl.col("B11")) / (pl.col("B03") + pl.col("B11"))).alias("ndwi"),
        pl.when(pl.col("month") == 12)
          .then(pl.col("year") + 1)
          .otherwise(pl.col("year"))
          .alias("wet_year"),
    ])
    wet_months = [12, 1, 2, 3]
    ndwi_df = (
        df_pl_ndwi.filter(pl.col("month").is_in(wet_months))
        .group_by(["point_id", "wet_year"])
        .agg([
            pl.col("ndwi").max().alias("ndwi_peak"),
            pl.col("ndwi").count().alias("_n"),
        ])
        .filter(pl.col("_n") >= p.min_wet_obs)
        .rename({"wet_year": "year"})
        .drop("_n")
        .to_pandas()
    )

    _curve_for_slopes = _curve if _curve is not None else annual_ndvi_curve(_df, p.smooth_days, q.min_obs_per_year)
    slopes_df = rec_signal._recession_slopes(_curve_for_slopes)
    paired = slopes_df.merge(ndwi_df, on=["point_id", "year"], how="inner")

    # Merge class labels
    paired = paired.merge(
        ranking[["point_id", "class", "prob_lr"]],
        on="point_id", how="left"
    )
    paired["class"] = paired["class"].fillna("middle")

    # ------------------------------------------------------------------
    # Figure 3a — Per-year recession slope distributions by class (strip/violin)
    # ------------------------------------------------------------------
    log("  Figure 3a: per-year recession slope distributions by class")

    fig, axes = plt.subplots(1, len(YEARS), figsize=(14, 5), sharey=True)
    fig.suptitle("Longreach — Per-year recession slope by class\n"
                 "(less negative = more persistent canopy; theory: Presence < Absence)", fontsize=10)

    for i, yr in enumerate(YEARS):
        ax = axes[i]
        yr_data = paired[paired["year"] == yr]
        for label, ids, col in [("Presence", pres_ids, PRESENCE_COLOUR),
                                  ("Absence", abs_ids, ABSENCE_COLOUR)]:
            vals = yr_data[yr_data["point_id"].isin(ids)]["recession_slope"].dropna()
            if vals.empty:
                continue
            # Violin
            parts = ax.violinplot([vals.values], positions=[{"Presence": 0, "Absence": 1}[label]],
                                  widths=0.7, showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(0.6)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

        ax.set_title(str(yr), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pres", "Abs"], fontsize=8)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_ylabel("Recession slope (NDVI/DOY)", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s3a_recession_slope_per_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s3a_recession_slope_per_year.png")

    # ------------------------------------------------------------------
    # Figure 3b — Recession slope vs. NDWI peak, per pixel coloured by prob_lr
    # ------------------------------------------------------------------
    log("  Figure 3b: recession slope vs. NDWI peak, coloured by prob_lr")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Longreach — Recession slope vs. wet-season NDWI peak\n"
                 "(each point = one pixel-year; colour = prob_lr; "
                 "theory: high-prob flat cloud, low-prob negative slope)", fontsize=10)

    # All pixels, all years
    all_pairs = paired[paired["prob_lr"].notna()].copy()
    sc = ax.scatter(all_pairs["ndwi_peak"], all_pairs["recession_slope"],
                    c=all_pairs["prob_lr"], cmap="RdYlGn",
                    vmin=0, vmax=1, s=6, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="prob_lr")
    ax.set_xlabel("Peak wet-season NDWI", fontsize=10)
    ax.set_ylabel("Recession slope (NDVI/DOY)", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s3b_slope_vs_ndwi_prob.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s3b_slope_vs_ndwi_prob.png")

    # ------------------------------------------------------------------
    # Figure 3c — Per-pixel recession sensitivity vs. prob_lr (scatter)
    # ------------------------------------------------------------------
    log("  Figure 3c: recession sensitivity vs. prob_lr")

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Longreach — Recession sensitivity (Pearson r: slope vs NDWI peak) vs. prob_lr\n"
                 "(theory: high prob → near-zero sensitivity; low prob → negative sensitivity)",
                 fontsize=10)

    plot_df = rec_stats.merge(ranking[["point_id", "prob_lr"]], on="point_id", how="left")
    plot_df = plot_df[plot_df["recession_sensitivity"].notna() & plot_df["prob_lr"].notna()]

    if not plot_df.empty:
        sc = ax.scatter(plot_df["prob_lr"], plot_df["recession_sensitivity"],
                        s=8, alpha=0.4, color="#555555", rasterized=True)
        # OLS trend line
        x = plot_df["prob_lr"].values
        y = plot_df["recession_sensitivity"].values
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, m * xr + b, color="red", linewidth=1.5, label=f"OLS slope={m:.3f}")
            r, p = pearsonr(x, y)
            ax.legend(title=f"r={r:.3f} p={p:.2e}", fontsize=9)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("prob_lr (Parkinsonia classifier score)", fontsize=10)
    ax.set_ylabel("Recession sensitivity (Pearson r)", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s3c_sensitivity_vs_prob.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s3c_sensitivity_vs_prob.png")

    # ------------------------------------------------------------------
    # Figure 3d — Spatial map of recession_sensitivity
    # ------------------------------------------------------------------
    log("  Figure 3d: spatial map of recession_sensitivity")

    # rec_stats already contains lon/lat from the signal's compute() method
    map_df = rec_stats.copy()

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Longreach — Recession sensitivity (Pearson r: slope vs NDWI peak)\n"
                 "negative = rain-dependent; near-zero = decoupled", fontsize=10)

    vabs = map_df["recession_sensitivity"].abs().quantile(0.95)
    sc = ax.scatter(map_df["lon"], map_df["lat"],
                    c=map_df["recession_sensitivity"], cmap="RdBu",
                    vmin=-vabs, vmax=vabs, s=3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Recession sensitivity (r)")
    ax.set_xlabel("Lon", fontsize=9)
    ax.set_ylabel("Lat", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s3d_map_recession_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s3d_map_recession_sensitivity.png")

    return rec_stats, paired


# ---------------------------------------------------------------------------
# Stage 4 — Peak DOY estimates
# ---------------------------------------------------------------------------

def stage4(green_signal: GreenupTimingSignal, raw_df,
           ranking: pd.DataFrame, rip_ids: list[str], loc,
           df_filt: pl.DataFrame | None = None,
           curve: "pl.DataFrame | Path | None" = None) -> pd.DataFrame:
    """Compute greenup features and produce Stage 4 diagnostic figures.

    Returns per-pixel greenup stats DataFrame.
    """
    log("\n=== Stage 4: Peak DOY estimates — per-year consistency and peak-finding check ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pres_ids = ranking.loc[ranking["class"] == "Presence", "point_id"].tolist()
    abs_ids  = ranking.loc[ranking["class"] == "Absence",  "point_id"].tolist()

    p = green_signal.params
    q = p.quality
    # When a pre-computed curve Path is provided, df_filt is not needed by
    # compute() (coords come from the curve parquet).  Only load it if the
    # curve is absent and we have a raw_df to load from.
    _curve = curve  # may be None; compute() handles it
    if _curve is not None:
        _df = None  # compute() will read coords from the curve parquet
    elif df_filt is not None:
        _df = df_filt
    else:
        _df = load_and_filter(raw_df, q.scl_purity_min)

    log("  Computing GreenupTimingSignal ...")
    green_stats = green_signal.compute(raw_df, loc, _df=_df, _curve=_curve)
    log(f"    {len(green_stats):,} pixels with greenup stats")

    # Per-year peak DOY table — reuse the same curve
    _curve_for_peak = _curve if _curve is not None else annual_ndvi_curve(_df, p.smooth_days, q.min_obs_per_year)
    per_year = green_signal._peak_doy_per_year(_curve_for_peak)
    per_year = per_year.merge(
        ranking[["point_id", "class", "prob_lr"]],
        on="point_id", how="left"
    )
    per_year["class"] = per_year["class"].fillna("middle")

    # ------------------------------------------------------------------
    # Figure 4a — Per-year peak DOY by class (strip plot)
    # ------------------------------------------------------------------
    log("  Figure 4a: per-year peak DOY by class")

    fig, axes = plt.subplots(1, len(YEARS), figsize=(14, 5), sharey=True)
    fig.suptitle("Longreach — Per-year NDVI peak DOY by class\n"
                 "(reliable years only; theory: Presence consistent earlier DOY)", fontsize=10)

    for i, yr in enumerate(YEARS):
        ax = axes[i]
        yr_data = per_year[(per_year["year"] == yr) & per_year["reliable"]]
        for label, ids, col in [("Presence", pres_ids, PRESENCE_COLOUR),
                                  ("Absence", abs_ids, ABSENCE_COLOUR)]:
            vals = yr_data[yr_data["point_id"].isin(ids)]["peak_doy"].dropna().values
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            xpos = {"Presence": 0, "Absence": 1}[label]
            ax.scatter(np.full(len(vals), xpos) + jitter, vals,
                       s=6, alpha=0.4, color=col, rasterized=True)
            ax.plot([xpos - 0.25, xpos + 0.25],
                    [np.median(vals), np.median(vals)],
                    color="black", linewidth=2)

        ax.set_title(str(yr), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pres", "Abs"], fontsize=8)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_ylabel("Peak DOY", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s4a_peak_doy_per_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s4a_peak_doy_per_year.png")

    # ------------------------------------------------------------------
    # Figure 4b — Smoothed NDVI with annotated peak DOY for 6 sample pixels
    # ------------------------------------------------------------------
    log("  Figure 4b: smoothed NDVI curves with identified peaks annotated")

    def sample_pixels(ids, n=3):
        sub = ranking[ranking["point_id"].isin(ids)].sort_values("prob_lr", ascending=False)
        idx = np.linspace(0, len(sub) - 1, n, dtype=int)
        return sub.iloc[idx]["point_id"].tolist()

    sample_pids = sample_pixels(pres_ids, 3) + sample_pixels(abs_ids, 3)
    sample_labels = ["Presence"] * 3 + ["Absence"] * 3
    sample_colours = [PRESENCE_COLOUR] * 3 + [ABSENCE_COLOUR] * 3

    # Load only the 6 sample pixels' curve rows for the annotation plot —
    # avoids materialising the full 267M-row curve.
    if isinstance(_curve, Path):
        curve_pd = (
            pl.scan_parquet(_curve)
            .filter(pl.col("point_id").is_in(sample_pids))
            .collect()
            .to_pandas()
        )
    else:
        curve_pd = _curve[_curve["point_id"].isin(sample_pids)].to_pandas() if isinstance(_curve, pl.DataFrame) else _curve[_curve["point_id"].isin(sample_pids)]
    year_colours = plt.cm.tab10(np.linspace(0, 0.9, len(YEARS)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Longreach — Smoothed NDVI with identified peak DOY (vertical lines)\n"
                 "Each year in a different colour; unreliable years in grey", fontsize=10)

    for pi, (pid, lbl, col) in enumerate(zip(sample_pids, sample_labels, sample_colours)):
        ax = axes[pi // 3, pi % 3]
        pix_curve = curve_pd[curve_pd["point_id"] == pid].sort_values("date")
        pix_peaks = per_year[(per_year["point_id"] == pid)]

        for j, yr in enumerate(YEARS):
            yr_curve = pix_curve[pix_curve["year"] == yr]
            if yr_curve.empty:
                continue
            yr_col = year_colours[j]
            ax.plot(yr_curve["doy"], yr_curve["ndvi_smooth"],
                    color=yr_col, linewidth=1.2, alpha=0.85, label=str(yr))
            # Annotate peak
            yr_pk = pix_peaks[pix_peaks["year"] == yr]
            if not yr_pk.empty and yr_pk.iloc[0]["reliable"]:
                pk_doy = yr_pk.iloc[0]["peak_doy"]
                ax.axvline(pk_doy, color=yr_col, linewidth=1.0, linestyle="--", alpha=0.7)
            elif not yr_pk.empty:
                pk_doy = yr_pk.iloc[0]["peak_doy"]
                if not np.isnan(pk_doy):
                    ax.axvline(pk_doy, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)

        prob = ranking.loc[ranking["point_id"] == pid, "prob_lr"]
        prob_str = f"prob={prob.values[0]:.3f}" if not prob.empty else ""
        ax.set_title(f"{lbl} — {pid}\n{prob_str}", fontsize=8)
        ax.set_xlabel("DOY", fontsize=7)
        ax.set_ylabel("NDVI (smoothed)", fontsize=7)
        ax.tick_params(labelsize=7)
        if pi == 0:
            ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s4b_peak_annotation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s4b_peak_annotation.png")

    # ------------------------------------------------------------------
    # Figure 4c — Peak DOY SD across years: histogram by class
    # ------------------------------------------------------------------
    log("  Figure 4c: per-pixel peak DOY SD across years, by class")

    pix_sd = (
        per_year[per_year["reliable"]]
        .groupby("point_id")["peak_doy"]
        .std()
        .reset_index()
        .rename(columns={"peak_doy": "peak_doy_sd"})
    )
    pix_sd = pix_sd.merge(ranking[["point_id", "class"]], on="point_id", how="left")
    pix_sd["class"] = pix_sd["class"].fillna("middle")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Longreach — Per-pixel peak DOY standard deviation across years\n"
                 "(lower = more consistent timing; theory: Presence < Absence)", fontsize=10)

    for label, col in [("Presence", PRESENCE_COLOUR), ("Absence", ABSENCE_COLOUR)]:
        vals = pix_sd[pix_sd["class"] == label]["peak_doy_sd"].dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=30, alpha=0.55, color=col, label=f"{label} (med={vals.median():.1f}d)",
                edgecolor="none")

    ax.axvline(14, color="grey", linewidth=1, linestyle="--", label="14-day threshold")
    ax.axvline(30, color="grey", linewidth=1, linestyle=":",  label="30-day threshold")
    ax.set_xlabel("Peak DOY SD (days)", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)

    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s4c_peak_doy_sd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s4c_peak_doy_sd.png")

    # ------------------------------------------------------------------
    # Figure 4d — Spatial map of mean peak_doy
    # ------------------------------------------------------------------
    log("  Figure 4d: spatial map of peak_doy")

    # green_stats already contains lon/lat from the signal's compute() method
    # Sort so extreme late-peaking pixels are drawn last (on top of the majority).
    map_df = green_stats.dropna(subset=["peak_doy"]).sort_values("peak_doy")

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Longreach — Mean annual NDVI peak DOY\n"
                 "(earlier = earlier wet-season flush)", fontsize=10)

    sc = ax.scatter(map_df["lon"], map_df["lat"],
                    c=map_df["peak_doy"], cmap="plasma",
                    s=3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Peak DOY")
    ax.set_xlabel("Lon", fontsize=9)
    ax.set_ylabel("Lat", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s4d_map_peak_doy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s4d_map_peak_doy.png")

    # ------------------------------------------------------------------
    # Figure 4e — Peak DOY histogram: top-20% vs bottom-40% prob_lr
    # ------------------------------------------------------------------
    log("  Figure 4e: peak DOY histogram by prob_lr tier")

    doy_df = green_stats[["point_id", "peak_doy"]].dropna().merge(
        ranking[["point_id", "prob_lr"]], on="point_id", how="left"
    )
    q20 = doy_df["prob_lr"].quantile(0.80)
    q40 = doy_df["prob_lr"].quantile(0.40)

    top20  = doy_df[doy_df["prob_lr"] >= q20]["peak_doy"]
    bot40  = doy_df[doy_df["prob_lr"] <= q40]["peak_doy"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)
    fig.suptitle(
        "Longreach — Peak DOY distribution by Parkinsonia probability tier\n"
        f"Top 20% prob_lr (≥ {q20:.2f})  vs  Bottom 40% prob_lr (≤ {q40:.2f})",
        fontsize=10,
    )

    bins_full = range(0, 366, 7)   # weekly buckets, full year
    bins_tail = range(200, 366, 7) # weekly buckets, tail window

    axes[0].hist(top20, bins=bins_full, color=PRESENCE_COLOUR, alpha=0.8, edgecolor="none")
    axes[0].set_ylabel("Pixel count", fontsize=9)
    axes[0].set_title(f"Top 20% Parkinsonia probability  (n={len(top20):,})", fontsize=9)
    axes[0].tick_params(labelsize=8)

    axes[1].hist(bot40, bins=bins_full, color=ABSENCE_COLOUR, alpha=0.8, edgecolor="none")
    axes[1].set_ylabel("Pixel count", fontsize=9)
    axes[1].set_title(f"Bottom 40% Parkinsonia probability  (n={len(bot40):,})", fontsize=9)
    axes[1].tick_params(labelsize=8)

    # Tail zoom: DOY 200–365, both tiers overlaid
    top20_tail = top20[top20 >= 200]
    bot40_tail = bot40[bot40 >= 200]
    axes[2].hist(bot40_tail, bins=bins_tail, color=ABSENCE_COLOUR,  alpha=0.6,
                 edgecolor="none", label=f"Bottom 40% (n={len(bot40_tail):,})")
    axes[2].hist(top20_tail, bins=bins_tail, color=PRESENCE_COLOUR, alpha=0.6,
                 edgecolor="none", label=f"Top 20% (n={len(top20_tail):,})")
    axes[2].set_ylabel("Pixel count", fontsize=9)
    axes[2].set_title("Tail zoom: DOY 200–365", fontsize=9)
    axes[2].set_xlabel("Mean annual peak DOY", fontsize=9)
    axes[2].tick_params(labelsize=8)
    axes[2].legend(fontsize=8)

    month_refs = [(91, "Apr"), (152, "Jun"), (244, "Sep"), (305, "Nov")]
    for ax in axes[:2]:
        for doy, lbl in month_refs:
            ax.axvline(doy, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.text(doy + 2, 0.95, lbl, fontsize=7, color="grey", va="top",
                    transform=ax.get_xaxis_transform())

    for doy, lbl in month_refs:
        if doy >= 200:
            axes[2].axvline(doy, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
            axes[2].text(doy + 2, 0.95, lbl, fontsize=7, color="grey", va="top",
                         transform=axes[2].get_xaxis_transform())

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s4e_peak_doy_by_prob_tier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s4e_peak_doy_by_prob_tier.png")

    return green_stats, per_year


# ---------------------------------------------------------------------------
# Stage 5 — Riparian proxy case
# ---------------------------------------------------------------------------

def stage5(rec_stats: pd.DataFrame, paired: pd.DataFrame,
           green_stats: pd.DataFrame, per_year_green: pd.DataFrame,
           raw_df, ranking: pd.DataFrame, rip_ids: list[str],
           df_filt: pl.DataFrame | None = None) -> None:
    log("\n=== Stage 5: Riparian proxy case ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not rip_ids:
        log("  No riparian proxy pixels found — skipping Stage 5.")
        return

    pres_ids = ranking.loc[ranking["class"] == "Presence", "point_id"].tolist()
    # Cap absence to bottom-10% prob_lr (same logic as stage1) to avoid OOM.
    abs_all  = ranking.loc[ranking["class"] == "Absence"].sort_values("prob_lr", ascending=True)
    abs_ids  = abs_all.iloc[:max(1, len(abs_all) // 10)]["point_id"].tolist()

    # ------------------------------------------------------------------
    # Figure 5a — Recession slope vs. NDWI peak: overlay riparian
    # ------------------------------------------------------------------
    log("  Figure 5a: recession slope vs. NDWI peak with riparian overlay")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Longreach — Recession slope vs. NDWI peak: three classes\n"
                 "(riparian proxy overlaid on presence/absence)", fontsize=10)

    for label, ids, col, zord in [
        ("Presence", pres_ids, PRESENCE_COLOUR, 2),
        ("Absence",  abs_ids,  ABSENCE_COLOUR,  2),
        ("Riparian", rip_ids,  RIPARIAN_COLOUR, 4),
    ]:
        sub = paired[paired["point_id"].isin(ids)]
        if sub.empty:
            continue
        ax.scatter(sub["ndwi_peak"], sub["recession_slope"],
                   s=12 if label == "Riparian" else 6,
                   alpha=0.7 if label == "Riparian" else 0.3,
                   color=col, label=label, zorder=zord, rasterized=True)

    ax.set_xlabel("Peak wet-season NDWI", fontsize=10)
    ax.set_ylabel("Recession slope (NDVI/DOY)", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s5a_riparian_recession_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s5a_riparian_recession_overlay.png")

    # ------------------------------------------------------------------
    # Figure 5b — Riparian NDVI time series alongside class means
    # ------------------------------------------------------------------
    log("  Figure 5b: riparian proxy NDVI time series vs. class means")

    q = QualityParams()
    _filt = df_filt if df_filt is not None else load_and_filter(raw_df, q.scl_purity_min)
    all_ids_s5 = list(set(pres_ids) | set(abs_ids) | set(rip_ids))
    df = (
        _filt.filter(pl.col("point_id").is_in(all_ids_s5))
        .to_pandas()
    )
    df["ndvi"] = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"])

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Longreach — NDVI time series: Presence, Absence, and Riparian proxy means\n"
                 "(riparian proxy = top-10% NIR mean in extension sub-bbox)", fontsize=10)

    for label, ids, col in [("Presence", pres_ids, PRESENCE_COLOUR),
                              ("Absence",  abs_ids,  ABSENCE_COLOUR),
                              ("Riparian", rip_ids,  RIPARIAN_COLOUR)]:
        if not ids:
            continue
        sub = df[df["point_id"].isin(ids)]
        daily = sub.groupby("date")["ndvi"].mean()
        ax.plot(daily.index, daily.values, color=col, linewidth=1.8, label=label)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Mean NDVI", fontsize=10)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s5b_riparian_ndvi_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s5b_riparian_ndvi_timeseries.png")

    # ------------------------------------------------------------------
    # Numeric summary: riparian stats vs. presence/absence
    # ------------------------------------------------------------------
    log("  Riparian proxy summary statistics:")
    for feat in ["recession_slope", "recession_slope_cv", "recession_sensitivity"]:
        rip_vals = rec_stats[rec_stats["point_id"].isin(rip_ids)][feat].dropna()
        pres_vals = rec_stats[rec_stats["point_id"].isin(pres_ids)][feat].dropna()
        abs_vals  = rec_stats[rec_stats["point_id"].isin(abs_ids)][feat].dropna()
        log(f"    {feat:28s}  Riparian med={rip_vals.median():.4f}  "
            f"Presence med={pres_vals.median():.4f}  "
            f"Absence med={abs_vals.median():.4f}")


# ---------------------------------------------------------------------------
# Stage 6 — Feature correlation with existing signals
# ---------------------------------------------------------------------------

def stage6(rec_stats: pd.DataFrame, green_stats: pd.DataFrame,
           ranking: pd.DataFrame) -> None:
    log("\n=== Stage 6: Feature correlation with existing signals ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    new_feats = ["recession_slope", "recession_slope_cv", "recession_sensitivity",
                 "peak_doy", "peak_doy_cv"]
    old_feats = ["nir_cv", "rec_p", "re_p10"]

    # Merge all features
    combined = ranking[["point_id", "prob_lr"] + old_feats].copy()
    combined = combined.merge(
        rec_stats[["point_id"] + [f for f in new_feats if f in rec_stats.columns]],
        on="point_id", how="left"
    )
    combined = combined.merge(
        green_stats[["point_id"] + [f for f in new_feats if f in green_stats.columns]],
        on="point_id", how="left"
    )

    all_feats = old_feats + [f for f in new_feats if f in combined.columns]

    # ------------------------------------------------------------------
    # Correlation matrix table
    # ------------------------------------------------------------------
    log("  Pearson r: new features vs. existing features")
    corr_data = []
    for nf in new_feats:
        if nf not in combined.columns:
            continue
        row = {"feature": nf}
        for ef in old_feats + ["prob_lr"]:
            sub = combined[[nf, ef]].dropna()
            if len(sub) >= 10:
                r, _ = pearsonr(sub[nf], sub[ef])
                row[ef] = r
            else:
                row[ef] = float("nan")
        corr_data.append(row)

    corr_df = pd.DataFrame(corr_data).set_index("feature")
    log(corr_df.to_string(float_format=lambda x: f"{x:.3f}"))
    corr_df.to_csv(OUT_DIR / "s6_feature_correlation_table.csv", float_format="%.4f")
    log("    Saved: s6_feature_correlation_table.csv")

    # ------------------------------------------------------------------
    # Figure 6 — Heatmap of correlation matrix
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, max(4, len(new_feats))))
    fig.suptitle("Longreach — Pearson r: new signals vs. existing signals + prob_lr\n"
                 "|r| > 0.8 → likely redundant; |r| < 0.4 → new information", fontsize=10)

    cmat = corr_df.values.astype(float)
    im = ax.imshow(cmat, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(list(corr_df.columns), fontsize=9)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(list(corr_df.index), fontsize=9)
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            v = cmat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < 0.6 else "white")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s6_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s6_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # Figure 6b — 2D scatter: each new feature vs. its most-correlated existing feature
    # ------------------------------------------------------------------
    log("  Figure 6b: new feature vs. most-correlated existing feature, coloured by prob_lr")

    plot_pairs = []
    for nf in new_feats:
        if nf not in combined.columns:
            continue
        r_vals = {ef: abs(corr_df.loc[nf, ef]) if ef in corr_df.columns else 0.0
                  for ef in old_feats}
        best_ef = max(r_vals, key=r_vals.get)
        plot_pairs.append((nf, best_ef, r_vals[best_ef]))

    if plot_pairs:
        ncols = min(3, len(plot_pairs))
        nrows = (len(plot_pairs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        fig.suptitle("New feature vs. most-correlated existing feature (colour = prob_lr)", fontsize=10)

        for idx, (nf, ef, r) in enumerate(plot_pairs):
            ax = axes[idx // ncols, idx % ncols]
            sub = combined[[nf, ef, "prob_lr"]].dropna()
            if not sub.empty:
                sc = ax.scatter(sub[ef], sub[nf],
                                c=sub["prob_lr"], cmap="RdYlGn", vmin=0, vmax=1,
                                s=6, alpha=0.4, rasterized=True)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="prob_lr")
            ax.set_xlabel(ef, fontsize=9)
            ax.set_ylabel(nf, fontsize=9)
            ax.set_title(f"{nf} vs. {ef}  (r={r:.2f})", fontsize=9)
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for idx in range(len(plot_pairs), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        plt.tight_layout()
        fig.savefig(OUT_DIR / "s6b_new_vs_existing_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("    Saved: s6b_new_vs_existing_scatter.png")


# ---------------------------------------------------------------------------
# Stage 7 — Parameter sensitivity
# ---------------------------------------------------------------------------

def stage7(raw_df, ranking: pd.DataFrame,
           df_filt: pl.DataFrame | None = None,
           curve_30: "pl.DataFrame | Path | None" = None,
           by_pixel_path: Path | None = None) -> None:
    log("\n=== Stage 7: Parameter sensitivity sweep ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pres_ids = ranking.loc[ranking["class"] == "Presence", "point_id"].tolist()
    abs_ids  = ranking.loc[ranking["class"] == "Absence",  "point_id"].tolist()
    loc = get(SCENE_LOC_ID)

    # ------------------------------------------------------------------
    # 7a — Recession window boundaries
    # ------------------------------------------------------------------
    log("  7a: Recession window sweep (start_month × end_month)")

    start_months = [3, 4, 5]
    end_months   = [8, 9, 10]
    rec_sep_results = []

    # Pre-resolve df/curve for 7a — all sweeps use default smooth_days=30.
    # Load df_filt internally so the caller can have cleared its reference
    # before calling stage7, keeping peak RAM lower.
    log("  Loading filtered frame for 7a ...")
    _raw_7a = load_raw_pixels(loc)
    _df_7a = load_and_filter(_raw_7a, QualityParams().scl_purity_min).drop("scl_purity")
    del _raw_7a
    gc.collect()
    _curve_7a = curve_30  # may be None; compute() will build it once if so

    for sm in start_months:
        for em in end_months:
            if sm >= em:
                continue
            params = RecessionSensitivitySignal.Params(
                recession_start_month=sm,
                recession_end_month=em,
            )
            sig = RecessionSensitivitySignal(params)
            try:
                stats = sig.compute(raw_df, loc, _df=_df_7a, _curve=_curve_7a)
                pres_vals = stats[stats["point_id"].isin(pres_ids)]["recession_slope"].dropna()
                abs_vals  = stats[stats["point_id"].isin(abs_ids)]["recession_slope"].dropna()
                if pres_vals.empty or abs_vals.empty:
                    sep = float("nan")
                else:
                    sep = abs(pres_vals.median() - abs_vals.median())
                rec_sep_results.append({
                    "start_month": sm, "end_month": em, "separation": sep,
                    "pres_median": pres_vals.median() if not pres_vals.empty else float("nan"),
                    "abs_median":  abs_vals.median()  if not abs_vals.empty  else float("nan"),
                })
                log(f"    start={sm} end={em}: |median separation| = {sep:.6f}")
            except Exception as exc:
                log(f"    start={sm} end={em}: ERROR — {exc}")

    # 7a done — free df_filt / _df_7a before 7b's streaming collect.
    del _df_7a
    gc.collect()

    if rec_sep_results:
        rec_sweep_df = pd.DataFrame(rec_sep_results)
        rec_sweep_df.to_csv(OUT_DIR / "s7a_recession_window_sweep.csv", index=False)
        log("    Saved: s7a_recession_window_sweep.csv")

        # Heatmap
        import itertools
        pivot = rec_sweep_df.pivot(index="start_month", columns="end_month", values="separation")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle("Recession slope class separation by window choice\n"
                     "(higher = better class separation; April–Sept = default)", fontsize=10)
        im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"end={m}" for m in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"start={m}" for m in pivot.index], fontsize=9)
        for i, j in itertools.product(range(len(pivot.index)), range(len(pivot.columns))):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.5f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|median separation|")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "s7a_recession_window_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("    Saved: s7a_recession_window_heatmap.png")

    # ------------------------------------------------------------------
    # 7b — Smoothing window effect on peak DOY SD
    # ------------------------------------------------------------------
    log("  7b: Smoothing window sweep for peak DOY SD")

    smooth_windows = [15, 21, 30, 45]
    smooth_results = []

    for sw in smooth_windows:
        params = GreenupTimingSignal.Params(smooth_days=sw)
        sig = GreenupTimingSignal(params)
        _tmp_curve = None
        try:
            if sw == 30 and curve_30 is not None:
                curve = curve_30
                _tmp_curve = None
            elif by_pixel_path is not None:
                _tmp_curve = OUT_DIR / f"_tmp_curve_{sw}d.parquet"
                curve = annual_ndvi_curve_chunked(
                    by_pixel_path, out_path=_tmp_curve,
                    smooth_days=sw, min_obs_per_year=params.quality.min_obs_per_year,
                    scl_purity_min=0.0,
                )
            else:
                curve = annual_ndvi_curve(_df_7a, sw, params.quality.min_obs_per_year)
                _tmp_curve = None
            per_year = sig._peak_doy_per_year(curve)
            reliable = per_year[per_year["reliable"]]
            pix_sd = reliable.groupby("point_id")["peak_doy"].std().reset_index()
            pix_sd.columns = ["point_id", "peak_doy_sd"]

            for label, ids in [("Presence", pres_ids), ("Absence", abs_ids)]:
                cls_sd = pix_sd[pix_sd["point_id"].isin(ids)]["peak_doy_sd"].dropna()
                smooth_results.append({
                    "smooth_days": sw,
                    "class": label,
                    "median_sd": cls_sd.median() if not cls_sd.empty else float("nan"),
                    "p25_sd": cls_sd.quantile(0.25) if not cls_sd.empty else float("nan"),
                    "p75_sd": cls_sd.quantile(0.75) if not cls_sd.empty else float("nan"),
                })
                log(f"    smooth={sw}d  {label:8s}: median peak-DOY SD = "
                    f"{cls_sd.median():.1f} days")
        except Exception as exc:
            log(f"    smooth={sw}d: ERROR — {exc}")
        finally:
            if _tmp_curve is not None and _tmp_curve.exists():
                _tmp_curve.unlink()

    if smooth_results:
        smooth_df = pd.DataFrame(smooth_results)
        smooth_df.to_csv(OUT_DIR / "s7b_smooth_window_sweep.csv", index=False)
        log("    Saved: s7b_smooth_window_sweep.csv")

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle("Peak DOY SD vs. smoothing window width\n"
                     "(lower SD = more consistent timing estimates)", fontsize=10)
        for label, col in [("Presence", PRESENCE_COLOUR), ("Absence", ABSENCE_COLOUR)]:
            sub = smooth_df[smooth_df["class"] == label]
            ax.plot(sub["smooth_days"], sub["median_sd"], color=col,
                    marker="o", linewidth=1.5, label=label)
            ax.fill_between(sub["smooth_days"], sub["p25_sd"], sub["p75_sd"],
                            color=col, alpha=0.2)
        ax.set_xlabel("Smoothing window (days)", fontsize=10)
        ax.set_ylabel("Median peak-DOY SD (days)", fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "s7b_smooth_window_peak_doy_sd.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("    Saved: s7b_smooth_window_peak_doy_sd.png")

    # ------------------------------------------------------------------
    # 7c — Minimum wet-season observations threshold
    # ------------------------------------------------------------------
    log("  7c: min_wet_obs threshold sweep for reliable pixel-year fraction")

    min_wet_obs_vals = [3, 5, 8]
    mwo_results = []

    loc = get(SCENE_LOC_ID)
    base_sig = GreenupTimingSignal()
    # curve_30 is always provided via by_pixel_path in the normal run path;
    # _df_7a has been deleted above so we cannot fall back to it here.
    if curve_30 is None:
        raise RuntimeError("stage7 7c requires curve_30 — no df_filt available after 7a cleanup")
    curve_base = curve_30

    for mwo in min_wet_obs_vals:
        params = GreenupTimingSignal.Params(min_wet_obs=mwo)
        sig = GreenupTimingSignal(params)
        per_year = sig._peak_doy_per_year(curve_base)
        n_total    = len(per_year)
        n_reliable = per_year["reliable"].sum()
        frac = n_reliable / n_total if n_total > 0 else float("nan")
        mwo_results.append({"min_wet_obs": mwo, "n_total": n_total,
                            "n_reliable": n_reliable, "frac_reliable": frac})
        log(f"    min_wet_obs={mwo}: {n_reliable:,}/{n_total:,} pixel-years reliable "
            f"({frac:.1%})")

    if mwo_results:
        pd.DataFrame(mwo_results).to_csv(OUT_DIR / "s7c_min_wet_obs_sweep.csv", index=False)
        log("    Saved: s7c_min_wet_obs_sweep.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> pl.DataFrame | None:
    """Return a cached Polars DataFrame, or None if the file does not exist."""
    if path.exists():
        log(f"  Cache hit: {path.name}")
        return pl.read_parquet(path)
    return None


def _save_cache(df: pl.DataFrame | pd.DataFrame, path: Path) -> None:
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    df.write_parquet(path)
    log(f"  Cache written: {path.name}")


def run(stages: list[int] | None = None, fast: bool = False,
        no_cache: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stages = stages is None
    def run_stage(n):
        return all_stages or n in stages

    log("Loading scene location and data ...")
    loc = get(SCENE_LOC_ID)
    ranking = load_ranking()

    _q = QualityParams()
    _curve_cache = OUT_DIR / "_cache_ndvi_curve.parquet"
    _by_pixel    = sorted_parquet_path(loc)

    # --- NDVI curve ----------------------------------------------------------
    # annual_ndvi_curve_chunked writes the curve to _curve_cache and returns
    # the path.  Signals and stages receive the Path and use scan_parquet with
    # streaming — the 267M-row curve is never loaded fully into RAM.
    log("Pre-computing NDVI curve ...")
    if not no_cache and _curve_cache.exists():
        curve_30 = _curve_cache
        log(f"  NDVI curve loaded from cache: {_curve_cache.name}")
    else:
        log(f"  Computing annual_ndvi_curve_chunked from {_by_pixel.name} ...")
        curve_30 = annual_ndvi_curve_chunked(
            _by_pixel, out_path=_curve_cache, smooth_days=30,
            min_obs_per_year=_q.min_obs_per_year,
            scl_purity_min=0.0,  # parquet is pre-filtered; all scl_purity == 1.0
        )
        log(f"  NDVI curve written: {_curve_cache.name}")

    # --- Shared filtered frame -----------------------------------------------
    # Loaded after the curve is cached. load_raw_pixels already downcasts
    # float64 → float32, so df_filt settles at ~13 GB rather than ~22 GB.
    log("Loading shared filtered frame ...")
    raw_df  = load_raw_pixels(loc)
    df_filt = load_and_filter(raw_df, _q.scl_purity_min)
    del raw_df
    log(f"  load_and_filter done: {len(df_filt):,} rows")

    # --- Riparian proxy ------------------------------------------------------
    # Derived from df_filt while scl_purity is still present (NirCvSignal needs
    # it internally).  Drop scl_purity afterwards — it is not used by any stage.
    log("Deriving riparian proxy pixels ...")
    rip_ids = derive_riparian_proxy(df_filt, loc)
    df_filt = df_filt.drop("scl_purity")

    # Instantiate signals with defaults
    rec_signal   = RecessionSensitivitySignal()
    green_signal = GreenupTimingSignal()

    _rec_cache        = OUT_DIR / "_cache_rec_stats.parquet"
    _paired_cache     = OUT_DIR / "_cache_paired.parquet"
    _green_cache      = OUT_DIR / "_cache_green_stats.parquet"
    _per_year_g_cache = OUT_DIR / "_cache_per_year_green.parquet"

    def _require_rec():
        nonlocal rec_stats, paired
        if rec_stats is not None and paired is not None:
            return
        log("  Stage requires recession stats — computing ...")
        rec_stats, paired = stage3(rec_signal, None, ranking, rip_ids, loc,
                                   df_filt=df_filt, curve=curve_30)
        _save_cache(rec_stats, _rec_cache)
        _save_cache(paired, _paired_cache)

    def _require_green():
        nonlocal green_stats, per_year_green, df_filt
        if green_stats is not None and per_year_green is not None:
            return
        log("  Stage requires greenup stats — computing ...")
        # Stage 4 only needs the curve Path — free df_filt before the
        # streaming collect to avoid OOM (13 GB frame vs 267M-row curve).
        # We must drop ALL references so the GC can reclaim the memory.
        # Reload df_filt afterward if any later stage still needs it.
        _need_df_after = df_filt is not None
        df_filt = None
        gc.collect()
        green_stats, per_year_green = stage4(green_signal, None, ranking, rip_ids, loc,
                                             df_filt=None, curve=curve_30)
        _save_cache(green_stats, _green_cache)
        _save_cache(per_year_green, _per_year_g_cache)
        if _need_df_after:
            log("  Reloading shared filtered frame after greenup compute ...")
            _raw = load_raw_pixels(loc)
            df_filt = load_and_filter(_raw, _q.scl_purity_min).drop("scl_purity")
            del _raw
            gc.collect()

    rec_stats = paired = green_stats = per_year_green = None

    # Pre-load signal caches if available (skipped in stage 7 which varies params)
    if not no_cache:
        if (cached := _load_cache(_rec_cache)) is not None:
            rec_stats = cached.to_pandas()
        if (cached := _load_cache(_paired_cache)) is not None:
            paired = cached.to_pandas()
        if (cached := _load_cache(_green_cache)) is not None:
            green_stats = cached.to_pandas()
        if (cached := _load_cache(_per_year_g_cache)) is not None:
            per_year_green = cached.to_pandas()

    if run_stage(1):
        stage1(None, ranking, rip_ids, loc, df_filt=df_filt)

    if run_stage(2):
        stage2(None, ranking, df_filt=df_filt)

    if run_stage(3):
        rec_stats, paired = stage3(rec_signal, None, ranking, rip_ids, loc,
                                   df_filt=df_filt, curve=curve_30)
        _save_cache(rec_stats, _rec_cache)
        _save_cache(paired, _paired_cache)

    if run_stage(4):
        # Stage 4 only needs the curve Path (coords embedded) — df_filt not
        # required.  Drop ALL references before the streaming collect so the
        # OOM-causing 13 GB frame is reclaimed.  Reload afterward if needed.
        _need_df_after_4 = any(run_stage(s) for s in [5, 7])
        del df_filt
        gc.collect()

        green_stats, per_year_green = stage4(green_signal, None, ranking, rip_ids, loc,
                                             df_filt=None, curve=curve_30)
        _save_cache(green_stats, _green_cache)
        _save_cache(per_year_green, _per_year_g_cache)

        if _need_df_after_4:
            log("  Reloading shared filtered frame after greenup compute ...")
            _raw = load_raw_pixels(loc)
            df_filt = load_and_filter(_raw, _q.scl_purity_min).drop("scl_purity")
            del _raw
            gc.collect()
        else:
            df_filt = None  # type: ignore[assignment]

    if run_stage(5):
        _require_rec()
        _require_green()
        stage5(rec_stats, paired, green_stats, per_year_green, None, ranking, rip_ids,
               df_filt=df_filt)

    if run_stage(6):
        _require_rec()
        _require_green()
        stage6(rec_stats, green_stats, ranking)

    if run_stage(7) and not fast:
        # stage7 loads df_filt internally for 7a and frees it before 7b.
        # Don't pass df_filt from here — keeping it alive in this frame
        # would defeat the memory-freeing inside stage7.
        df_filt = None  # type: ignore[assignment]
        gc.collect()
        stage7(None, ranking, df_filt=None, curve_30=curve_30, by_pixel_path=_by_pixel)
    elif run_stage(7) and fast:
        log("\n=== Stage 7: skipped (--fast) ===")

    log(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recession + greenup signal diagnostic exploration at Longreach 8×8 km"
    )
    parser.add_argument(
        "--stage", type=int, nargs="+", metavar="N",
        help="Run only the specified stage(s) (1–7). Omit to run all.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip Stage 7 parameter sweep (slower, runs multiple signal computations).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached intermediates and recompute from raw parquet.",
    )
    args = parser.parse_args()
    run(stages=args.stage, fast=args.fast, no_cache=args.no_cache)
