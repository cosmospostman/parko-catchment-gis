"""research/ndvi-integral/longreach-integral.py — NDVI integral investigation at Longreach 8×8 km.

Four diagnostic stages corresponding to the hypotheses in HYPOTHESIS.md:

  Stage 1 — Compute and map
             Compute NdviIntegralSignal from the cached NDVI curve.
             Produce a spatial map and class histograms (infestation bbox vs
             grassland bbox — same controlled labels as green-up.py).

  Stage 2 — Correlation analysis (H2)
             Pearson r between ndvi_integral and [rec_p, nir_cv, re_p10, prob_lr].
             Residual correlation after regressing out rec_p.
             Tests whether ndvi_integral carries information beyond existing features.

  Stage 3 — Riparian proxy check (H3)
             Strip plot of ndvi_integral for presence / absence / riparian proxy.
             H3 predicts: riparian median < presence median despite similar rec_p,
             because their greenness is concentrated in a shorter wet-season window.

  Stage 4 — Smoothing sensitivity (H4 / Step 5 from hypothesis)
             Repeat Stage 1 class separation for smooth_days in [15, 30, 45].
             A feature robust across window choices is more trustworthy.

Class labels
------------
  Presence : pixels within the confirmed infestation sub-bbox (INFESTATION_BBOX)
  Absence  : pixels within the grassland sub-bbox only (GRASSLAND_BBOX)
  Middle   : all other pixels — excluded from class comparisons

  This matches the controlled labelling scheme from green-up.py to keep
  class definitions consistent across investigations.

Riparian proxy
--------------
  Derived dynamically: top-10% dry-season NIR mean within EXT_BBOX (the
  grassland sub-bbox), same algorithm as recession-greenup-explore.py.
  Requires loading the raw pixel parquet — done once and shared with
  derive_riparian_proxy().

NDVI curve
----------
  Reuses the curve cache built by recession-greenup-explore.py at:
    research/recession-and-greenup/longreach-recession-greenup/_cache_ndvi_curve.parquet

  If the cache is absent (e.g. first run on a new machine) the script
  builds it from the pixel-sorted parquet.  Building takes ~5 min.

Usage
-----
    python -m research."ndvi-integral"."longreach-integral"          # all stages
    python -m research."ndvi-integral"."longreach-integral" --stage 1
    python -m research."ndvi-integral"."longreach-integral" --stage 2
    python -m research."ndvi-integral"."longreach-integral" --stage 3
    python -m research."ndvi-integral"."longreach-integral" --stage 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.location import get
from signals import QualityParams, NdviIntegralSignal, NirCvSignal
from signals._shared import annual_ndvi_curve_chunked

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENE_LOC_ID = "longreach-8x8km"
RANKING_CSV  = PROJECT_ROOT / "outputs" / "scores" / "longreach-8x8km" / "longreach_8x8km_pixel_ranking.csv"

# Reuse the NDVI curve cache from recession-and-greenup (same scene, same params).
CURVE_CACHE = (
    PROJECT_ROOT / "research" / "recession-and-greenup"
    / "longreach-recession-greenup" / "_cache_ndvi_curve.parquet"
)

OUT_DIR = PROJECT_ROOT / "research" / "ndvi-integral" / "longreach-integral"

# Confirmed infestation sub-bbox (Presence class) — from longreach.yaml.
INFESTATION_BBOX = [145.423948, -22.764033, 145.424956, -22.761054]

# Grassland sub-bbox (Absence class) — same as EXT_BBOX in recession-greenup-explore.py.
# Also used as the pool for deriving the riparian proxy.
GRASSLAND_BBOX = [145.423948, -22.767104, 145.424956, -22.764033]

CLASS_COLOURS = {"Presence": "#2ca02c", "Absence": "#ff7f0e", "Riparian": "#1f77b4"}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ranking() -> pd.DataFrame:
    """Load the pixel ranking CSV and assign controlled class labels.

    Presence  — infestation bbox pixels.
    Absence   — grassland bbox pixels only.
    Middle    — all other pixels (excluded from class comparisons).
    """
    ranking = pd.read_csv(RANKING_CSV)

    lon_min, lat_min, lon_max, lat_max = INFESTATION_BBOX
    in_inf = (
        (ranking["lon"] >= lon_min) & (ranking["lon"] <= lon_max) &
        (ranking["lat"] >= lat_min) & (ranking["lat"] <= lat_max)
    )

    glon_min, glat_min, glon_max, glat_max = GRASSLAND_BBOX
    in_grass = (
        (ranking["lon"] >= glon_min) & (ranking["lon"] <= glon_max) &
        (ranking["lat"] >= glat_min) & (ranking["lat"] <= glat_max)
    )

    ranking["class"] = "Middle"
    ranking.loc[in_grass, "class"] = "Absence"
    ranking.loc[in_inf,   "class"] = "Presence"

    log(f"  Pixel ranking loaded: {len(ranking):,} total")
    log(f"    Presence (infestation bbox): {in_inf.sum():,}")
    log(f"    Absence  (grassland bbox):   {in_grass.sum():,}")
    log(f"    Middle   (excluded):         {(~in_inf & ~in_grass).sum():,}")
    return ranking


def require_curve(loc) -> Path:
    """Return a Path to the NDVI curve parquet, building it if not cached."""
    if CURVE_CACHE.exists():
        log(f"  NDVI curve cache hit: {CURVE_CACHE.name}")
        return CURVE_CACHE

    base     = loc.parquet_path()
    by_pixel = base.with_name(base.stem + "-by-pixel.parquet")
    q        = QualityParams()
    log(f"  Building NDVI curve from {by_pixel.name} (~5 min) ...")
    CURVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    annual_ndvi_curve_chunked(
        by_pixel, out_path=CURVE_CACHE,
        smooth_days=30, min_obs_per_year=q.min_obs_per_year,
        scl_purity_min=0.0,
    )
    log(f"  NDVI curve written: {CURVE_CACHE}")
    return CURVE_CACHE


def load_raw_pixels(loc) -> pl.DataFrame:
    """Load raw observation parquet, selecting only columns needed here."""
    _LOAD_COLS = ["point_id", "lon", "lat", "date", "scl_purity", "B04", "B08"]
    path = loc.parquet_path()
    log(f"  Loading raw pixels from {path.name} ...")
    df = pl.read_parquet(path, columns=_LOAD_COLS).with_columns([
        pl.col("lon").cast(pl.Float32),
        pl.col("lat").cast(pl.Float32),
        pl.col("B04").cast(pl.Float32),
        pl.col("B08").cast(pl.Float32),
        pl.col("scl_purity").cast(pl.Float32),
    ])
    log(f"    {len(df):,} observations, {df['point_id'].n_unique():,} pixels")
    return df


def derive_riparian_proxy(raw_df: pl.DataFrame, loc) -> list[str]:
    """Return point_ids for the riparian proxy: top-10% dry-season NIR mean
    within the grassland sub-bbox.

    Replicates the algorithm from recession-greenup-explore.py so that the
    riparian proxy is consistent across investigations.
    """
    glon_min, glat_min, glon_max, glat_max = GRASSLAND_BBOX
    ext_df = raw_df.filter(
        (pl.col("lon") >= glon_min) & (pl.col("lon") <= glon_max) &
        (pl.col("lat") >= glat_min) & (pl.col("lat") <= glat_max)
    )
    n_ext = ext_df["point_id"].n_unique()
    log(f"  Grassland bbox: {n_ext} pixels for riparian proxy derivation")

    if n_ext == 0:
        log("  WARNING: No pixels in grassland bbox — riparian proxy will be empty.")
        return []

    nir_stats = NirCvSignal().compute(ext_df, loc)
    thresh    = nir_stats["nir_mean"].quantile(0.90)
    rip_ids   = nir_stats.loc[nir_stats["nir_mean"] >= thresh, "point_id"].tolist()
    log(f"  Riparian proxy: {len(rip_ids)} pixels (top-10% NIR mean, threshold={thresh:.1f})")
    return rip_ids


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def class_vals(df: pd.DataFrame, col: str, ranking: pd.DataFrame, cls: str) -> pd.Series:
    ids = ranking.loc[ranking["class"] == cls, "point_id"]
    return df.loc[df["point_id"].isin(ids), col].dropna()


def median_sep(a: pd.Series, b: pd.Series) -> float:
    if a.empty or b.empty:
        return float("nan")
    return abs(float(a.median()) - float(b.median()))


# ---------------------------------------------------------------------------
# Stage 1 — Compute and map
# ---------------------------------------------------------------------------

def stage1(ranking: pd.DataFrame, curve_path: Path) -> pd.DataFrame:
    """Compute NdviIntegralSignal, write spatial map and class histograms.

    Returns the stats DataFrame (reused by later stages).
    """
    log("\n=== Stage 1: Compute ndvi_integral — spatial map and class histograms ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loc   = get(SCENE_LOC_ID)
    sig   = NdviIntegralSignal()
    stats = sig.compute(pixel_df=None, loc=loc, _curve=curve_path)
    log(f"    {len(stats):,} pixels | {stats['n_reliable_years'].gt(0).sum():,} with ≥1 reliable year")

    pres_vals = class_vals(stats, "ndvi_integral", ranking, "Presence")
    abs_vals  = class_vals(stats, "ndvi_integral", ranking, "Absence")
    sep       = median_sep(pres_vals, abs_vals)
    log(f"    |median sep| = {sep:.4f}  "
        f"(Presence={pres_vals.median():.4f}, Absence={abs_vals.median():.4f})")

    # ------------------------------------------------------------------
    # Figure 1a — Spatial map of ndvi_integral
    # ------------------------------------------------------------------
    log("  Figure 1a: spatial map of ndvi_integral")

    map_df = stats.dropna(subset=["ndvi_integral"]).sort_values("ndvi_integral")
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        "Longreach — Mean annual NDVI (integral)\n"
        "higher = more sustained greenness across the year",
        fontsize=10,
    )
    sc = ax.scatter(map_df["lon"], map_df["lat"],
                    c=map_df["ndvi_integral"], cmap="YlGn",
                    s=3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="ndvi_integral")

    for bbox, col, lbl in [
        (INFESTATION_BBOX, CLASS_COLOURS["Presence"], "Infestation"),
        (GRASSLAND_BBOX,   CLASS_COLOURS["Absence"],  "Grassland"),
    ]:
        lon_min, lat_min, lon_max, lat_max = bbox
        rect = plt.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=1.5, edgecolor=col, facecolor="none", label=lbl,
        )
        ax.add_patch(rect)

    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("Lon", fontsize=9)
    ax.set_ylabel("Lat", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1a_integral_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s1a_integral_map.png")

    # ------------------------------------------------------------------
    # Figure 1b — Class histograms (presence vs absence)
    # ------------------------------------------------------------------
    log("  Figure 1b: class histograms (presence vs absence)")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Longreach — NDVI integral distribution by class\n"
        "Presence = infestation bbox  |  Absence = grassland bbox",
        fontsize=10,
    )
    for label, col in [("Presence", CLASS_COLOURS["Presence"]),
                       ("Absence",  CLASS_COLOURS["Absence"])]:
        vals = class_vals(stats, "ndvi_integral", ranking, label)
        if vals.empty:
            continue
        ax.hist(vals, bins=30, alpha=0.55, color=col, edgecolor="none",
                label=f"{label}  n={len(vals):,}  med={vals.median():.4f}")

    ax.set_xlabel("ndvi_integral (mean annual NDVI)", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    ax.text(0.97, 0.97, f"|median sep| = {sep:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1b_integral_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: s1b_integral_hist.png  |  |median sep| = {sep:.4f}")

    # ------------------------------------------------------------------
    # Figure 1c — ndvi_integral_cv class histograms
    # ------------------------------------------------------------------
    log("  Figure 1c: ndvi_integral_cv histograms by class")

    pres_cv = class_vals(stats, "ndvi_integral_cv", ranking, "Presence")
    abs_cv  = class_vals(stats, "ndvi_integral_cv", ranking, "Absence")
    sep_cv  = median_sep(pres_cv, abs_cv)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Longreach — NDVI integral CV (inter-annual consistency)\n"
        "lower = more consistent mean annual greenness across years",
        fontsize=10,
    )
    for label, col, vals in [
        ("Presence", CLASS_COLOURS["Presence"], pres_cv),
        ("Absence",  CLASS_COLOURS["Absence"],  abs_cv),
    ]:
        if vals.empty:
            continue
        ax.hist(vals, bins=30, alpha=0.55, color=col, edgecolor="none",
                label=f"{label}  n={len(vals):,}  med={vals.median():.4f}")

    ax.set_xlabel("ndvi_integral_cv", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    ax.text(0.97, 0.97, f"|median sep| = {sep_cv:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1c_integral_cv_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: s1c_integral_cv_hist.png  |  ndvi_integral_cv |median sep| = {sep_cv:.4f}")

    return stats


# ---------------------------------------------------------------------------
# Stage 2 — Correlation analysis
# ---------------------------------------------------------------------------

def stage2(stats: pd.DataFrame) -> None:
    """Pearson r between ndvi_integral and existing features / prob_lr.

    Prints a correlation table and computes the residual correlation after
    regressing out rec_p, to test H2: ndvi_integral carries information
    beyond rec_p.
    """
    log("\n=== Stage 2: Correlation analysis — ndvi_integral vs existing features ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(RANKING_CSV)
    merged  = stats.merge(
        ranking[["point_id", "rec_p", "nir_cv", "re_p10", "prob_lr"]],
        on="point_id", how="inner",
    ).dropna(subset=["ndvi_integral"])

    features = ["rec_p", "nir_cv", "re_p10", "prob_lr"]
    log("\n  Pearson r with ndvi_integral:")
    for feat in features:
        sub = merged[["ndvi_integral", feat]].dropna()
        if len(sub) < 10:
            log(f"    {feat:12s}: insufficient data")
            continue
        r = sub["ndvi_integral"].corr(sub[feat])
        log(f"    {feat:12s}:  r = {r:+.3f}  (n={len(sub):,})")

    # Residual correlation: partial out rec_p, then correlate residual with prob_lr.
    # This tests H2: does ndvi_integral still correlate with prob_lr after accounting
    # for the information rec_p already contains?
    log("\n  Residual correlation (ndvi_integral | rec_p) vs prob_lr:")
    sub2 = merged[["ndvi_integral", "rec_p", "prob_lr"]].dropna()
    if len(sub2) >= 10:
        from numpy.polynomial.polynomial import polyfit
        # Regress ndvi_integral on rec_p, keep residuals
        rec_p_norm = (sub2["rec_p"] - sub2["rec_p"].mean()) / sub2["rec_p"].std()
        coef = np.polyfit(rec_p_norm, sub2["ndvi_integral"], 1)
        residual = sub2["ndvi_integral"] - np.polyval(coef, rec_p_norm)
        r_res = residual.corr(sub2["prob_lr"])
        log(f"    r(residual, prob_lr) = {r_res:+.3f}  (n={len(sub2):,})")
        log(f"    H2 threshold: r > 0.20 → {'PASS' if abs(r_res) > 0.20 else 'FAIL / weak'}")
    else:
        log("    Insufficient overlapping data for residual analysis.")

    # ------------------------------------------------------------------
    # Figure 2a — Scatter: ndvi_integral vs rec_p (coloured by prob_lr)
    # ------------------------------------------------------------------
    log("  Figure 2a: scatter ndvi_integral vs rec_p, coloured by prob_lr")

    sub_scatter = merged[["ndvi_integral", "rec_p", "prob_lr"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        "Longreach — ndvi_integral vs rec_p\n"
        "colour = prob_lr (Parkinsonia probability)",
        fontsize=10,
    )
    sc = ax.scatter(
        sub_scatter["rec_p"], sub_scatter["ndvi_integral"],
        c=sub_scatter["prob_lr"], cmap="RdYlGn",
        s=2, alpha=0.4, rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="prob_lr")
    r_val = sub_scatter["ndvi_integral"].corr(sub_scatter["rec_p"])
    ax.text(0.03, 0.97, f"r = {r_val:+.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    ax.set_xlabel("rec_p (wet/dry NDVI amplitude)", fontsize=10)
    ax.set_ylabel("ndvi_integral (mean annual NDVI)", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s2a_scatter_integral_vs_recp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s2a_scatter_integral_vs_recp.png")

    # ------------------------------------------------------------------
    # Figure 2b — Correlation bar chart
    # ------------------------------------------------------------------
    log("  Figure 2b: correlation bar chart")

    corr_vals, corr_labs = [], []
    for feat in features:
        sub = merged[["ndvi_integral", feat]].dropna()
        if len(sub) >= 10:
            corr_vals.append(sub["ndvi_integral"].corr(sub[feat]))
            corr_labs.append(feat)

    if corr_vals:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle("Pearson r: ndvi_integral vs existing features", fontsize=10)
        colours = ["#e74c3c" if abs(v) > 0.5 else "#3498db" for v in corr_vals]
        ax.barh(corr_labs, corr_vals, color=colours, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(0.2,  color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(-0.2, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Pearson r", fontsize=10)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "s2b_corr_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("    Saved: s2b_corr_bar.png")


# ---------------------------------------------------------------------------
# Stage 3 — Riparian proxy check
# ---------------------------------------------------------------------------

def stage3(stats: pd.DataFrame, ranking: pd.DataFrame, rip_ids: list[str]) -> None:
    """Strip plot of ndvi_integral for presence / absence / riparian proxy.

    H3: riparian median should sit below presence despite similar rec_p.
    """
    log("\n=== Stage 3: Riparian proxy check (H3) ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not rip_ids:
        log("  No riparian proxy pixels available — skipping Stage 3.")
        return

    full_ranking_with_rip = ranking.copy()
    full_ranking_with_rip.loc[
        full_ranking_with_rip["point_id"].isin(rip_ids), "class"
    ] = "Riparian"

    pres_vals = class_vals(stats, "ndvi_integral", full_ranking_with_rip, "Presence")
    abs_vals  = class_vals(stats, "ndvi_integral", full_ranking_with_rip, "Absence")
    rip_vals  = stats.loc[stats["point_id"].isin(rip_ids), "ndvi_integral"].dropna()

    log(f"  Presence  n={len(pres_vals):,}  median={pres_vals.median():.4f}")
    log(f"  Absence   n={len(abs_vals):,}  median={abs_vals.median():.4f}")
    log(f"  Riparian  n={len(rip_vals):,}  median={rip_vals.median():.4f}")
    if not rip_vals.empty and not pres_vals.empty:
        direction = "below" if rip_vals.median() < pres_vals.median() else "above"
        log(f"  H3 direction: riparian sits {direction} presence — "
            f"{'consistent' if direction == 'below' else 'inconsistent'} with H3")

    # Also compare against rec_p to contextualise the separation
    ranking_csv = pd.read_csv(RANKING_CSV)
    merged_rip = stats.merge(
        ranking_csv[["point_id", "rec_p"]], on="point_id", how="inner"
    )
    merged_rip["class3"] = "Middle"
    merged_rip.loc[merged_rip["point_id"].isin(
        full_ranking_with_rip.loc[full_ranking_with_rip["class"] == "Presence", "point_id"]
    ), "class3"] = "Presence"
    merged_rip.loc[merged_rip["point_id"].isin(
        full_ranking_with_rip.loc[full_ranking_with_rip["class"] == "Absence", "point_id"]
    ), "class3"] = "Absence"
    merged_rip.loc[merged_rip["point_id"].isin(rip_ids), "class3"] = "Riparian"

    log("\n  rec_p summary by class:")
    for cls in ["Presence", "Absence", "Riparian"]:
        v = merged_rip.loc[merged_rip["class3"] == cls, "rec_p"].dropna()
        if not v.empty:
            log(f"    {cls:10s}  rec_p median={v.median():.4f}")

    # ------------------------------------------------------------------
    # Figure 3a — Strip plot: ndvi_integral by class (Presence/Absence/Riparian)
    # ------------------------------------------------------------------
    log("  Figure 3a: strip plot — ndvi_integral by class")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "Longreach — ndvi_integral by class\n"
        "H3: riparian should sit below presence despite similar rec_p",
        fontsize=10,
    )
    rng = np.random.default_rng(42)
    classes = [
        ("Presence", pres_vals, CLASS_COLOURS["Presence"]),
        ("Absence",  abs_vals,  CLASS_COLOURS["Absence"]),
        ("Riparian", rip_vals,  CLASS_COLOURS["Riparian"]),
    ]
    for xi, (label, vals, col) in enumerate(classes):
        if vals.empty:
            continue
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(np.full(len(vals), xi) + jitter, vals,
                   s=10, alpha=0.4, color=col, rasterized=True)
        ax.plot([xi - 0.25, xi + 0.25], [vals.median()] * 2,
                color="black", linewidth=2.0)
        ax.text(xi, vals.median() + 0.003, f"med={vals.median():.4f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c[0] for c in classes], fontsize=10)
    ax.set_ylabel("ndvi_integral (mean annual NDVI)", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s3a_riparian_strip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s3a_riparian_strip.png")

    # ------------------------------------------------------------------
    # Figure 3b — Scatter: ndvi_integral vs rec_p, riparian overlaid
    # ------------------------------------------------------------------
    log("  Figure 3b: scatter ndvi_integral vs rec_p with riparian overlay")

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        "Longreach — ndvi_integral vs rec_p\n"
        "Riparian overlaid: do they sit below presence at similar rec_p?",
        fontsize=10,
    )
    for cls, col, alpha, size in [
        ("Absence",  CLASS_COLOURS["Absence"],  0.3, 2),
        ("Presence", CLASS_COLOURS["Presence"], 0.5, 4),
    ]:
        sub = merged_rip[merged_rip["class3"] == cls].dropna(subset=["ndvi_integral", "rec_p"])
        ax.scatter(sub["rec_p"], sub["ndvi_integral"],
                   color=col, s=size, alpha=alpha, rasterized=True, label=cls)

    rip_sub = merged_rip[merged_rip["class3"] == "Riparian"].dropna(subset=["ndvi_integral", "rec_p"])
    ax.scatter(rip_sub["rec_p"], rip_sub["ndvi_integral"],
               color=CLASS_COLOURS["Riparian"], s=25, alpha=0.9,
               edgecolors="black", linewidths=0.5, zorder=5, label=f"Riparian (n={len(rip_sub)})")

    ax.set_xlabel("rec_p (wet/dry NDVI amplitude)", fontsize=10)
    ax.set_ylabel("ndvi_integral (mean annual NDVI)", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s3b_riparian_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s3b_riparian_scatter.png")


# ---------------------------------------------------------------------------
# Stage 4 — Smoothing sensitivity
# ---------------------------------------------------------------------------

def stage4(ranking: pd.DataFrame, loc) -> None:
    """Repeat class separation for smooth_days in [15, 30, 45].

    Reuses the cached 30-day curve for the 30d case.  For 15d and 45d,
    builds the curve from the pixel-sorted parquet (streaming, not cached).
    """
    log("\n=== Stage 4: Smoothing sensitivity — smooth_days in [15, 30, 45] ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base     = loc.parquet_path()
    by_pixel = base.with_name(base.stem + "-by-pixel.parquet")
    q        = QualityParams()

    results = []
    for sd in [15, 30, 45]:
        if sd == 30 and CURVE_CACHE.exists():
            # Reuse cached 30-day curve — no rebuild
            curve_path = CURVE_CACHE
            log(f"  smooth_days={sd}: reusing cached curve")
        else:
            curve_path = OUT_DIR / f"_cache_ndvi_curve_{sd}d.parquet"
            if not curve_path.exists():
                log(f"  smooth_days={sd}: building curve from {by_pixel.name} ...")
                annual_ndvi_curve_chunked(
                    by_pixel, out_path=curve_path,
                    smooth_days=sd, min_obs_per_year=q.min_obs_per_year,
                    scl_purity_min=0.0,
                )
                log(f"    Written: {curve_path.name}")
            else:
                log(f"  smooth_days={sd}: cache hit ({curve_path.name})")

        sig   = NdviIntegralSignal(NdviIntegralSignal.Params(smooth_days=sd))
        stats = sig.compute(pixel_df=None, loc=loc, _curve=curve_path)

        pres_vals = class_vals(stats, "ndvi_integral", ranking, "Presence")
        abs_vals  = class_vals(stats, "ndvi_integral", ranking, "Absence")
        sep       = median_sep(pres_vals, abs_vals)

        log(f"    smooth_days={sd:2d}: |sep|={sep:.4f}  "
            f"Presence={pres_vals.median():.4f}  Absence={abs_vals.median():.4f}")

        results.append({
            "smooth_days":    sd,
            "separation":     sep,
            "pres_median":    pres_vals.median() if not pres_vals.empty else float("nan"),
            "abs_median":     abs_vals.median()  if not abs_vals.empty  else float("nan"),
        })

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(OUT_DIR / "s4_smoothing_sweep.csv", index=False)
    log("  Saved: s4_smoothing_sweep.csv")

    # ------------------------------------------------------------------
    # Figure 4a — Separation vs smooth_days
    # ------------------------------------------------------------------
    log("  Figure 4a: separation vs smooth_days")

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(
        "NDVI integral — class separation vs smoothing window\n"
        "|Presence median − Absence median| (higher = better)",
        fontsize=10,
    )
    ax.plot(sweep_df["smooth_days"], sweep_df["separation"],
            marker="o", linewidth=1.8, color="#2c7bb6")
    for _, row in sweep_df.iterrows():
        ax.text(row["smooth_days"], row["separation"] + 0.0002,
                f"{row['separation']:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("smooth_days", fontsize=10)
    ax.set_ylabel("|Presence − Absence| median", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xticks(sweep_df["smooth_days"].tolist())
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s4a_smoothing_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s4a_smoothing_sensitivity.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(stages: list[int] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stages = stages is None
    def run_stage(n: int) -> bool:
        return all_stages or n in stages

    loc        = get(SCENE_LOC_ID)
    ranking    = load_ranking()
    curve_path = require_curve(loc)

    # Stage 1 produces stats — reused by 2 and 3.
    # Stage 3 also needs raw pixels for riparian derivation.
    stats   = None
    rip_ids = None

    def _require_stats():
        nonlocal stats
        if stats is not None:
            return
        log("  Requiring integral stats — running Stage 1 ...")
        stats = stage1(ranking, curve_path)

    def _require_rip_ids():
        nonlocal rip_ids
        if rip_ids is not None:
            return
        raw_df  = load_raw_pixels(loc)
        rip_ids = derive_riparian_proxy(raw_df, loc)

    if run_stage(1):
        _require_stats()

    if run_stage(2):
        _require_stats()
        stage2(stats)

    if run_stage(3):
        _require_stats()
        _require_rip_ids()
        stage3(stats, ranking, rip_ids)

    if run_stage(4):
        stage4(ranking, loc)

    log("\nDone. Outputs in: " + str(OUT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, action="append", dest="stages",
                        metavar="N", help="Run only this stage (repeatable)")
    args = parser.parse_args()
    run(stages=args.stages)


if __name__ == "__main__":
    main()
