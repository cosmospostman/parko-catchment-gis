"""scripts/green-up.py — Green-up timing investigation at Longreach 8×8 km.

Three diagnostic stages as specified in research/day-of-year-peak/GREEN-UP.md:

  Stage 1 — Controlled absolute peak_doy validation
             Infestation bbox (Presence) vs grassland bbox only (Absence).
             Avoids the prob_lr circularity and the heterogeneous-absence
             pessimism of the previous two labelling schemes.

  Stage 2 — Relative greenup shift (peak_doy_shift)
             Per-pixel-year offset from same-year neighbourhood median peak DOY,
             averaged across years. Negative = consistently earlier than neighbours.

  Stage 3 — Sensitivity sweep
             Sweep radius R over [250, 500, 1000 m] and amplitude gate
             percentile over [0.05, 0.10, 0.20]. Report class separation
             (infestation bbox vs grassland bbox) for each combination.

Class labels (Steps 1–3)
------------------------
  Presence : pixels within the confirmed infestation sub-bbox (INFESTATION_BBOX)
  Absence  : pixels within the grassland sub-bbox only (GRASSLAND_BBOX)
  Middle   : all other pixels — excluded from class comparisons

Bounding boxes
--------------
  INFESTATION_BBOX — confirmed dense Parkinsonia strip from longreach.yaml.
  GRASSLAND_BBOX   — extension sub-bbox from longreach.yaml (the training
                     "absence" region). Known to be predominantly grassland
                     with no confirmed Parkinsonia; narrow enough that the
                     heterogeneity problem from the full-scene absence class
                     is avoided.

Usage
-----
    python -m scripts."green-up"              # all stages
    python -m scripts."green-up" --stage 1   # single stage
    python -m scripts."green-up" --fast       # stage 3 sweep uses coarser grid
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
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.location import get
from signals import QualityParams, GreenupTimingSignal, GreenupShiftSignal
from signals._shared import annual_ndvi_curve_chunked

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENE_LOC_ID    = "longreach-8x8km"
RANKING_CSV     = PROJECT_ROOT / "outputs" / "scores" / "longreach-8x8km" / "longreach_8x8km_pixel_ranking.csv"
# Reuse the NDVI curve cache produced by recession-greenup-explore (same scene, same params).
CURVE_CACHE     = (PROJECT_ROOT / "research" / "recession-and-greenup"
                   / "longreach-recession-greenup" / "_cache_ndvi_curve.parquet")
OUT_DIR         = PROJECT_ROOT / "research" / "day-of-year-peak" / "longreach-green-up"

# Confirmed infestation sub-bbox (Presence class).
# From longreach.yaml — the dense riverside Parkinsonia strip.
INFESTATION_BBOX = [145.423948, -22.764033, 145.424956, -22.761054]  # [lon_min, lat_min, lon_max, lat_max]

# Grassland sub-bbox (Absence class).
# From longreach.yaml extension region — known predominantly grassland,
# no confirmed Parkinsonia. Narrow homogeneous window avoids the heterogeneous-
# absence pessimism of using the full 8×8 km scene background.
GRASSLAND_BBOX   = [145.423948, -22.767104, 145.424956, -22.764033]

CLASS_COLOURS = {"Presence": "#2ca02c", "Absence": "#ff7f0e"}
YEARS         = list(range(2016, 2022))


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ranking() -> pd.DataFrame:
    """Load pixel ranking CSV and assign controlled class labels.

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

    # Build from the pixel-sorted parquet (same convention as recession-greenup-explore).
    base        = loc.parquet_path()
    by_pixel    = base.with_name(base.stem + "-by-pixel.parquet")
    q           = QualityParams()
    log(f"  Building NDVI curve from {by_pixel.name} ...")
    CURVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    annual_ndvi_curve_chunked(
        by_pixel, out_path=CURVE_CACHE,
        smooth_days=30, min_obs_per_year=q.min_obs_per_year,
        scl_purity_min=0.0,
    )
    log(f"  NDVI curve written: {CURVE_CACHE}")
    return CURVE_CACHE


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def median_sep(pres: pd.Series, abs_: pd.Series) -> float:
    """Absolute difference in class medians (NaN if either class is empty)."""
    if pres.empty or abs_.empty:
        return float("nan")
    return abs(float(pres.median()) - float(abs_.median()))


def class_vals(df: pd.DataFrame, col: str, ranking: pd.DataFrame,
               cls: str) -> pd.Series:
    ids = ranking.loc[ranking["class"] == cls, "point_id"]
    return df.loc[df["point_id"].isin(ids), col].dropna()


# ---------------------------------------------------------------------------
# Stage 1 — Controlled peak_doy validation
# ---------------------------------------------------------------------------

def stage1(ranking: pd.DataFrame, curve_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute GreenupTimingSignal with controlled labels and produce figures.

    Returns (green_stats, per_year) — both reused by later stages.
    """
    log("\n=== Stage 1: Controlled absolute peak_doy — infestation vs grassland bbox ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sig      = GreenupTimingSignal()
    loc      = get(SCENE_LOC_ID)

    log("  Computing GreenupTimingSignal (streaming from curve cache) ...")
    green_stats = sig.compute(pixel_df=None, loc=loc, _curve=curve_path)
    per_year    = sig._peak_doy_per_year(curve_path)
    log(f"    {len(green_stats):,} pixels | {per_year['reliable'].sum():,} reliable pixel-years")

    pres_ids = ranking.loc[ranking["class"] == "Presence", "point_id"].tolist()
    abs_ids  = ranking.loc[ranking["class"] == "Absence",  "point_id"].tolist()

    per_year = per_year.merge(ranking[["point_id", "class"]], on="point_id", how="left")
    per_year["class"] = per_year["class"].fillna("Middle")

    # ------------------------------------------------------------------
    # Figure 1a — Per-year peak DOY strip plot (controlled labels)
    # ------------------------------------------------------------------
    log("  Figure 1a: per-year peak DOY strip plot")

    fig, axes = plt.subplots(1, len(YEARS), figsize=(14, 5), sharey=True)
    fig.suptitle(
        "Longreach — Per-year NDVI peak DOY by class (controlled labelling)\n"
        "Presence = infestation bbox  |  Absence = grassland bbox only",
        fontsize=10,
    )

    rng = np.random.default_rng(42)
    for i, yr in enumerate(YEARS):
        ax    = axes[i]
        yr_df = per_year[(per_year["year"] == yr) & per_year["reliable"]]
        for xi, (label, col) in enumerate([("Presence", CLASS_COLOURS["Presence"]),
                                            ("Absence",  CLASS_COLOURS["Absence"])]):
            vals    = yr_df[yr_df["class"] == label]["peak_doy"].dropna().values
            if len(vals) == 0:
                continue
            jitter  = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), xi) + jitter, vals,
                       s=8, alpha=0.45, color=col, rasterized=True)
            ax.plot([xi - 0.25, xi + 0.25], [np.median(vals)] * 2,
                    color="black", linewidth=2.0)

        ax.set_title(str(yr), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pres", "Abs"], fontsize=8)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_ylabel("Peak DOY", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1a_peak_doy_per_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s1a_peak_doy_per_year.png")

    # ------------------------------------------------------------------
    # Figure 1b — Peak DOY SD histogram (controlled labels)
    # ------------------------------------------------------------------
    log("  Figure 1b: peak DOY SD histogram by class")

    pix_sd = (
        per_year[per_year["reliable"]]
        .groupby("point_id")["peak_doy"]
        .std()
        .reset_index()
        .rename(columns={"peak_doy": "peak_doy_sd"})
    )
    pix_sd = pix_sd.merge(ranking[["point_id", "class"]], on="point_id", how="left")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Longreach — Per-pixel peak DOY standard deviation across years\n"
        "(controlled labels: infestation bbox vs grassland bbox)",
        fontsize=10,
    )
    for label, col in [("Presence", CLASS_COLOURS["Presence"]),
                       ("Absence",  CLASS_COLOURS["Absence"])]:
        vals = pix_sd[pix_sd["class"] == label]["peak_doy_sd"].dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=25, alpha=0.55, color=col, edgecolor="none",
                label=f"{label}  n={len(vals):,}  med={vals.median():.1f}d")

    ax.set_xlabel("Peak DOY SD (days)", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1b_peak_doy_sd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s1b_peak_doy_sd.png")

    # ------------------------------------------------------------------
    # Figure 1c — Mean peak_doy histogram
    # ------------------------------------------------------------------
    log("  Figure 1c: mean peak_doy histogram by class")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Longreach — Mean annual peak DOY distribution by class\n"
        "(controlled labels: infestation bbox vs grassland bbox)",
        fontsize=10,
    )
    for label, col in [("Presence", CLASS_COLOURS["Presence"]),
                       ("Absence",  CLASS_COLOURS["Absence"])]:
        vals = class_vals(green_stats, "peak_doy", ranking, label)
        if vals.empty:
            continue
        ax.hist(vals, bins=30, alpha=0.55, color=col, edgecolor="none",
                label=f"{label}  n={len(vals):,}  med={vals.median():.1f}")

    sep = median_sep(
        class_vals(green_stats, "peak_doy", ranking, "Presence"),
        class_vals(green_stats, "peak_doy", ranking, "Absence"),
    )
    ax.set_title(ax.get_title() or "")
    ax.set_xlabel("Mean annual peak DOY", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    ax.text(0.97, 0.97, f"|median sep| = {sep:.1f} days",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s1c_mean_peak_doy_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: s1c_mean_peak_doy_hist.png  |  |median sep| = {sep:.1f} days")

    # ------------------------------------------------------------------
    # Figure 1d — Spatial map of mean peak_doy
    # ------------------------------------------------------------------
    log("  Figure 1d: spatial map of mean peak_doy")

    map_df = green_stats.dropna(subset=["peak_doy"]).sort_values("peak_doy")
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Longreach — Mean annual NDVI peak DOY\n"
                 "(earlier = sooner wet-season flush)", fontsize=10)
    sc = ax.scatter(map_df["lon"], map_df["lat"],
                    c=map_df["peak_doy"], cmap="plasma",
                    s=3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Peak DOY")
    # Overlay bbox outlines
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
    fig.savefig(OUT_DIR / "s1d_map_peak_doy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s1d_map_peak_doy.png")

    log(f"  Stage 1 summary — |median sep| peak_doy = {sep:.1f} days "
        f"(Presence med={class_vals(green_stats, 'peak_doy', ranking, 'Presence').median():.1f}, "
        f"Absence med={class_vals(green_stats, 'peak_doy', ranking, 'Absence').median():.1f})")

    return green_stats, per_year


# ---------------------------------------------------------------------------
# Stage 2 — Relative greenup shift
# ---------------------------------------------------------------------------

def stage2(
    per_year: pd.DataFrame,
    ranking: pd.DataFrame,
    params: GreenupShiftSignal.Params | None = None,
) -> tuple[pd.DataFrame, "GreenupShiftSignal"]:
    """Compute GreenupShiftSignal and produce diagnostic figures.

    Returns (shift_stats, sig) — sig is passed to stage3 so it can reuse
    the pre-built lookup for the default R/gate combination.
    """
    log("\n=== Stage 2: Relative greenup shift — peak_doy_shift ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if params is None:
        params = GreenupShiftSignal.Params()

    coords     = ranking[["point_id", "lon", "lat"]].copy()
    amp_series = ranking.set_index("point_id")["rec_p"]

    sig = GreenupShiftSignal(params)

    log(f"  Building neighbourhood lookup  R={params.radius_m:.0f}m  "
        f"amp_gate={params.amp_gate_percentile:.0%} ...")
    sig.build_lookup(per_year, coords, amp_series=amp_series)
    log(f"    Lookup built: {len(sig._lookup):,} (pixel, year) entries")

    shift_stats = sig.compute(per_year, coords)

    n_valid    = shift_stats["peak_doy_shift"].notna().sum()
    pres_shift = class_vals(shift_stats, "peak_doy_shift", ranking, "Presence")
    abs_shift  = class_vals(shift_stats, "peak_doy_shift", ranking, "Absence")
    sep        = median_sep(pres_shift, abs_shift)
    log(f"    {n_valid:,} pixels with valid shift | "
        f"|median sep| = {sep:.2f}d  "
        f"(Presence={pres_shift.median():.2f}, Absence={abs_shift.median():.2f})")

    # ------------------------------------------------------------------
    # Figure 2a — peak_doy_shift histogram by class
    # ------------------------------------------------------------------
    log("  Figure 2a: peak_doy_shift histogram by class")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Longreach — Relative greenup shift (R={params.radius_m:.0f}m)\n"
        "Presence = infestation bbox  |  Absence = grassland bbox",
        fontsize=10,
    )
    for label, col in [("Presence", CLASS_COLOURS["Presence"]),
                       ("Absence",  CLASS_COLOURS["Absence"])]:
        vals = class_vals(shift_stats, "peak_doy_shift", ranking, label)
        if vals.empty:
            continue
        ax.hist(vals, bins=25, alpha=0.55, color=col, edgecolor="none",
                label=f"{label}  n={len(vals):,}  med={vals.median():.1f}d")

    ax.axvline(0, color="grey", linewidth=1.2, linestyle="--", label="0 (same as neighbours)")
    ax.set_xlabel("Peak DOY shift (days; negative = earlier)", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    ax.text(0.97, 0.97, f"|median sep| = {sep:.1f} days",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s2a_shift_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s2a_shift_hist.png")

    # ------------------------------------------------------------------
    # Figure 2b — peak_doy_shift_sd histogram by class
    # ------------------------------------------------------------------
    log("  Figure 2b: peak_doy_shift_sd histogram by class")

    pres_sd = class_vals(shift_stats, "peak_doy_shift_sd", ranking, "Presence")
    abs_sd  = class_vals(shift_stats, "peak_doy_shift_sd", ranking, "Absence")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Longreach — SD of relative greenup shift across years (R={params.radius_m:.0f}m)\n"
        "Lower = more consistent offset from neighbourhood",
        fontsize=10,
    )
    for label, col, vals in [
        ("Presence", CLASS_COLOURS["Presence"], pres_sd),
        ("Absence",  CLASS_COLOURS["Absence"],  abs_sd),
    ]:
        if vals.empty:
            continue
        ax.hist(vals, bins=25, alpha=0.55, color=col, edgecolor="none",
                label=f"{label}  n={len(vals):,}  med={vals.median():.1f}d")

    ax.set_xlabel("Peak DOY shift SD (days)", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s2b_shift_sd_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s2b_shift_sd_hist.png")

    # ------------------------------------------------------------------
    # Figure 2c — Spatial map of peak_doy_shift
    # ------------------------------------------------------------------
    log("  Figure 2c: spatial map of peak_doy_shift")

    map_df = shift_stats.dropna(subset=["peak_doy_shift"]).sort_values("peak_doy_shift")
    vlim   = np.nanpercentile(map_df["peak_doy_shift"].abs(), 95)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        f"Longreach — Relative greenup shift (R={params.radius_m:.0f}m)\n"
        "Negative = peaks earlier than local neighbourhood",
        fontsize=10,
    )
    sc = ax.scatter(map_df["lon"], map_df["lat"],
                    c=map_df["peak_doy_shift"], cmap="RdBu_r",
                    vmin=-vlim, vmax=vlim, s=3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Shift (days)")
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
    fig.savefig(OUT_DIR / "s2c_shift_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s2c_shift_map.png")

    # ------------------------------------------------------------------
    # Figure 2d — Per-year shift strip plot
    # Re-use the lookup: merge per_year against sig._lookup to get per-year
    # offsets for labelled pixels directly — no extra spatial work.
    # ------------------------------------------------------------------
    log("  Figure 2d: per-year shift strip plot")

    labelled_ids = set(ranking.loc[ranking["class"].isin(["Presence", "Absence"]), "point_id"])
    per_year_lab = (
        per_year[per_year["point_id"].isin(labelled_ids) & per_year["reliable"]]
        .drop(columns=["class"], errors="ignore")
        .merge(sig._lookup, on=["point_id", "year"], how="inner")
        .merge(ranking[["point_id", "class"]], on="point_id", how="left")
    )
    per_year_lab["shift_yr"] = per_year_lab["peak_doy"] - per_year_lab["nbr_median"]

    rng  = np.random.default_rng(7)
    fig, axes = plt.subplots(1, len(YEARS), figsize=(14, 5), sharey=True)
    fig.suptitle(
        f"Longreach — Per-year relative greenup shift (R={params.radius_m:.0f}m)\n"
        "Presence = infestation bbox  |  Absence = grassland bbox",
        fontsize=10,
    )
    for i, yr in enumerate(YEARS):
        ax    = axes[i]
        yr_df = per_year_lab[per_year_lab["year"] == yr]
        for xi, (label, col) in enumerate([("Presence", CLASS_COLOURS["Presence"]),
                                            ("Absence",  CLASS_COLOURS["Absence"])]):
            vals = yr_df[yr_df["class"] == label]["shift_yr"].dropna().values
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), xi) + jitter, vals,
                       s=8, alpha=0.45, color=col, rasterized=True)
            ax.plot([xi - 0.25, xi + 0.25], [np.median(vals)] * 2,
                    color="black", linewidth=2.0)

        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(str(yr), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pres", "Abs"], fontsize=8)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_ylabel("Shift from neighbourhood median (days)", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "s2d_shift_per_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s2d_shift_per_year.png")

    return shift_stats, sig


# ---------------------------------------------------------------------------
# Stage 3 — Sensitivity sweep
# ---------------------------------------------------------------------------

def stage3(
    per_year: pd.DataFrame,
    ranking: pd.DataFrame,
    fast: bool = False,
) -> None:
    """Sweep radius and amplitude gate; report class separation for each combo.

    Each (R, gate) combination requires its own ``build_lookup`` call because
    both parameters affect which pixels are gated and what their neighbourhood
    pools are.  The lookup is built once per combination; ``compute`` is then
    a cheap merge.
    """
    log("\n=== Stage 3: Sensitivity sweep — R × amplitude gate ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    radii     = [250.0, 500.0, 1000.0]
    amp_gates = [0.05, 0.10, 0.20] if not fast else [0.10]

    coords     = ranking[["point_id", "lon", "lat"]].copy()
    amp_series = ranking.set_index("point_id")["rec_p"]

    results = []
    for R in radii:
        for gate in amp_gates:
            params = GreenupShiftSignal.Params(radius_m=R, amp_gate_percentile=gate)
            sig    = GreenupShiftSignal(params)
            try:
                log(f"    Building lookup  R={R:.0f}m  gate={gate:.0%} ...")
                sig.build_lookup(per_year, coords, amp_series=amp_series)
                stats     = sig.compute(per_year, coords)
                pres_vals = class_vals(stats, "peak_doy_shift", ranking, "Presence")
                abs_vals  = class_vals(stats, "peak_doy_shift", ranking, "Absence")
                sep       = median_sep(pres_vals, abs_vals)
                n_valid   = stats["peak_doy_shift"].notna().sum()
                results.append({
                    "radius_m":        R,
                    "amp_gate_pct":    gate,
                    "separation_days": sep,
                    "pres_median":     pres_vals.median() if not pres_vals.empty else float("nan"),
                    "abs_median":      abs_vals.median()  if not abs_vals.empty  else float("nan"),
                    "n_valid_pixels":  n_valid,
                })
                log(f"      |sep|={sep:.2f}d  n_valid={n_valid:,}")
            except Exception as exc:
                log(f"    R={R:.0f}m  gate={gate:.0%}  ERROR — {exc}")

    if not results:
        log("  No results — sweep produced no output.")
        return

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(OUT_DIR / "s3_sweep.csv", index=False)
    log("  Saved: s3_sweep.csv")

    # ------------------------------------------------------------------
    # Figure 3a — Heatmap of class separation
    # ------------------------------------------------------------------
    log("  Figure 3a: sweep heatmap")

    import itertools

    pivot = sweep_df.pivot(index="radius_m", columns="amp_gate_pct",
                           values="separation_days")

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(
        "Relative greenup shift — class separation by R and amplitude gate\n"
        "|Presence median − Absence median| (days; higher = better)",
        fontsize=10,
    )
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0%}" for v in pivot.columns], fontsize=9)
    ax.set_xlabel("Amplitude gate (percentile)", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(v)}m" for v in pivot.index], fontsize=9)
    ax.set_ylabel("Neighbourhood radius", fontsize=9)

    for i, j in itertools.product(range(len(pivot.index)), range(len(pivot.columns))):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}d", ha="center", va="center", fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|sep| (days)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s3a_sweep_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s3a_sweep_heatmap.png")

    # ------------------------------------------------------------------
    # Figure 3b — Separation vs radius, coloured by amplitude gate
    # ------------------------------------------------------------------
    log("  Figure 3b: separation vs radius line plot")

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Relative greenup shift — separation vs neighbourhood radius", fontsize=10)

    gate_colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(amp_gates)))
    for gate, col in zip(amp_gates, gate_colours):
        sub = sweep_df[sweep_df["amp_gate_pct"] == gate].sort_values("radius_m")
        ax.plot(sub["radius_m"], sub["separation_days"],
                marker="o", linewidth=1.5, color=col, label=f"gate={gate:.0%}")

    ax.set_xlabel("Neighbourhood radius (m)", fontsize=10)
    ax.set_ylabel("|Presence − Absence| median (days)", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "s3b_sep_vs_radius.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("    Saved: s3b_sep_vs_radius.png")

    best = sweep_df.loc[sweep_df["separation_days"].idxmax()]
    log(f"  Best combination: R={best['radius_m']:.0f}m  gate={best['amp_gate_pct']:.0%}  "
        f"|sep|={best['separation_days']:.2f}d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(stages: list[int] | None = None, fast: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stages = stages is None
    def run_stage(n: int) -> bool:
        return all_stages or n in stages

    loc        = get(SCENE_LOC_ID)
    ranking    = load_ranking()
    curve_path = require_curve(loc)

    green_stats = per_year = None

    def _require_green():
        nonlocal green_stats, per_year
        if green_stats is not None and per_year is not None:
            return
        log("  Requiring greenup stats — running Stage 1 ...")
        green_stats, per_year = stage1(ranking, curve_path)

    if run_stage(1):
        _require_green()

    if run_stage(2):
        _require_green()
        stage2(per_year, ranking)  # returns (shift_stats, sig) — not used here

    if run_stage(3):
        _require_green()
        stage3(per_year, ranking, fast=fast)

    log("\nDone. Outputs in: " + str(OUT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, action="append", dest="stages",
                        metavar="N", help="Run only this stage (repeatable)")
    parser.add_argument("--fast", action="store_true",
                        help="Stage 3 sweep uses a reduced parameter grid")
    args = parser.parse_args()
    run(stages=args.stages, fast=args.fast)


if __name__ == "__main__":
    main()
