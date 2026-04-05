"""Flowering flash detection — Longreach Parkinsonia infestation.

Computes five visible/NIR indices sensitive to a yellow-flower spectral signature
and scans the full 2020–2025 time series for a phenological window where infestation
pixels show an anomalous spike above their own baseline.

The extension population is retained for context but the primary signal is within-
infestation: for each pixel, each observation is expressed as a z-score anomaly
relative to that pixel's own annual median for the same DOY bin. A flowering flash
would appear as a transient positive anomaly shared across many infestation pixels
simultaneously, at any time of year.

Indices computed per observation:
  FI_rg   = (B03 + B04) / B08
  FI_r    = B04 / B08
  FI_by   = (B03 + B04) / (B02 + B08)   ← primary; suppresses bare-soil false positives
  dNDVI   = −(B08 − B04) / (B08 + B04)
  FI_swir = B11 / B08

Outputs:
  outputs/longreach-flowering/fi_doy_profiles.png         (DOY anomaly profiles, all indices)
  outputs/longreach-flowering/fi_by_timeseries.png        (raw FI_by + anomaly time series)
  outputs/longreach-flowering/fi_by_spatial_peak.png      (pixel map at peak acquisition date)
  outputs/longreach-flowering/fi_band_decomposition.png   (per-band anomaly on peak dates)
  outputs/longreach-flowering/flowering_window_by_year.csv

See research/LONGREACH-FLOWERING.md for the full analysis plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py")
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_PATH   = PROJECT_ROOT / "data" / "longreach_pixels.parquet"
DRY_NIR_PATH   = PROJECT_ROOT / "outputs" / "longreach-dry-nir" / "longreach_dry_nir_stats.parquet"
OUT_DIR        = PROJECT_ROOT / "outputs" / "longreach-flowering"

# Riparian proxy: extension pixels in the top percentile of dry-season NIR mean.
# These cluster around lat -22.765 in the southern extension (the water feature
# identified in LONGREACH-DRY-NIR.md as outscoring the Parkinsonia patch on raw NIR).
# Using p90 of extension nir_mean as the cutoff (~39 pixels).
RIPARIAN_NIR_PERCENTILE = 90

# Quality filter
SCL_PURITY_MIN = 0.5

# Haze filter: drop acquisition dates where the scene-mean B02 is more than this
# many reflectance units above its DOY-bin median.  Haze raises all visible bands
# together (B02 most strongly); the FI_by numerator rises with haze just as it does
# with flowering, so hazy dates must be excluded before anomaly detection.
# Threshold 0.010 removes strongly hazy dates while retaining the near-clean ones;
# the remaining B02 variance on kept dates is at the noise floor (<0.010).
HAZE_B02_ANOM_MAX = 0.010

# Minimum total observations across the archive to include a pixel
MIN_PIXEL_OBS  = 10

# DOY bin width in days (26 bins × 14 days ≈ full year)
DOY_BIN_DAYS   = 14

# Original infestation patch bbox (used for spatial overlay and success criteria)
HD_LON_MIN, HD_LON_MAX = 145.423948, 145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033, -22.761054

# Infestation patch lat boundary: pixels north of this are infestation, south are extension
INFESTATION_LAT_MIN = -22.764033   # southernmost row of the original 374-pixel fetch

# Full survey bbox (infestation + southern extension) — used for WMS and spatial maps
SURVEY_BBOX = [145.423948, -22.767104, 145.424956, -22.761054]

# Indices to compute — (column_name, label_for_plots)
INDICES = [
    ("FI_rg",   "FI_rg  = (B03+B04)/B08"),
    ("FI_r",    "FI_r   = B04/B08"),
    ("FI_by",   "FI_by  = (B03+B04)/(B02+B08)"),
    ("dNDVI",   "dNDVI  = −(B08−B04)/(B08+B04)"),
    ("FI_swir", "FI_swir = B11/B08"),
]
PRIMARY_INDEX = "FI_by"

# Year-by-year peak detection: flag dates where FI_by exceeds this percentile
# of the pixel's own annual distribution
PEAK_PERCENTILE = 75

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def build_haze_mask(df: pd.DataFrame) -> pd.Index:
    """Return the set of acquisition dates that pass the scene-level haze filter.

    For each date compute the scene-mean B02 reflectance.  Subtract the per-DOY-bin
    median (26 bins of 14 days) to get the B02 anomaly relative to the seasonal
    baseline.  Dates where this anomaly exceeds HAZE_B02_ANOM_MAX are flagged as hazy
    and excluded from all downstream analysis.

    AOT is available in the parquet but has near-zero correlation with B02 anomaly
    (r = −0.09) so B02 itself is the more reliable haze proxy here.
    """
    scene = df.groupby("date")["B02"].mean().reset_index()
    scene["doy"]     = scene["date"].dt.dayofyear
    scene["doy_bin"] = ((scene["doy"] - 1) // DOY_BIN_DAYS) * DOY_BIN_DAYS + 1
    b2_baseline      = scene.groupby("doy_bin")["B02"].median().rename("B02_base")
    scene            = scene.join(b2_baseline, on="doy_bin")
    scene["B02_anom"] = scene["B02"] - scene["B02_base"]

    clean_dates = scene.loc[scene["B02_anom"] <= HAZE_B02_ANOM_MAX, "date"]
    hazy_dates  = scene.loc[scene["B02_anom"] >  HAZE_B02_ANOM_MAX, "date"]

    log(f"\n  Haze filter (scene-mean B02 anomaly > {HAZE_B02_ANOM_MAX}):")
    log(f"    Total acquisition dates: {len(scene)}")
    log(f"    Hazy dates removed:      {len(hazy_dates)}")
    log(f"    Clean dates retained:    {len(clean_dates)}")
    if len(hazy_dates) > 0:
        top = scene[scene["B02_anom"] > HAZE_B02_ANOM_MAX].nlargest(5, "B02_anom")
        for _, row in top.iterrows():
            log(f"      {row['date'].date()}  B02_anom={row['B02_anom']:.4f}")

    return pd.DatetimeIndex(clean_dates)


def load_and_filter(path: Path) -> pd.DataFrame:
    log(f"Loading parquet: {path}")
    df = pd.read_parquet(path)
    log(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    log(f"  Pixels: {df['point_id'].nunique()}  |  Date range: {df['date'].min().date()} – {df['date'].max().date()}")

    before = len(df)
    df = df[df["scl_purity"] >= SCL_PURITY_MIN].copy()
    dropped = before - len(df)
    log(f"  Quality filter (scl_purity ≥ {SCL_PURITY_MIN}): dropped {dropped:,} rows "
        f"({100 * dropped / before:.1f}%), retained {len(df):,}")

    # Drop pixels with too few total observations (edge pixels)
    obs_per_pixel = df.groupby("point_id").size()
    valid_pixels  = obs_per_pixel[obs_per_pixel >= MIN_PIXEL_OBS].index
    n_dropped_px  = df["point_id"].nunique() - len(valid_pixels)
    df = df[df["point_id"].isin(valid_pixels)].copy()
    log(f"  Min-obs filter (≥ {MIN_PIXEL_OBS} obs per pixel): dropped {n_dropped_px} pixels, "
        f"retained {df['point_id'].nunique()} pixels")

    # Haze filter: drop acquisition dates with elevated scene-mean B02
    clean_dates = build_haze_mask(df)
    before_haze = len(df)
    df = df[df["date"].isin(clean_dates)].copy()
    log(f"  Rows after haze filter: {len(df):,} (dropped {before_haze - len(df):,})")

    return df


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    log("\nComputing flowering indices...")
    df = df.copy()

    df["FI_rg"]   = (df["B03"] + df["B04"]) / df["B08"]
    df["FI_r"]    = df["B04"] / df["B08"]
    df["FI_by"]   = (df["B03"] + df["B04"]) / (df["B02"] + df["B08"])
    df["dNDVI"]   = -(df["B08"] - df["B04"]) / (df["B08"] + df["B04"])
    df["FI_swir"] = df["B11"] / df["B08"]

    # Replace inf / -inf from near-zero denominators with NaN
    index_cols = [name for name, _ in INDICES]
    df[index_cols] = df[index_cols].replace([np.inf, -np.inf], np.nan)

    for name, label in INDICES:
        valid = df[name].dropna()
        log(f"  {name:8s}  n_valid={len(valid):,}  "
            f"min={valid.min():.4f}  median={valid.median():.4f}  max={valid.max():.4f}")

    # Label pixels as infestation or extension
    coords = df[["point_id", "lat"]].drop_duplicates("point_id")
    infestation_ids = coords[coords["lat"] >= INFESTATION_LAT_MIN]["point_id"]
    df["population"] = np.where(df["point_id"].isin(infestation_ids), "infestation", "extension")
    n_inf = df[df["population"] == "infestation"]["point_id"].nunique()
    n_ext = df[df["population"] == "extension"]["point_id"].nunique()
    log(f"\n  Infestation pixels: {n_inf}  |  Extension pixels: {n_ext}")

    return df


def doy_bin(doy: pd.Series) -> pd.Series:
    """Map day-of-year to bin start DOY (0-indexed bins of DOY_BIN_DAYS days)."""
    return ((doy - 1) // DOY_BIN_DAYS) * DOY_BIN_DAYS + 1


def compute_pixel_anomalies(pop_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-pixel z-score anomaly columns to a population subset.

    Baseline: per-pixel per-DOY-bin median across all years.
    Denominator: per-pixel overall std across all dates and seasons.

    Works on any population — call separately for infestation and extension
    so each pixel's baseline is estimated from its own time series.
    """
    pop_df = pop_df.copy()
    pop_df["doy"]     = pop_df["date"].dt.dayofyear
    pop_df["doy_bin"] = doy_bin(pop_df["doy"])
    pop_df["year"]    = pop_df["date"].dt.year

    index_cols = [name for name, _ in INDICES]

    baseline = (
        pop_df.groupby(["point_id", "doy_bin"])[index_cols]
        .median()
        .reset_index()
        .rename(columns={c: f"{c}_base" for c in index_cols})
    )
    pixel_std = (
        pop_df.groupby("point_id")[index_cols]
        .std()
        .reset_index()
        .rename(columns={c: f"{c}_std" for c in index_cols})
    )

    pop_df = pop_df.merge(baseline,   on=["point_id", "doy_bin"], how="left")
    pop_df = pop_df.merge(pixel_std,  on="point_id",              how="left")

    for name in index_cols:
        std_safe = pop_df[f"{name}_std"].where(pop_df[f"{name}_std"] > 1e-6, np.nan)
        pop_df[f"{name}_z"] = (pop_df[name] - pop_df[f"{name}_base"]) / std_safe

    return pop_df


def build_doy_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-pixel anomaly z-scores for both populations; DOY profiles for infestation.

    Returns
    -------
    profiles      : infestation mean z-score per DOY bin (used for plots + criteria)
    inf_anomalies : per-(pixel, date) anomaly scores for infestation pixels
    ext_anomalies : per-(pixel, date) anomaly scores for extension pixels
    """
    log("\nBuilding within-pixel DOY anomaly z-scores (both populations)...")

    inf_anomalies = compute_pixel_anomalies(df[df["population"] == "infestation"])
    ext_anomalies = compute_pixel_anomalies(df[df["population"] == "extension"])

    log(f"  Infestation pixels: {inf_anomalies['point_id'].nunique()}  "
        f"observations: {len(inf_anomalies):,}")
    log(f"  Extension pixels:   {ext_anomalies['point_id'].nunique()}  "
        f"observations: {len(ext_anomalies):,}")

    z_cols    = [f"{name}_z" for name, _ in INDICES]
    primary_z = f"{PRIMARY_INDEX}_z_mean"

    # Mean anomaly across infestation pixels per DOY bin (profiles for plots)
    profiles_long = (
        inf_anomalies.groupby("doy_bin")[z_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    profiles_long.columns = [
        "doy_bin" if c[0] == "doy_bin" else f"{c[0]}_{c[1]}"
        for c in profiles_long.columns
    ]

    log(f"\n  {PRIMARY_INDEX} mean within-infestation z-score anomaly by DOY bin:")
    for _, row in profiles_long.sort_values("doy_bin").iterrows():
        val = row[primary_z]
        if np.isnan(val):
            log(f"    DOY {int(row['doy_bin']):3d}  z=   NaN")
            continue
        bar_pos = "█" * max(0, int(val * 10))
        bar_neg = "▒" * max(0, int(-val * 10))
        log(f"    DOY {int(row['doy_bin']):3d}  z={val:+.3f}  {bar_neg}{bar_pos}")

    peak_bin = int(profiles_long.loc[profiles_long[primary_z].idxmax(), "doy_bin"])
    log(f"\n  Peak mean z-score anomaly at DOY bin {peak_bin} "
        f"(z={profiles_long.loc[profiles_long['doy_bin']==peak_bin, primary_z].values[0]:+.3f})")

    return profiles_long, inf_anomalies, ext_anomalies


def build_timeseries(df: pd.DataFrame, inf_anomalies: pd.DataFrame) -> pd.DataFrame:
    """Scene-mean FI_by per date (both populations) plus mean infestation z-score."""
    log("\nBuilding FI_by scene-mean time series...")

    # Raw FI_by mean per date × population
    ts = (
        df.groupby(["population", "date"])[PRIMARY_INDEX]
        .mean()
        .reset_index()
        .pivot(index="date", columns="population", values=PRIMARY_INDEX)
        .reset_index()
    )
    ts.columns.name = None

    # Mean z-score anomaly across infestation pixels per date
    z_col = f"{PRIMARY_INDEX}_z"
    inf_z = (
        inf_anomalies.groupby("date")[z_col]
        .mean()
        .reset_index()
        .rename(columns={z_col: "inf_z"})
    )
    ts = ts.merge(inf_z, on="date", how="left")

    log(f"  Dates in time series: {len(ts)}")
    log(f"  FI_by mean — infestation: {ts['infestation'].mean():.4f}  "
        f"extension: {ts['extension'].mean():.4f}")
    log(f"  Infestation z-score — mean: {ts['inf_z'].mean():.3f}  "
        f"max: {ts['inf_z'].max():.3f}  "
        f"on {ts.loc[ts['inf_z'].idxmax(), 'date'].date() if ts['inf_z'].notna().any() else 'n/a'}")

    return ts


def build_contrast_timeseries(inf_anomalies: pd.DataFrame,
                               ext_anomalies: pd.DataFrame) -> pd.DataFrame:
    """Per-date contrast: mean infestation FI_by z-score minus mean extension z-score.

    Mirrors step 3 of the red-edge analysis.  A positive contrast means the infestation
    is above its own baseline more than the extension is above its own baseline on that
    date — i.e. the two populations are diverging, not moving together (which would
    indicate a scene-wide effect like haze or rain).

    Returns a dataframe with one row per date containing:
      inf_z, ext_z, contrast, doy, year
    plus a 30-day rolling mean of contrast.
    """
    log("\nBuilding inter-population FI_by z-score contrast time series...")
    z_col = f"{PRIMARY_INDEX}_z"

    inf_scene = inf_anomalies.groupby("date")[z_col].mean().rename("inf_z")
    ext_scene = ext_anomalies.groupby("date")[z_col].mean().rename("ext_z")

    ct = pd.concat([inf_scene, ext_scene], axis=1).reset_index()
    ct["contrast"] = ct["inf_z"] - ct["ext_z"]
    ct["doy"]      = ct["date"].dt.dayofyear
    ct["year"]     = ct["date"].dt.year

    ct = ct.sort_values("date").reset_index(drop=True)
    ct["contrast_30d"] = ct.set_index("date")["contrast"].rolling("30D").mean().values

    frac_pos = (ct["contrast"] > 0).mean()
    log(f"  Dates: {len(ct)}  |  Fraction with contrast > 0: {frac_pos:.2f}")

    log(f"\n  Per-year contrast summary:")
    for year, grp in ct.groupby("year"):
        peak_row = grp.loc[grp["contrast"].idxmax()]
        log(f"    {year}  max contrast={peak_row['contrast']:+.3f} on {peak_row['date'].date()} "
            f"(DOY {int(peak_row['doy'])})  frac>0={( grp['contrast']>0).mean():.2f}")

    # DOY profile: mean contrast per 14-day bin
    ct["doy_bin"] = ((ct["doy"] - 1) // DOY_BIN_DAYS) * DOY_BIN_DAYS + 1
    doy_contrast  = ct.groupby("doy_bin")["contrast"].agg(["mean","std"]).reset_index()
    peak_bin      = int(doy_contrast.loc[doy_contrast["mean"].idxmax(), "doy_bin"])
    log(f"\n  Peak mean contrast DOY bin: {peak_bin} "
        f"(contrast={doy_contrast.loc[doy_contrast['doy_bin']==peak_bin,'mean'].values[0]:+.3f})")

    return ct


def build_flowering_windows(inf_anomalies: pd.DataFrame) -> pd.DataFrame:
    """Per-year dates where mean infestation FI_by z-score anomaly exceeds 1.0.

    Uses the within-infestation anomaly time series so the threshold is relative
    to each pixel's own seasonal baseline, not an absolute index value.
    """
    log(f"\nDetecting per-year flowering windows (mean FI_by z-score > 1.0)...")

    z_col = f"{PRIMARY_INDEX}_z"
    inf   = inf_anomalies.copy()
    inf["year"] = inf["date"].dt.year

    rows = []
    for year, grp in inf.groupby("year"):
        # Mean z-score across pixels per acquisition date
        scene_z  = grp.groupby("date")[z_col].mean()
        elevated = scene_z[scene_z >= 1.0]
        if len(elevated) == 0:
            log(f"  {year}: no dates with mean z-score ≥ 1.0  "
                f"(year max z={scene_z.max():.3f})")
            rows.append({"year": year, "n_dates": 0,
                         "doy_start": np.nan, "doy_end": np.nan,
                         "peak_z": scene_z.max()})
            continue
        doys   = elevated.index.dayofyear
        peak_z = elevated.max()
        log(f"  {year}: {len(elevated)} elevated dates, "
            f"DOY {doys.min()}–{doys.max()}  peak z={peak_z:.3f}  "
            f"dates: {', '.join(str(d.date()) for d in elevated.index)}")
        rows.append({"year": year, "n_dates": len(elevated),
                     "doy_start": int(doys.min()), "doy_end": int(doys.max()),
                     "peak_z": peak_z})

    windows = pd.DataFrame(rows)
    return windows


def build_spatial_peak(df: pd.DataFrame, inf_anomalies: pd.DataFrame,
                        ext_anomalies: pd.DataFrame,
                        windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Per-pixel mean FI_by z-score on the single highest-z acquisition date.

    Uses the date with the highest mean z-score across infestation pixels (from the
    flowering windows) as the spatial snapshot.  This avoids the cancellation problem
    of averaging residuals across a whole DOY bin.
    """
    z_col = f"{PRIMARY_INDEX}_z"

    # Find the single date with the highest mean z-score across infestation pixels
    scene_z = inf_anomalies.groupby("date")[z_col].mean()
    peak_date = scene_z.idxmax()
    peak_z_mean = scene_z.max()
    log(f"\nBuilding spatial map for peak date {peak_date.date()} "
        f"(mean z={peak_z_mean:.3f} across infestation pixels)...")

    coords = df[["point_id", "lon", "lat", "population"]].drop_duplicates("point_id")

    # Infestation: z-score anomaly on peak date
    inf_subset = inf_anomalies[inf_anomalies["date"] == peak_date]
    log(f"  Infestation observations on date: {len(inf_subset):,} across {inf_subset['point_id'].nunique()} pixels")

    inf_spatial = (
        inf_subset[["point_id", z_col]]
        .rename(columns={z_col: "fi_peak_z"})
        .merge(coords, on="point_id", how="left")
    )
    log(f"  fi_peak_z range (infestation): "
        f"{inf_spatial['fi_peak_z'].min():.3f} – {inf_spatial['fi_peak_z'].max():.3f}  "
        f"mean={inf_spatial['fi_peak_z'].mean():.3f}")

    # Extension: z-score anomaly on peak date (same method, own baseline)
    ext_subset = ext_anomalies[ext_anomalies["date"] == peak_date]
    log(f"  Extension observations on date: {len(ext_subset):,} across {ext_subset['point_id'].nunique()} pixels")

    z_col = f"{PRIMARY_INDEX}_z"
    ext_spatial = (
        ext_subset[["point_id", z_col]]
        .rename(columns={z_col: "fi_peak_z"})
        .merge(coords, on="point_id", how="left")
    )
    log(f"  fi_peak_z range (extension):   "
        f"{ext_spatial['fi_peak_z'].min():.3f} – {ext_spatial['fi_peak_z'].max():.3f}  "
        f"mean={ext_spatial['fi_peak_z'].mean():.3f}")

    return inf_spatial, ext_spatial, peak_date


def build_pixel_p90(inf_anomalies: pd.DataFrame, ext_anomalies: pd.DataFrame,
                    ct: pd.DataFrame) -> pd.DataFrame:
    """Per-pixel annual 90th-percentile FI_by z-score, in two variants.

    Unrestricted (fi_p90):
      Annual p90 across all haze-filtered observations.  Asks "how high does this
      pixel's anomaly reach in a typical year?"

    Contrast-gated (fi_p90_cg):
      Annual p90 restricted to dates where the scene-level contrast (infestation mean
      z minus extension mean z) was positive that year.  By conditioning on dates when
      the infestation was genuinely diverging from the extension, this removes wet-season
      greenness dates that lift both populations equally and concentrates the statistic on
      candidate flowering observations.  Mirrors the red-edge approach of identifying the
      window from the contrast series first, then computing the percentile within it —
      but window-free: the "window" is defined per-year by the contrast sign rather than
      a fixed DOY range.
    """
    log("\nComputing per-pixel annual p90 FI_by z-score (unrestricted + contrast-gated)...")
    z_col = f"{PRIMARY_INDEX}_z"

    # Build per-year set of contrast-positive dates
    ct_copy = ct.copy()
    ct_copy["year"] = ct_copy["date"].dt.year
    contrast_pos_dates = set(
        ct_copy.loc[ct_copy["contrast"] > 0, "date"]
    )

    rows = []
    for pop_label, pop_df in [("infestation", inf_anomalies), ("extension", ext_anomalies)]:
        pop_df = pop_df.copy()
        pop_df["year"] = pop_df["date"].dt.year

        # Unrestricted
        annual_p90 = (
            pop_df.groupby(["point_id", "year"])[z_col]
            .quantile(0.90)
            .reset_index()
            .rename(columns={z_col: "fi_p90_year"})
        )
        pixel_p90 = (
            annual_p90.groupby("point_id")["fi_p90_year"]
            .mean()
            .reset_index()
            .rename(columns={"fi_p90_year": "fi_p90"})
        )

        # Contrast-gated: restrict to dates where scene contrast > 0
        cg_df = pop_df[pop_df["date"].isin(contrast_pos_dates)]
        if len(cg_df) > 0:
            annual_p90_cg = (
                cg_df.groupby(["point_id", "year"])[z_col]
                .quantile(0.90)
                .reset_index()
                .rename(columns={z_col: "fi_p90_cg_year"})
            )
            pixel_p90_cg = (
                annual_p90_cg.groupby("point_id")["fi_p90_cg_year"]
                .mean()
                .reset_index()
                .rename(columns={"fi_p90_cg_year": "fi_p90_cg"})
            )
            pixel_p90 = pixel_p90.merge(pixel_p90_cg, on="point_id", how="left")
        else:
            pixel_p90["fi_p90_cg"] = np.nan

        pixel_p90["population"] = pop_label
        rows.append(pixel_p90)

    p90 = pd.concat(rows, ignore_index=True)

    log(f"  {'Population':<14}  {'n':>4}  {'fi_p90 med':>10}  "
        f"{'fi_p90 p10':>10}  {'fi_p90 p90':>10}  "
        f"{'fi_p90_cg med':>13}  {'fi_p90_cg p10':>13}  {'fi_p90_cg p90':>13}")
    for pop_label in ("infestation", "extension"):
        sub  = p90[p90["population"] == pop_label]
        v    = sub["fi_p90"].dropna()
        vcg  = sub["fi_p90_cg"].dropna()
        log(f"  {pop_label:<14}  {len(sub):>4}  "
            f"{v.median():>10.3f}  {v.quantile(0.10):>10.3f}  {v.quantile(0.90):>10.3f}  "
            f"{vcg.median():>13.3f}  {vcg.quantile(0.10):>13.3f}  {vcg.quantile(0.90):>13.3f}")

    # IQR overlap for both variants
    for col, label in [("fi_p90", "unrestricted"), ("fi_p90_cg", "contrast-gated")]:
        inf_v = p90[p90["population"] == "infestation"][col].dropna()
        ext_v = p90[p90["population"] == "extension"][col].dropna()
        inf_iqr = (inf_v.quantile(0.25), inf_v.quantile(0.75))
        ext_iqr = (ext_v.quantile(0.25), ext_v.quantile(0.75))
        overlap = max(0, min(inf_iqr[1], ext_iqr[1]) - max(inf_iqr[0], ext_iqr[0]))
        span    = max(inf_iqr[1], ext_iqr[1]) - min(inf_iqr[0], ext_iqr[0])
        frac    = overlap / span if span > 0 else 1.0
        log(f"  IQR overlap ({label}): "
            f"inf=[{inf_iqr[0]:.3f},{inf_iqr[1]:.3f}]  "
            f"ext=[{ext_iqr[0]:.3f},{ext_iqr[1]:.3f}]  "
            f"overlap fraction={frac:.3f}")

    return p90


def build_riparian_p90(df: pd.DataFrame, ct: pd.DataFrame) -> pd.DataFrame | None:
    """Compute fi_p90_cg for the riparian proxy pixel subset.

    Riparian proxy: extension pixels whose dry-season mean NIR (from the dry-NIR
    stats parquet) is in the top RIPARIAN_NIR_PERCENTILE of all extension pixels.
    These cluster at lat ~-22.765 and outscored the Parkinsonia patch on raw NIR in
    the dry-NIR analysis, flagged there as the primary confounder.

    Returns a DataFrame with columns [point_id, fi_p90, fi_p90_cg, population]
    where population == "riparian_proxy", or None if the parquet is missing.
    """
    if not DRY_NIR_PATH.exists():
        log(f"  WARNING: dry-NIR parquet not found at {DRY_NIR_PATH} — skipping riparian test")
        return None

    dry = pd.read_parquet(DRY_NIR_PATH)
    ext_dry = dry[~dry["in_hd_bbox"]]
    threshold = ext_dry["nir_mean"].quantile(RIPARIAN_NIR_PERCENTILE / 100)
    riparian_ids = set(ext_dry.loc[ext_dry["nir_mean"] >= threshold, "point_id"])
    log(f"\nRiparian proxy selection:")
    log(f"  Extension nir_mean p{RIPARIAN_NIR_PERCENTILE} threshold: {threshold:.4f}")
    log(f"  Riparian proxy pixels selected: {len(riparian_ids)}")

    # Pull time series for these pixels from the main quality-filtered df
    # (df already has FI_by_z from add_indices + compute_pixel_anomalies)
    # We need to re-compute pixel anomalies for these pixels as a standalone population
    rip_df = df[df["point_id"].isin(riparian_ids)].copy()
    if len(rip_df) == 0:
        log("  WARNING: no observations found for riparian proxy pixels — skipping")
        return None

    log(f"  Riparian proxy observations after quality filter: {len(rip_df):,}")

    rip_anomalies = compute_pixel_anomalies(rip_df)
    rip_anomalies["year"] = rip_anomalies["date"].dt.year

    z_col = f"{PRIMARY_INDEX}_z"

    # Contrast-positive dates (same definition as build_pixel_p90)
    ct_copy = ct.copy()
    ct_copy["year"] = ct_copy["date"].dt.year
    contrast_pos_dates = set(ct_copy.loc[ct_copy["contrast"] > 0, "date"])

    # Unrestricted annual p90, then mean across years
    annual_p90 = (
        rip_anomalies.groupby(["point_id", "year"])[z_col]
        .quantile(0.90)
        .reset_index()
        .rename(columns={z_col: "fi_p90_year"})
    )
    pixel_p90 = (
        annual_p90.groupby("point_id")["fi_p90_year"]
        .mean()
        .reset_index()
        .rename(columns={"fi_p90_year": "fi_p90"})
    )

    # Contrast-gated p90
    cg_df = rip_anomalies[rip_anomalies["date"].isin(contrast_pos_dates)]
    if len(cg_df) > 0:
        annual_p90_cg = (
            cg_df.groupby(["point_id", "year"])[z_col]
            .quantile(0.90)
            .reset_index()
            .rename(columns={z_col: "fi_p90_cg_year"})
        )
        pixel_p90_cg = (
            annual_p90_cg.groupby("point_id")["fi_p90_cg_year"]
            .mean()
            .reset_index()
            .rename(columns={"fi_p90_cg_year": "fi_p90_cg"})
        )
        pixel_p90 = pixel_p90.merge(pixel_p90_cg, on="point_id", how="left")
    else:
        pixel_p90["fi_p90_cg"] = np.nan

    pixel_p90["population"] = "riparian_proxy"

    v   = pixel_p90["fi_p90"].dropna()
    vcg = pixel_p90["fi_p90_cg"].dropna()
    log(f"  riparian_proxy  n={len(pixel_p90)}  "
        f"fi_p90 med={v.median():.3f}  p10={v.quantile(0.10):.3f}  p90={v.quantile(0.90):.3f}  "
        f"fi_p90_cg med={vcg.median():.3f}  p10={vcg.quantile(0.10):.3f}  p90={vcg.quantile(0.90):.3f}")

    return pixel_p90


def log_riparian_test(p90: pd.DataFrame, rip_p90: pd.DataFrame) -> None:
    """Three-way fi_p90_cg comparison: infestation vs grassland vs riparian proxy.

    'Grassland' is the extension minus the riparian proxy pixels.  This isolates
    the pure grassland signal from the potential riparian confounder already present
    in the 'extension' population used elsewhere.
    """
    log("\n--- Priority 2: fi_p90_cg riparian test ---")

    rip_ids = set(rip_p90["point_id"])

    # Separate extension into grassland (non-riparian) and riparian
    ext_all  = p90[p90["population"] == "extension"].copy()
    grass    = ext_all[~ext_all["point_id"].isin(rip_ids)]
    inf      = p90[p90["population"] == "infestation"]

    populations = [
        ("infestation",    inf),
        ("grassland",      grass),
        ("riparian_proxy", rip_p90),
    ]

    log(f"\n  {'Population':<16}  {'n':>4}  {'fi_p90_cg med':>13}  "
        f"{'IQR p25':>9}  {'IQR p75':>9}  {'min':>7}  {'max':>7}")
    stats = {}
    for label, sub in populations:
        v = sub["fi_p90_cg"].dropna()
        if len(v) == 0:
            log(f"  {label:<16}  n=0  (no data)")
            continue
        p25, p75 = v.quantile(0.25), v.quantile(0.75)
        stats[label] = {"v": v, "p25": p25, "p75": p75, "med": v.median()}
        log(f"  {label:<16}  {len(v):>4}  {v.median():>13.3f}  "
            f"{p25:>9.3f}  {p75:>9.3f}  {v.min():>7.3f}  {v.max():>7.3f}")

    log("\n  IQR overlap matrix (fi_p90_cg):")
    labels = [l for l, _ in populations if l in stats]
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            sa, sb = stats[a], stats[b]
            overlap = max(0, min(sa["p75"], sb["p75"]) - max(sa["p25"], sb["p25"]))
            span    = max(sa["p75"], sb["p75"]) - min(sa["p25"], sb["p25"])
            frac    = overlap / span if span > 0 else 1.0
            log(f"    {a} vs {b}: "
                f"[{sa['p25']:.3f},{sa['p75']:.3f}] vs [{sb['p25']:.3f},{sb['p75']:.3f}]  "
                f"overlap fraction={frac:.3f}")

    # Key diagnostic: does riparian score high (like re_p10 did) or low (clean separation)?
    if "riparian_proxy" in stats and "infestation" in stats:
        rip_med  = stats["riparian_proxy"]["med"]
        inf_med  = stats["infestation"]["med"]
        inf_p25  = stats["infestation"]["p25"]
        rip_above_inf_iqr = rip_med > inf_p25
        log(f"\n  Riparian median ({rip_med:.3f}) {'>' if rip_above_inf_iqr else '<='} "
            f"infestation IQR p25 ({inf_p25:.3f})")
        if rip_above_inf_iqr:
            log("  → Riparian scores HIGH: fi_p90_cg is redundant with the known riparian confound")
        else:
            log("  → Riparian scores LOW: fi_p90_cg achieves separation from riparian — "
                "candidate reserve feature")

    log("---------------------------------------------------")


def build_band_decomposition(df: pd.DataFrame, inf_anomalies: pd.DataFrame,
                              windows: pd.DataFrame) -> pd.DataFrame:
    """Per-band anomaly (B02, B03, B04, B08) on each elevated acquisition date.

    For each date where mean infestation FI_by z-score ≥ 1.0, compute the difference
    between the observed per-date mean band value across infestation pixels and the
    DOY-bin baseline for those pixels.  Returns a dataframe suitable for plotting.
    """
    log("\nBuilding band decomposition on elevated dates...")
    BANDS = ["B02", "B03", "B04", "B08"]

    inf = df[df["population"] == "infestation"].copy()
    inf["doy"]     = inf["date"].dt.dayofyear
    inf["doy_bin"] = doy_bin(inf["doy"])

    # Per-pixel per-DOY-bin baseline for the four bands
    baseline = (
        inf.groupby(["point_id", "doy_bin"])[BANDS]
        .median()
        .reset_index()
        .rename(columns={b: f"{b}_base" for b in BANDS})
    )
    inf = inf.merge(baseline, on=["point_id", "doy_bin"], how="left")
    for b in BANDS:
        inf[f"{b}_anom"] = inf[b] - inf[f"{b}_base"]

    # Identify elevated dates (mean z-score ≥ 1.0 across infestation pixels)
    z_col = f"{PRIMARY_INDEX}_z"
    scene_z = inf_anomalies.groupby("date")[z_col].mean()
    elevated_dates = scene_z[scene_z >= 1.0].index

    log(f"  Elevated dates (mean z ≥ 1.0): {len(elevated_dates)}")

    rows = []
    for date in sorted(elevated_dates):
        obs = inf[inf["date"] == date]
        if len(obs) == 0:
            continue
        z_mean = scene_z.get(date, np.nan)
        for b in BANDS:
            anom = obs[f"{b}_anom"].mean()
            rows.append({"date": date, "band": b, "anomaly": anom, "z_mean": z_mean})

    decomp = pd.DataFrame(rows)

    # Classify each date's signature
    if len(decomp) > 0:
        pivot = decomp.pivot_table(index="date", columns="band", values="anomaly")
        pivot = pivot.reindex(sorted(elevated_dates))

        n_flowering, n_haze, n_greenness, n_other = 0, 0, 0, 0
        for date, row in pivot.iterrows():
            if row.isna().any():
                n_other += 1
                continue
            b03_up  = row["B03"] > 0
            b04_up  = row["B04"] > 0
            b08_up  = row["B08"] > 0
            # Flowering: B04 and B03 elevated, B02 suppressed *relative to the visible
            # bands*.  Use a ratio test: B02 must be less than half of B04 anomaly.
            # This tolerates small positive B02 noise (< 0.010 after haze filtering)
            # without mis-classifying it as haze.
            b02_suppressed = row["B02"] < row["B04"] * 0.5

            if b03_up and b04_up and b02_suppressed and not b08_up:
                n_flowering += 1
            elif b03_up and b04_up and b02_suppressed and b08_up:
                n_greenness += 1
            elif b03_up and b04_up and not b02_suppressed:
                n_haze += 1
            else:
                n_other += 1

        total = len(pivot)
        log(f"  Signature classification across {total} peak dates:")
        log(f"    Flowering (B03↑ B04↑ B02↓): {n_flowering} ({100*n_flowering/total:.0f}%)")
        log(f"    Haze      (B02↑ B03↑ B04↑): {n_haze}       ({100*n_haze/total:.0f}%)")
        log(f"    Greenness (B03↑ B04↑ B08↑): {n_greenness}  ({100*n_greenness/total:.0f}%)")
        log(f"    Other:                        {n_other}       ({100*n_other/total:.0f}%)")

        frac_flowering = n_flowering / total if total > 0 else 0.0
        log(f"  Fraction showing flowering signature: {frac_flowering:.2f}")

    return decomp


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_doy_profiles(doy_profiles: pd.DataFrame, out_path: Path) -> None:
    """5-panel within-infestation z-score anomaly profiles across DOY bins."""
    n = len(INDICES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)

    for ax, (name, label) in zip(axes, INDICES):
        mean_col = f"{name}_z_mean"
        std_col  = f"{name}_z_std"

        if mean_col not in doy_profiles.columns:
            continue

        z_mean = doy_profiles[mean_col]
        z_std  = doy_profiles[std_col] if std_col in doy_profiles.columns else None
        x      = doy_profiles["doy_bin"]

        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        if z_std is not None:
            ax.fill_between(x, z_mean - z_std, z_mean + z_std,
                            alpha=0.18, color="steelblue", label="±1 std across pixels")
        ax.plot(x, z_mean, color="steelblue", linewidth=1.8, label="Mean z-score anomaly")
        ax.fill_between(x, 0, z_mean, where=z_mean > 0,
                        alpha=0.25, color="gold", label="Above baseline (candidate flowering)")
        ax.fill_between(x, 0, z_mean, where=z_mean < 0,
                        alpha=0.15, color="tomato", label="Below baseline")

        ax.set_ylabel("z-score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        ax.set_title(label, fontsize=8, loc="left")

    axes[-1].set_xlabel("Day of year", fontsize=8)
    axes[-1].set_xticks(doy_profiles["doy_bin"].values[::2])
    axes[-1].tick_params(axis="x", labelsize=7, rotation=45)

    fig.suptitle(
        "Longreach — within-infestation flowering index anomaly (z-score) by DOY\n"
        "Each panel: mean z-score across infestation pixels per 14-day bin (2020–2025 pooled)\n"
        "Positive = above pixel's own seasonal baseline → candidate flowering flash",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved DOY profiles: {out_path.relative_to(PROJECT_ROOT)}")


def plot_timeseries(ts: pd.DataFrame, out_path: Path) -> None:
    """Raw FI_by scene-mean (both populations) + infestation z-score anomaly."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    if "infestation" in ts.columns:
        ax1.plot(ts["date"], ts["infestation"], color="forestgreen", linewidth=0.9,
                 alpha=0.85, label="Infestation (raw FI_by)")
    if "extension" in ts.columns:
        ax1.plot(ts["date"], ts["extension"], color="peru", linewidth=0.9,
                 alpha=0.85, linestyle="--", label="Extension (raw FI_by)")

    ax1.set_ylabel("FI_by  (B03+B04)/(B02+B08)", fontsize=8)
    ax1.legend(fontsize=8, framealpha=0.7)
    ax1.tick_params(labelsize=7)
    ax1.set_title(
        "Longreach — FI_by scene-mean time series 2020–2025\n"
        "Each point = mean across all pixels in that population for one S2 acquisition",
        fontsize=9,
    )

    # Lower panel: within-infestation z-score anomaly
    if "inf_z" in ts.columns:
        ax2.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        ax2.fill_between(ts["date"], 0, ts["inf_z"],
                         where=ts["inf_z"] > 0, alpha=0.55, color="gold",
                         label="Above pixel baseline (candidate flowering)")
        ax2.fill_between(ts["date"], 0, ts["inf_z"],
                         where=ts["inf_z"] < 0, alpha=0.35, color="steelblue",
                         label="Below pixel baseline")
        ax2.plot(ts["date"], ts["inf_z"], color="darkgoldenrod", linewidth=0.7, alpha=0.7)
        ax2.axhline(1,  color="gold",     linewidth=0.8, linestyle="--", alpha=0.7)
        ax2.axhline(-1, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.7)

    ax2.set_ylabel("Mean z-score\n(infestation pixels)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.7)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax2.set_xlabel("Date", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved time series: {out_path.relative_to(PROJECT_ROOT)}")


def plot_spatial_peak(inf_spatial: pd.DataFrame, ext_spatial: pd.DataFrame,
                      peak_date, out_path: Path,
                      bg_img: np.ndarray | None) -> None:
    """Side-by-side spatial maps: infestation (left) and extension (right).

    Both panels use the same RdYlGn z-score scale anchored to the infestation
    p02–p98 range, so the two populations are directly comparable.
    Each panel shows its own population's pixels only, with the WMS aerial
    as background.
    """
    lon_min, lat_min, lon_max, lat_max = SURVEY_BBOX

    # Infestation occupies north half of survey bbox; extension occupies south half
    inf_lat_min = INFESTATION_LAT_MIN
    inf_lat_max = lat_max
    ext_lat_min = lat_min
    ext_lat_max = INFESTATION_LAT_MIN

    lat_centre    = (lat_min + lat_max) / 2
    lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
    lat_m_per_deg = 111320
    lon_span      = lon_max - lon_min

    # Each panel height proportional to its lat span
    inf_lat_span = inf_lat_max - inf_lat_min
    ext_lat_span = ext_lat_max - ext_lat_min
    panel_w      = 5.5
    inf_h        = panel_w * (inf_lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg)
    ext_h        = panel_w * (ext_lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg)

    # Symmetric colour scale anchored to infestation p02–p98
    inf_z = inf_spatial["fi_peak_z"].dropna()
    vlim  = max(abs(inf_z.quantile(0.02)), abs(inf_z.quantile(0.98)), 1.0)

    # Marker size scaled to 10 m pixel at panel width
    pt_per_deg = panel_w * 72 / lon_span
    marker_pt  = (10 / lon_m_per_deg) * pt_per_deg
    marker_s   = max(1.0, marker_pt ** 2 / 8)

    fig, (ax_inf, ax_ext) = plt.subplots(
        1, 2,
        figsize=(panel_w * 2 + 1.5, max(inf_h, ext_h) + 1.2),
        dpi=150,
    )

    for ax, spatial, bbox_lat_min, bbox_lat_max, title in [
        (ax_inf, inf_spatial, inf_lat_min, inf_lat_max,
         f"Infestation (north)\nmean z = {inf_spatial['fi_peak_z'].mean():.2f}"),
        (ax_ext, ext_spatial, ext_lat_min, ext_lat_max,
         f"Extension / grassland (south)\nmean z = {ext_spatial['fi_peak_z'].mean():.2f}"),
    ]:
        if bg_img is not None:
            # Slice the WMS image to this panel's lat range
            full_lat_span = lat_max - lat_min
            img_h = bg_img.shape[0]
            # WMS origin="upper": row 0 = lat_max, row img_h = lat_min
            row_top    = int((lat_max - bbox_lat_max) / full_lat_span * img_h)
            row_bottom = int((lat_max - bbox_lat_min) / full_lat_span * img_h)
            row_top    = max(0, row_top)
            row_bottom = min(img_h, row_bottom)
            panel_bg   = bg_img[row_top:row_bottom, :, :]
            ax.imshow(
                panel_bg,
                extent=[lon_min, lon_max, bbox_lat_min, bbox_lat_max],
                origin="upper", aspect="auto", interpolation="bilinear", zorder=0,
            )

        sc = ax.scatter(
            spatial["lon"], spatial["lat"],
            c=spatial["fi_peak_z"], cmap="RdYlGn",
            s=marker_s, linewidths=0.0, alpha=0.75, zorder=2,
            vmin=-vlim, vmax=vlim,
        )

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(bbox_lat_min, bbox_lat_max)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
        ax.set_title(title, fontsize=8)

    # Infestation bbox outline on left panel
    ax_inf.add_patch(mpatches.Rectangle(
        (HD_LON_MIN, HD_LAT_MIN),
        HD_LON_MAX - HD_LON_MIN,
        HD_LAT_MAX - HD_LAT_MIN,
        fill=False, edgecolor="white", linewidth=1.0, linestyle="--",
        label="High-density bbox", zorder=3,
    ))
    ax_inf.legend(loc="lower right", fontsize=6, framealpha=0.7,
                  facecolor="black", labelcolor="white", edgecolor="none")

    # Shared colorbar
    cb = fig.colorbar(sc, ax=[ax_inf, ax_ext], fraction=0.02, pad=0.02, aspect=30)
    cb.set_label("FI_by z-score anomaly (above own seasonal baseline)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Longreach — FI_by z-score anomaly on peak date {pd.Timestamp(peak_date).date()}\n"
        f"Uniform scale ±{vlim:.1f} σ — positive (green) = above pixel's own seasonal baseline",
        fontsize=9, y=1.01,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved spatial peak map: {out_path.relative_to(PROJECT_ROOT)}")


def plot_contrast_timeseries(ct: pd.DataFrame, out_path: Path) -> None:
    """Two-panel contrast time series + DOY profile, mirroring the red-edge contrast plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                   gridspec_kw={"height_ratios": [3, 2]})

    # Upper panel: per-date contrast scatter + 30-day rolling mean
    pos = ct["contrast"] > 0
    ax1.scatter(ct.loc[pos,  "date"], ct.loc[pos,  "contrast"],
                s=8, color="forestgreen", alpha=0.6, zorder=2, label="Contrast > 0 (inf above ext)")
    ax1.scatter(ct.loc[~pos, "date"], ct.loc[~pos, "contrast"],
                s=8, color="tomato",      alpha=0.6, zorder=2, label="Contrast ≤ 0 (ext above inf)")
    ax1.plot(ct["date"], ct["contrast_30d"], color="black", linewidth=1.4,
             alpha=0.8, label="30-day rolling mean", zorder=3)
    ax1.axhline(0, color="grey", linewidth=0.7, linestyle=":")

    frac_pos = (ct["contrast"] > 0).mean()
    ax1.set_ylabel("FI_by z-score contrast\n(infestation − extension)", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.legend(fontsize=7, framealpha=0.7)
    ax1.set_title(
        f"Longreach — FI_by inter-population z-score contrast 2020–2025  "
        f"(haze-filtered)\n"
        f"Positive = infestation above its own baseline more than extension is above its own  |  "
        f"Fraction > 0: {frac_pos:.2f}",
        fontsize=9,
    )

    # Lower panel: mean contrast by 14-day DOY bin
    ct2 = ct.copy()
    ct2["doy_bin"] = ((ct2["date"].dt.dayofyear - 1) // DOY_BIN_DAYS) * DOY_BIN_DAYS + 1
    doy_ct = ct2.groupby("doy_bin")["contrast"].agg(["mean", "std"]).reset_index()

    ax2.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax2.fill_between(doy_ct["doy_bin"],
                     doy_ct["mean"] - doy_ct["std"],
                     doy_ct["mean"] + doy_ct["std"],
                     alpha=0.18, color="steelblue")
    ax2.plot(doy_ct["doy_bin"], doy_ct["mean"], color="steelblue",
             linewidth=1.8, label="Mean contrast ± 1 std")
    ax2.fill_between(doy_ct["doy_bin"], 0, doy_ct["mean"],
                     where=doy_ct["mean"] > 0, alpha=0.3, color="forestgreen")
    ax2.fill_between(doy_ct["doy_bin"], 0, doy_ct["mean"],
                     where=doy_ct["mean"] < 0, alpha=0.2, color="tomato")

    ax2.set_xlabel("Day of year", fontsize=8)
    ax2.set_ylabel("Mean contrast\n(pooled years)", fontsize=8)
    ax2.set_xticks(doy_ct["doy_bin"].values[::2])
    ax2.tick_params(labelsize=7, axis="x", rotation=45)
    ax2.legend(fontsize=7, framealpha=0.7)
    ax2.set_title("DOY profile of contrast (all years pooled)", fontsize=8, loc="left")

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved contrast time series: {out_path.relative_to(PROJECT_ROOT)}")


def plot_band_decomposition(decomp: pd.DataFrame, out_path: Path) -> None:
    """Small-multiple bar chart: per-band anomaly on each elevated acquisition date.

    One panel per date.  Four bars (B02, B03, B04, B08) show (observed − DOY-bin
    baseline) for infestation pixels.  Bars coloured by sign: positive = orange,
    negative = steelblue.  Expected flowering signature: B03 and B04 positive, B02
    negative, B08 near zero.
    """
    if len(decomp) == 0:
        log("  No elevated dates — skipping band decomposition plot")
        return

    dates  = sorted(decomp["date"].unique())
    n      = len(dates)
    ncols  = min(4, n)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8),
                              squeeze=False)

    BANDS       = ["B02", "B03", "B04", "B08"]
    BAND_COLORS = {b: ("tomato" if b != "B02" else "steelblue") for b in BANDS}

    for i, date in enumerate(dates):
        ax  = axes[i // ncols][i % ncols]
        sub = decomp[decomp["date"] == date]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        pivot = sub.set_index("band")["anomaly"]
        z_mean = sub["z_mean"].iloc[0]

        vals   = [pivot.get(b, np.nan) for b in BANDS]
        colors = ["tomato" if v >= 0 else "steelblue" for v in vals]
        ax.bar(BANDS, vals, color=colors, width=0.6, zorder=2)
        ax.axhline(0, color="black", linewidth=0.6, zorder=3)
        ax.set_title(f"{pd.Timestamp(date).date()}\nz={z_mean:.2f}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("Δ reflectance\n(obs − baseline)", fontsize=6)

        # Annotate expected vs confound
        b02 = pivot.get("B02", np.nan)
        b03 = pivot.get("B03", np.nan)
        b04 = pivot.get("B04", np.nan)
        b08 = pivot.get("B08", np.nan)
        if not any(np.isnan(v) for v in [b02, b03, b04, b08]):
            b02_suppressed = b02 < b04 * 0.5
            if b03 > 0 and b04 > 0 and b02_suppressed and b08 <= 0:
                label, col = "flowering", "forestgreen"
            elif b03 > 0 and b04 > 0 and b02_suppressed and b08 > 0:
                label, col = "greenness", "steelblue"
            elif b03 > 0 and b04 > 0 and not b02_suppressed:
                label, col = "haze", "darkorange"
            else:
                label, col = "other", "grey"
            ax.text(0.02, 0.97, label, transform=ax.transAxes,
                    fontsize=6, color=col, va="top", fontweight="bold")

    # Hide unused panels
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="tomato",    label="Positive anomaly"),
        Patch(facecolor="steelblue", label="Negative anomaly"),
    ]
    fig.legend(handles=legend_elements, fontsize=7, loc="lower right", framealpha=0.7)

    fig.suptitle(
        "Longreach — per-band reflectance anomaly on elevated FI_by dates\n"
        "Observed − DOY-bin median baseline, infestation pixels\n"
        "Flowering expected: B03↑ B04↑ B02↓ B08≈0  |  Haze: all bands↑  |  Greenness: B03↑ B04↑ B08↑",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved band decomposition: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

def log_success_criteria(inf_anomalies: pd.DataFrame, windows: pd.DataFrame,
                          inf_spatial: pd.DataFrame,
                          decomp: pd.DataFrame) -> None:
    log("\n--- Success criteria (research/LONGREACH-FLOWERING.md) ---")

    z_col = f"{PRIMARY_INDEX}_z"

    # 1. Peak date z-score ≥ 2.0 — at least one acquisition date has mean FI_by z-score
    #    ≥ 2.0 across infestation pixels
    scene_z   = inf_anomalies.groupby("date")[z_col].mean()
    peak_date = scene_z.idxmax()
    peak_z    = scene_z.max()
    status1   = "PASS" if peak_z >= 2.0 else "FAIL"
    log(f"  [1] Peak acquisition date mean z ≥ 2.0: "
        f"z = {peak_z:.3f} on {peak_date.date()}  → {status1}")

    # 2. Recurs in ≥ 3 of 6 years — at least 3 years have ≥ 1 date with mean z ≥ 1.0
    n_years   = (windows["n_dates"] > 0).sum()
    status2   = "PASS" if n_years >= 3 else "FAIL"
    log(f"  [2] Elevated date (mean z ≥ 1.0) in ≥ 3 years: "
        f"{n_years} qualifying years  → {status2}")

    # 3. Band decomposition confirms flowering mechanism — majority of peak dates show
    #    B03↑ B04↑ B02↓ (not uniform broadband increase)
    status3 = "N/A"
    if len(decomp) > 0:
        dates = sorted(decomp["date"].unique())
        pivot = decomp.pivot_table(index="date", columns="band", values="anomaly")
        n_flowering = 0
        n_total     = 0
        for date in dates:
            if date not in pivot.index:
                continue
            row = pivot.loc[date]
            if row.isna().any():
                continue
            n_total += 1
            b02_suppressed = row.get("B02", np.nan) < row.get("B04", np.nan) * 0.5
            if row.get("B03", np.nan) > 0 and row.get("B04", np.nan) > 0 and b02_suppressed:
                n_flowering += 1
        frac = n_flowering / n_total if n_total > 0 else 0.0
        status3 = "PASS" if frac > 0.5 else "FAIL"
        log(f"  [3] Band decomposition: flowering signature on {n_flowering}/{n_total} dates "
            f"({100*frac:.0f}%)  → {status3}")
    else:
        log(f"  [3] Band decomposition: no elevated dates — skipped")

    # 4. Spatial coherence r ≥ 0.5 on peak date (infestation pixels)
    from scipy.spatial import cKDTree
    coords = inf_spatial[["lon", "lat"]].values
    tree   = cKDTree(coords)
    _, idxs = tree.query(coords, k=9)   # self + 8 neighbours
    neighbour_means = inf_spatial["fi_peak_z"].values[idxs[:, 1:]].mean(axis=1)
    r = np.corrcoef(inf_spatial["fi_peak_z"].values, neighbour_means)[0, 1]
    status4 = "PASS" if r >= 0.5 else "FAIL"
    log(f"  [4] Spatial coherence (Pearson r with 8-neighbour mean): "
        f"r = {r:.3f}  → {status4}")

    log("----------------------------------------------------------")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== Longreach flowering flash detection ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(PARQUET_PATH)
    df = add_indices(df)

    doy_profiles, inf_anomalies, ext_anomalies = build_doy_profiles(df)
    ts                          = build_timeseries(df, inf_anomalies)
    ct                          = build_contrast_timeseries(inf_anomalies, ext_anomalies)
    windows                     = build_flowering_windows(inf_anomalies)
    inf_spatial, ext_spatial, peak_date = build_spatial_peak(df, inf_anomalies, ext_anomalies, windows)

    p90   = build_pixel_p90(inf_anomalies, ext_anomalies, ct)
    decomp = build_band_decomposition(df, inf_anomalies, windows)

    # Priority 2: riparian proxy test
    rip_p90 = build_riparian_p90(df, ct)

    # Save tabular outputs
    windows_path = OUT_DIR / "flowering_window_by_year.csv"
    windows.to_csv(windows_path, index=False)
    log(f"\nSaved flowering windows: {windows_path.relative_to(PROJECT_ROOT)}")

    p90_path = OUT_DIR / "fi_p90_per_pixel.csv"
    p90.to_csv(p90_path, index=False)
    log(f"Saved fi_p90 per pixel: {p90_path.relative_to(PROJECT_ROOT)}")

    if rip_p90 is not None:
        rip_path = OUT_DIR / "fi_p90_riparian_proxy.csv"
        rip_p90.to_csv(rip_path, index=False)
        log(f"Saved riparian proxy fi_p90: {rip_path.relative_to(PROJECT_ROOT)}")

    # Fetch WMS background
    log("\nFetching Queensland Globe WMS background tile...")
    try:
        bg_img = fetch_wms_image(SURVEY_BBOX, width_px=2048)
        log(f"  Background tile: {bg_img.shape[1]}×{bg_img.shape[0]} px for bbox {SURVEY_BBOX}")
    except Exception as exc:
        log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without background")
        bg_img = None

    log("\nGenerating plots...")
    plot_doy_profiles(doy_profiles,   OUT_DIR / "fi_doy_profiles.png")
    plot_timeseries(ts,                OUT_DIR / "fi_by_timeseries.png")
    plot_contrast_timeseries(ct,       OUT_DIR / "fi_by_contrast.png")
    plot_spatial_peak(inf_spatial, ext_spatial, peak_date,
                      OUT_DIR / "fi_by_spatial_peak.png", bg_img)
    plot_band_decomposition(decomp,   OUT_DIR / "fi_band_decomposition.png")

    log_success_criteria(inf_anomalies, windows, inf_spatial, decomp)

    if rip_p90 is not None:
        log_riparian_test(p90, rip_p90)

    log("\nDone.")


if __name__ == "__main__":
    main()
