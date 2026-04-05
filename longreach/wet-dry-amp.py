"""Wet/dry seasonal amplitude analysis over the Longreach Parkinsonia infestation.

Metric: NDVI seasonal recession = post-wet NDVI peak (Mar–May median) minus
dry-season trough (Jul–Sep median), computed per pixel per year, then averaged.

The hypothesis is *inverted* from a naive wet/dry B08 swing:
  - Parkinsonia: deep-rooted, sustains active chlorophyll into Apr–May (strong
    post-wet peak), then recedes as days shorten → HIGH recession
  - Riparian / grassland: lower post-wet NDVI, flatter year-round → LOW recession

Combined with dry-season NIR CV from the previous analysis, this gives a 2D
discriminator:
  Parkinsonia  → low nir_cv  + HIGH rec_mean  (bottom-right)
  Grassland    → high nir_cv + low  rec_mean  (top-left)
  Riparian     → moderate CV + low  rec_mean  (top-left, lower CV than grassland)

Outputs:
  outputs/longreach-wet-dry-amp/longreach_amp_stats.parquet
  outputs/longreach-wet-dry-amp/longreach_rec_map.png
  outputs/longreach-wet-dry-amp/longreach_rec_hist.png
  outputs/longreach-wet-dry-amp/longreach_nir_cv_vs_rec.png
  outputs/longreach-wet-dry-amp/longreach_monthly_profiles.png

See research/LONGREACH-WET-DRY-AMP.md for the analysis plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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

PARQUET_PATH  = PROJECT_ROOT / "data" / "longreach_pixels.parquet"
DRY_NIR_STATS = PROJECT_ROOT / "outputs" / "longreach-dry-nir" / "longreach_dry_nir_stats.parquet"
OUT_DIR       = PROJECT_ROOT / "outputs" / "longreach-wet-dry-amp"

# Post-wet green peak: grasses and Parkinsonia at their annual maximum
PEAK_MONTHS   = {3, 4, 5}      # March–May

# Dry-season trough: lowest canopy greenness
TROUGH_MONTHS = {7, 8, 9}      # July–September

# Minimum qualifying observations per (pixel, season, year)
MIN_OBS = 5

# Quality filter
SCL_PURITY_MIN = 0.5

# Infestation patch bbox
HD_LON_MIN, HD_LON_MAX = 145.423948, 145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033, -22.761054

# Full survey area
SURVEY_BBOX = [145.423948, -22.767104, 145.424956, -22.761054]

# Riparian proxy: top-10% nir_mean from dry-NIR results (~75 pixels near the
# water feature at lat -22.765)
RIPARIAN_PERCENTILE = 0.90

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def load_and_filter(path: Path) -> pd.DataFrame:
    log(f"Loading parquet: {path}")
    df = pd.read_parquet(path)
    log(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    log(f"  Pixels: {df['point_id'].nunique()}  |  "
        f"Date range: {df['date'].min().date()} – {df['date'].max().date()}")

    before = len(df)
    df = df[df["scl_purity"] >= SCL_PURITY_MIN].copy()
    dropped = before - len(df)
    log(f"  Quality filter (scl_purity ≥ {SCL_PURITY_MIN}): "
        f"dropped {dropped:,} rows ({100*dropped/before:.1f}%), "
        f"retained {len(df):,}")

    df["ndvi"]  = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    return df


def seasonal_ndvi_medians(df: pd.DataFrame, months: set[int],
                          season_name: str) -> pd.DataFrame:
    """Return per-(point_id, year) NDVI medians for the given month set."""
    subset = df[df["month"].isin(months)]
    counts = subset.groupby(["point_id", "year"])["ndvi"].count()

    valid = counts[counts >= MIN_OBS].index
    n_excluded = len(counts) - len(valid)
    log(f"  {season_name} (months {sorted(months)}): "
        f"{len(subset):,} rows, {len(counts):,} (pixel×year) groups, "
        f"{n_excluded} excluded (< {MIN_OBS} obs)")

    subset_valid = subset.set_index(["point_id", "year"]).loc[valid].reset_index()
    medians = (
        subset_valid
        .groupby(["point_id", "year"])["ndvi"]
        .median()
        .rename(f"ndvi_{season_name}")
        .reset_index()
    )
    log(f"    Annual medians: {len(medians):,}  |  "
        f"range [{medians[f'ndvi_{season_name}'].min():.4f}, "
        f"{medians[f'ndvi_{season_name}'].max():.4f}]")
    return medians


def compute_recession(peak_med: pd.DataFrame, trough_med: pd.DataFrame) -> pd.DataFrame:
    """Paired join of peak and trough medians; compute recession per (pixel, year)."""
    paired = peak_med.merge(trough_med, on=["point_id", "year"], how="inner")
    log(f"\nPaired (pixel, year) rows: {len(paired):,}  "
        f"(of {len(peak_med):,} peak × {len(trough_med):,} trough)")
    paired["recession"] = paired["ndvi_peak"] - paired["ndvi_trough"]
    log(f"  Recession — min: {paired['recession'].min():.4f}  "
        f"median: {paired['recession'].median():.4f}  "
        f"max: {paired['recession'].max():.4f}")
    neg = (paired["recession"] < 0).sum()
    if neg:
        log(f"  WARNING: {neg} rows with negative recession "
            f"(trough > peak — likely cloud-affected peaks or anomalous years)")
    return paired


def per_pixel_stats(paired: pd.DataFrame, df: pd.DataFrame,
                    dry_nir: pd.DataFrame) -> pd.DataFrame:
    log("\nAggregating per-pixel recession statistics across years...")

    stats = (
        paired
        .groupby("point_id")["recession"]
        .agg(rec_mean="mean", rec_std="std", n_years="count")
        .reset_index()
    )
    stats["rec_cv"] = stats["rec_std"] / stats["rec_mean"].abs()

    coords = df[["point_id", "lon", "lat"]].drop_duplicates("point_id")
    stats = stats.merge(coords, on="point_id", how="left")
    stats = stats.merge(dry_nir[["point_id", "nir_cv", "nir_mean", "in_hd_bbox"]],
                        on="point_id", how="left")

    # Riparian proxy label (top-RIPARIAN_PERCENTILE by nir_mean, not in HD bbox)
    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    stats["is_riparian"] = (~stats["in_hd_bbox"]) & (stats["nir_mean"] >= rip_thresh)
    stats["is_grassland"] = (~stats["in_hd_bbox"]) & (~stats["is_riparian"])

    log(f"  Pixels: {len(stats)}")
    log(f"  n_years per pixel — min: {stats['n_years'].min()}  "
        f"median: {stats['n_years'].median():.0f}  max: {stats['n_years'].max()}")
    log(f"  rec_mean — min: {stats['rec_mean'].min():.4f}  "
        f"median: {stats['rec_mean'].median():.4f}  "
        f"max: {stats['rec_mean'].max():.4f}")
    log(f"  rec_cv   — min: {stats['rec_cv'].min():.4f}  "
        f"median: {stats['rec_cv'].median():.4f}")

    # Class breakdown
    log(f"\n  Class counts:")
    log(f"    Infestation (Parkinsonia proxy): {stats['in_hd_bbox'].sum()}")
    log(f"    Riparian (top-{100*(1-RIPARIAN_PERCENTILE):.0f}% nir_mean, "
        f"not HD bbox):  {stats['is_riparian'].sum()}")
    log(f"    Grassland (remainder):          {stats['is_grassland'].sum()}")

    log("\n  Class centroids in (nir_cv, rec_mean) space:")
    for label, mask in [("Infestation", stats["in_hd_bbox"]),
                        ("Riparian",    stats["is_riparian"]),
                        ("Grassland",   stats["is_grassland"])]:
        sub = stats[mask]
        log(f"    {label:12s}: nir_cv={sub['nir_cv'].mean():.4f}  "
            f"rec_mean={sub['rec_mean'].mean():.4f}  "
            f"rec_mean median={sub['rec_mean'].median():.4f}")

    return stats


def save_stats(stats: pd.DataFrame) -> None:
    out_path = OUT_DIR / "longreach_amp_stats.parquet"
    cols = ["point_id", "lon", "lat", "rec_mean", "rec_std", "rec_cv", "n_years",
            "nir_cv", "nir_mean", "in_hd_bbox", "is_riparian", "is_grassland"]
    stats[cols].to_parquet(out_path, index=False)
    log(f"\nSaved stats: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _scatter_base(stats: pd.DataFrame, col: str, cmap: str, label: str,
                  title: str, bg_img: np.ndarray | None) -> tuple:
    lon_min, lat_min, lon_max, lat_max = SURVEY_BBOX
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lat_centre = (lat_min + lat_max) / 2
    lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
    lat_m_per_deg = 111320
    fig_w = 7
    fig_h = max(4.0, fig_w * (lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    if bg_img is not None:
        ax.imshow(bg_img, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin="upper", aspect="auto", interpolation="bilinear", zorder=0)

    pt_per_deg = fig_w * 72 / lon_span
    marker_pt  = (10 / lon_m_per_deg) * pt_per_deg
    marker_s   = max(0.6, marker_pt ** 2 / 10)

    sc = ax.scatter(stats["lon"], stats["lat"], c=stats[col], cmap=cmap,
                    s=marker_s, linewidths=0, alpha=0.6, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(label, fontsize=8)

    ax.add_patch(mpatches.Rectangle(
        (HD_LON_MIN, HD_LAT_MIN), HD_LON_MAX - HD_LON_MIN, HD_LAT_MAX - HD_LAT_MIN,
        fill=False, edgecolor="white", linewidth=1.2, linestyle="--",
        label="Infestation patch", zorder=4,
    ))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.7,
              facecolor="black", labelcolor="white", edgecolor="none")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_recession_map(stats: pd.DataFrame, bg_img: np.ndarray | None) -> None:
    fig, _ = _scatter_base(
        stats, col="rec_mean", cmap="RdYlGn",
        label="Mean NDVI recession (peak − trough)",
        title="Longreach — NDVI seasonal recession (post-wet peak Mar–May minus dry trough Jul–Sep)\n"
              "Higher = stronger green flush then recession = Parkinsonia signature",
        bg_img=bg_img,
    )
    out = OUT_DIR / "longreach_rec_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved recession map: {out.relative_to(PROJECT_ROOT)}")


def plot_recession_histogram(stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    colors = {"Infestation": "darkorange", "Riparian": "steelblue", "Grassland": "olivedrab"}
    groups = [
        ("Infestation", stats[stats["in_hd_bbox"]],   colors["Infestation"]),
        ("Riparian",    stats[stats["is_riparian"]],   colors["Riparian"]),
        ("Grassland",   stats[stats["is_grassland"]],  colors["Grassland"]),
    ]

    bins = np.linspace(stats["rec_mean"].min() - 0.005,
                       stats["rec_mean"].max() + 0.005, 35)
    for label, sub, color in groups:
        ax.hist(sub["rec_mean"], bins=bins, alpha=0.65, color=color,
                edgecolor="white", linewidth=0.4,
                label=f"{label} (n={len(sub)}, med={sub['rec_mean'].median():.3f})")

    ax.set_xlabel("Mean NDVI recession (Mar–May peak − Jul–Sep trough)", fontsize=9)
    ax.set_ylabel("Pixel count", fontsize=9)
    ax.set_title("Distribution of NDVI seasonal recession by land-cover class\n"
                 "Higher = stronger green peak then dry-season senescence", fontsize=10)
    ax.legend(fontsize=8)
    ax.axvline(stats["rec_mean"].median(), color="black", linestyle=":",
               linewidth=1.2, label="_nolegend_")
    ax.text(stats["rec_mean"].median() + 0.001, ax.get_ylim()[1] * 0.92,
            f"dataset median\n{stats['rec_mean'].median():.3f}",
            fontsize=7, va="top")

    fig.tight_layout()
    out = OUT_DIR / "longreach_rec_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"Saved recession histogram: {out.relative_to(PROJECT_ROOT)}")


def plot_2d_scatter(stats: pd.DataFrame) -> None:
    """nir_cv × rec_mean two-axis discriminator."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    groups = [
        ("Grassland",   stats[stats["is_grassland"]],  "olivedrab",  10, 0.45),
        ("Riparian",    stats[stats["is_riparian"]],    "steelblue",  18, 0.65),
        ("Infestation", stats[stats["in_hd_bbox"]],     "darkorange", 22, 0.80),
    ]

    for label, sub, color, size, alpha in groups:
        ax.scatter(sub["nir_cv"], sub["rec_mean"],
                   c=color, s=size, alpha=alpha, linewidths=0,
                   label=f"{label} (n={len(sub)})", zorder=3)

    # Annotate class centroids
    for label, sub, color, *_ in groups:
        cx, cy = sub["nir_cv"].mean(), sub["rec_mean"].mean()
        ax.scatter([cx], [cy], c=color, s=80, edgecolors="white",
                   linewidths=1.2, zorder=5)
        ax.annotate(f"{label}\n({cx:.3f}, {cy:.3f})",
                    xy=(cx, cy), xytext=(8, 6), textcoords="offset points",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # Quadrant guide lines at dataset medians
    med_cv  = stats["nir_cv"].median()
    med_rec = stats["rec_mean"].median()
    ax.axvline(med_cv,  color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(med_rec, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(med_cv + 0.001, ax.get_ylim()[0] + 0.001,
            f"CV median\n{med_cv:.3f}", fontsize=7, color="grey", va="bottom")
    ax.text(ax.get_xlim()[0] + 0.001, med_rec + 0.001,
            f"rec median {med_rec:.3f}", fontsize=7, color="grey", va="bottom")

    ax.set_xlabel("Dry-season NIR CV (inter-annual stability; lower = more stable)",
                  fontsize=9)
    ax.set_ylabel("Mean NDVI seasonal recession (Mar–May peak − Jul–Sep trough;\n"
                  "higher = stronger green flush then recession)", fontsize=9)
    ax.set_title("Two-axis NIR/NDVI discriminator\n"
                 "Infestation target: low CV + high recession (bottom-right quadrant)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / "longreach_nir_cv_vs_rec.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved 2D scatter: {out.relative_to(PROJECT_ROOT)}")


def plot_monthly_profiles(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Mean NDVI monthly profile per class, with ±1 std band and per-year lines."""
    fig = plt.figure(figsize=(14, 5), dpi=150)
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    groups = [
        ("Infestation", stats[stats["in_hd_bbox"]]["point_id"],   "darkorange"),
        ("Riparian",    stats[stats["is_riparian"]]["point_id"],   "steelblue"),
        ("Grassland",   stats[stats["is_grassland"]]["point_id"],  "olivedrab"),
    ]

    months_ordered = list(range(1, 13))
    month_labels   = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    for i, (label, pids, color) in enumerate(groups):
        ax  = fig.add_subplot(gs[i])
        sub = df[df["point_id"].isin(pids)]

        # Per-year monthly mean NDVI
        yr_month = sub.groupby(["year", "month"])["ndvi"].mean().unstack("month")
        for yr in yr_month.index:
            vals = [yr_month.loc[yr, m] if m in yr_month.columns else np.nan
                    for m in months_ordered]
            ax.plot(months_ordered, vals, color=color, alpha=0.25, linewidth=0.8)

        # Mean ± 1 std across years
        mean_profile = yr_month.mean()
        std_profile  = yr_month.std()
        ym = [mean_profile.get(m, np.nan) for m in months_ordered]
        ys = [std_profile.get(m, np.nan)  for m in months_ordered]
        ym_arr = np.array(ym, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        ax.plot(months_ordered, ym_arr, color=color, linewidth=2, zorder=4)
        ax.fill_between(months_ordered,
                        ym_arr - ys_arr, ym_arr + ys_arr,
                        color=color, alpha=0.2)

        # Highlight peak and trough windows
        for months_set, shade in [(PEAK_MONTHS, "limegreen"), (TROUGH_MONTHS, "tomato")]:
            for m in months_set:
                ax.axvspan(m - 0.5, m + 0.5, color=shade, alpha=0.08, zorder=0)

        ax.set_xticks(months_ordered)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_xlabel("Month", fontsize=8)
        ax.set_ylabel("NDVI" if i == 0 else "", fontsize=8)
        ax.set_title(f"{label}\n(n={len(pids)} pixels)", fontsize=9)
        ax.tick_params(labelsize=7)

        # Annotate recession magnitude
        if len(pids) > 0:
            rec_med = stats.loc[stats["point_id"].isin(pids), "rec_mean"].median()
            ax.text(0.97, 0.04, f"recession median: {rec_med:.3f}",
                    transform=ax.transAxes, fontsize=7.5, ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

    fig.suptitle(
        "Monthly NDVI profiles by class — Longreach 2020–2025\n"
        "Green shading = peak window (Mar–May), red = trough (Jul–Sep)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "longreach_monthly_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved monthly profiles: {out.relative_to(PROJECT_ROOT)}")


def log_success_criteria(stats: pd.DataFrame) -> None:
    log("\n--- Success criteria ---")

    hd      = stats[stats["in_hd_bbox"]]
    non_hd  = stats[~stats["in_hd_bbox"]]
    rip     = stats[stats["is_riparian"]]
    grass   = stats[stats["is_grassland"]]
    med_rec = stats["rec_mean"].median()

    # 1. Infestation above dataset median recession
    n_above = (hd["rec_mean"] > med_rec).sum()
    pct = 100 * n_above / len(hd)
    status = "PASS" if pct >= 60 else "FAIL"
    log(f"  [1] Infestation pixels above dataset median recession "
        f"({med_rec:.4f}): {n_above}/{len(hd)} ({pct:.0f}%)  → {status}")

    # 2. IQR separation: infestation vs combined other classes
    hd_q25, hd_q75   = hd["rec_mean"].quantile([0.25, 0.75])
    ot_q25, ot_q75   = non_hd["rec_mean"].quantile([0.25, 0.75])
    overlap = max(0, min(hd_q75, ot_q75) - max(hd_q25, ot_q25))
    span    = max(hd_q75, ot_q75) - min(hd_q25, ot_q25)
    overlap_frac = overlap / span if span > 0 else 1.0
    status2 = "PASS" if overlap_frac < 0.5 else "FAIL"
    log(f"  [2] IQR separation — infestation [{hd_q25:.4f}, {hd_q75:.4f}] vs "
        f"non-infestation [{ot_q25:.4f}, {ot_q75:.4f}], "
        f"overlap fraction: {overlap_frac:.2f}  → {status2}")

    # 3. Recession ordering: infestation > grassland > riparian (or infestation > both)
    parko_med = hd["rec_mean"].median()
    grass_med = grass["rec_mean"].median()
    rip_med   = rip["rec_mean"].median()
    parko_gt_grass = parko_med > grass_med
    parko_gt_rip   = parko_med > rip_med
    status3 = "PASS" if (parko_gt_grass and parko_gt_rip) else "PARTIAL"
    log(f"  [3] Recession ordering (infestation > others):\n"
        f"        infestation={parko_med:.4f}  grassland={grass_med:.4f}  riparian={rip_med:.4f}\n"
        f"        infestation > grassland: {parko_gt_grass} | infestation > riparian: {parko_gt_rip}  → {status3}")

    # 4. 2D separation: infestation centroid in low-CV / high-rec quadrant
    hd_cv_cent  = hd["nir_cv"].mean()
    ot_cv_cent  = non_hd["nir_cv"].mean()
    hd_rec_cent = hd["rec_mean"].mean()
    ot_rec_cent = non_hd["rec_mean"].mean()
    low_cv  = hd_cv_cent  < ot_cv_cent
    high_rec = hd_rec_cent > ot_rec_cent
    status4 = "PASS" if (low_cv and high_rec) else "PARTIAL"
    log(f"  [4] Infestation centroid in low-CV/high-rec quadrant vs other pixels:\n"
        f"        CV:  infestation={hd_cv_cent:.4f}  others={ot_cv_cent:.4f} → lower: {low_cv}\n"
        f"        rec: infestation={hd_rec_cent:.4f}  others={ot_rec_cent:.4f} → higher: {high_rec}  → {status4}")

    log("-------------------------------------------------------")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== Longreach wet/dry seasonal amplitude (NDVI recession) ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(PARQUET_PATH)

    log("\nLoading dry-NIR stats...")
    dry_nir = pd.read_parquet(DRY_NIR_STATS)
    log(f"  {len(dry_nir)} pixel records loaded")

    log("\nComputing seasonal NDVI medians...")
    peak_med   = seasonal_ndvi_medians(df, PEAK_MONTHS,   "peak")
    trough_med = seasonal_ndvi_medians(df, TROUGH_MONTHS, "trough")

    paired = compute_recession(peak_med, trough_med)
    stats  = per_pixel_stats(paired, df, dry_nir)
    save_stats(stats)

    log("\nFetching Queensland Globe WMS background tile...")
    try:
        bg_img = fetch_wms_image(SURVEY_BBOX, width_px=2048)
        log(f"  Background tile: {bg_img.shape[1]}×{bg_img.shape[0]} px")
    except Exception as exc:
        log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without background")
        bg_img = None

    log("\nGenerating plots...")
    plot_recession_map(stats, bg_img)
    plot_recession_histogram(stats)
    plot_2d_scatter(stats)
    plot_monthly_profiles(df, stats)

    log_success_criteria(stats)
    log("\nDone.")


if __name__ == "__main__":
    main()
