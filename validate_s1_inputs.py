"""validate_s1_inputs.py — Validate scientific assumptions about S1 input data.

Runs against the local S1 cache (LOCAL_S1_ROOT) and reports how the actual
data compares against the assumptions baked into stage 4 (flood_mask_from_scene).

Usage
-----
    source config.sh
    YEAR=2024 python validate_s1_inputs.py [--n-scenes 20] [--plots]

Optional flags
--------------
--n-scenes N    Max scenes to sample (default: 20, stratified across the season)
--plots         Save per-scene histogram PNG strips to OUTPUTS_DIR/<year>/s1_validation/
--season        "wet" | "dry" | "both" (default: both)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,  # suppress noisy production logging
    format="%(levelname)-8s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scientific expectations (mirroring production constants + literature)
# ---------------------------------------------------------------------------

# From sar.py / 04_flood_extent.py
VH_WATER_THRESHOLD_DB   = -20.0   # fixed guard in flood_mask_from_scene
VV_WATER_FRACTION_LIMIT = 0.40    # sanity guard — Otsu unreliable above this
DRY_MASK_THRESHOLD_DB   = -16.0   # threshold formerly used for reference mask
FLOOD_MIN_FREQUENCY     = 0.33
MIN_OBS                 = 4

# From the literature review (PDF) and sar.py comments
VV_WATER_PEAK_EXPECTED_DB   = -20.0   # calm water, C-band VV
VV_LAND_PEAK_EXPECTED_DB    =  -8.0   # dry soil / sparse vegetation
VV_FLOOD_THRESHOLD_DB       = -14.0   # active floodwater delineation threshold (lit)
BIMODAL_PEAK_SEP_MIN_DB     =   6.0   # minimum separation to trust Otsu
BIMODAL_VALLEY_RATIO_MAX    =   0.70  # valley/lower-peak height; above → unimodal

DN2_LAND_MODE_EXPECTED_RANGE = (-12.0, -4.0)   # plausible land backscatter after DN²/1e6
DN2_WATER_MODE_EXPECTED_RANGE = (-30.0, -14.0)  # plausible water backscatter


# ---------------------------------------------------------------------------
# Per-scene result
# ---------------------------------------------------------------------------

class SceneResult(NamedTuple):
    scene_id: str
    season: str           # "dry" | "wet"
    n_valid_px: int

    # VV histogram shape
    vv_peak1_db: float    # lower (water-like) peak centre
    vv_peak2_db: float    # upper (land-like) peak centre
    vv_peak_sep_db: float
    vv_valley_ratio: float  # valley height / lower-peak height (lower = more bimodal)
    is_bimodal: bool

    # Otsu result
    otsu_threshold_db: float
    vv_water_fraction: float  # fraction of valid pixels below Otsu
    sanity_guard_triggered: bool

    # VH guard analysis (pixels that Otsu classifies as water)
    vh_guard_available: bool
    vh_otsu_water_median_db: float   # median VH of VV-Otsu-water pixels
    vh_guard_retention_pct: float    # % of Otsu-water pixels that survive VH guard

    # Calibration check
    land_mode_db: float   # mode of upper (land) peak
    water_mode_db: float  # mode of lower (water) peak

    error: str            # non-empty if processing failed


# ---------------------------------------------------------------------------
# Histogram analysis helpers
# ---------------------------------------------------------------------------

N_BINS = 512


def _histogram_db(values_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, centres) for values_db clipped to [-35, 5] dB."""
    lo, hi = -35.0, 5.0
    counts, edges = np.histogram(values_db, bins=N_BINS, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    return counts, centres


def _smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
    """Simple box-car smoothing for peak detection."""
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def _find_peaks_and_valley(counts: np.ndarray, centres: np.ndarray
                           ) -> tuple[int, int, int]:
    """Return (peak1_idx, peak2_idx, valley_idx) for a smoothed histogram.

    peak1 is the lower-dB (water-like) peak, peak2 is the upper (land-like).
    If the histogram is unimodal, peak1 == peak2 == global max and valley is
    between them (meaningless — caller checks valley_ratio).
    """
    smoothed = _smooth(counts.astype(float), w=9)
    global_max = int(np.argmax(smoothed))

    # Look for a secondary peak by scanning left from the global max
    left_half = smoothed[:global_max]
    if left_half.size == 0:
        return global_max, global_max, global_max

    # Find the minimum (valley candidate) in the left half
    valley_idx = int(np.argmin(left_half))
    if valley_idx == 0:
        return global_max, global_max, global_max

    # Secondary peak is the max to the left of the valley
    peak1_idx = int(np.argmax(smoothed[:valley_idx]))
    return peak1_idx, global_max, valley_idx


def analyse_histogram(values_db: np.ndarray
                      ) -> tuple[float, float, float, float, bool]:
    """Return (peak1_db, peak2_db, sep_db, valley_ratio, is_bimodal)."""
    counts, centres = _histogram_db(values_db)
    p1, p2, v = _find_peaks_and_valley(counts, centres)

    peak1_db = float(centres[p1])
    peak2_db = float(centres[p2])
    sep_db   = float(peak2_db - peak1_db)

    if p1 == p2:
        # Unimodal
        return peak1_db, peak2_db, 0.0, 1.0, False

    smoothed = _smooth(counts.astype(float), w=9)
    valley_h = float(smoothed[v])
    lower_peak_h = float(smoothed[p1])
    valley_ratio = valley_h / max(lower_peak_h, 1e-6)

    is_bimodal = (sep_db >= BIMODAL_PEAK_SEP_MIN_DB) and (valley_ratio < BIMODAL_VALLEY_RATIO_MAX)
    return peak1_db, peak2_db, sep_db, valley_ratio, is_bimodal


# ---------------------------------------------------------------------------
# Otsu (copied from sar.py so we don't mutate shared state)
# ---------------------------------------------------------------------------

def _otsu_threshold(values: np.ndarray, n_bins: int = 512) -> float:
    counts, edges = np.histogram(values, bins=n_bins)
    bin_centres = (edges[:-1] + edges[1:]) / 2
    total = counts.sum()
    sum_total = (counts * bin_centres).sum()
    sum_b, weight_b = 0.0, 0.0
    best_var, best_t = 0.0, bin_centres[0]
    for i in range(n_bins):
        weight_b += counts[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += counts[i] * bin_centres[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = bin_centres[i]
    return float(best_t)


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def process_scene(item, bbox_wgs84: list, season: str, resolution: int = 50) -> SceneResult:
    """Run the same preprocessing path as production, then analyse the result."""
    from utils.sar import _preprocess_gcp_warp, _focal_mean_inplace

    try:
        ds = _preprocess_gcp_warp(item, bbox_wgs84, resolution, polarisations=("VV", "VH"))
    except Exception as exc:
        return SceneResult(
            scene_id=item.id, season=season, n_valid_px=0,
            vv_peak1_db=np.nan, vv_peak2_db=np.nan, vv_peak_sep_db=np.nan,
            vv_valley_ratio=np.nan, is_bimodal=False,
            otsu_threshold_db=np.nan, vv_water_fraction=np.nan,
            sanity_guard_triggered=False,
            vh_guard_available=False, vh_otsu_water_median_db=np.nan,
            vh_guard_retention_pct=np.nan,
            land_mode_db=np.nan, water_mode_db=np.nan,
            error=str(exc),
        )

    if "VV" not in ds:
        return SceneResult(
            scene_id=item.id, season=season, n_valid_px=0,
            vv_peak1_db=np.nan, vv_peak2_db=np.nan, vv_peak_sep_db=np.nan,
            vv_valley_ratio=np.nan, is_bimodal=False,
            otsu_threshold_db=np.nan, vv_water_fraction=np.nan,
            sanity_guard_triggered=False,
            vh_guard_available=False, vh_otsu_water_median_db=np.nan,
            vh_guard_retention_pct=np.nan,
            land_mode_db=np.nan, water_mode_db=np.nan,
            error="VV band missing",
        )

    vv_lin = ds["VV"].values.copy()
    vh_lin = ds["VH"].values.copy() if "VH" in ds else None

    observed = np.isfinite(vv_lin) & (vv_lin > 0)
    n_valid = int(observed.sum())

    if n_valid < 100:
        return SceneResult(
            scene_id=item.id, season=season, n_valid_px=n_valid,
            vv_peak1_db=np.nan, vv_peak2_db=np.nan, vv_peak_sep_db=np.nan,
            vv_valley_ratio=np.nan, is_bimodal=False,
            otsu_threshold_db=np.nan, vv_water_fraction=np.nan,
            sanity_guard_triggered=False,
            vh_guard_available=False, vh_otsu_water_median_db=np.nan,
            vh_guard_retention_pct=np.nan,
            land_mode_db=np.nan, water_mode_db=np.nan,
            error="too few valid pixels",
        )

    # Mirror production: speckle filter then dB conversion
    vv_nan = ~observed
    vv_lin[vv_nan] = np.nan
    _focal_mean_inplace(vv_lin, vv_nan, radius=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log10(vv_lin + 1e-12, out=vv_lin)
        vv_lin *= 10
    vv_db = vv_lin
    vv_valid = vv_db[observed]

    # Histogram shape analysis
    p1_db, p2_db, sep_db, valley_ratio, is_bimodal = analyse_histogram(vv_valid)

    # Otsu
    otsu_t = _otsu_threshold(vv_valid)
    water_mask = observed & (vv_db < otsu_t)
    vv_water_frac = float(water_mask.sum()) / max(observed.sum(), 1)
    sanity_triggered = vv_water_frac > VV_WATER_FRACTION_LIMIT

    # VH guard analysis
    vh_guard_available = False
    vh_otsu_water_median = np.nan
    vh_retention_pct = np.nan

    if vh_lin is not None:
        vh_obs = np.isfinite(vh_lin) & (vh_lin > 0)
        vh_nan = ~vh_obs
        vh_lin[vh_nan] = np.nan
        _focal_mean_inplace(vh_lin, vh_nan, radius=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log10(vh_lin + 1e-12, out=vh_lin)
            vh_lin *= 10
        vh_db = vh_lin

        # VH values at pixels Otsu classifies as water
        otsu_water_pixels = water_mask & vh_obs
        if otsu_water_pixels.sum() > 0:
            vh_at_water = vh_db[otsu_water_pixels]
            vh_otsu_water_median = float(np.nanmedian(vh_at_water))
            surviving = float((vh_at_water < VH_WATER_THRESHOLD_DB).sum())
            vh_retention_pct = 100.0 * surviving / len(vh_at_water)
            vh_guard_available = True

    # Calibration: land mode is the upper peak, water mode is the lower peak
    land_mode_db  = p2_db
    water_mode_db = p1_db

    return SceneResult(
        scene_id=item.id, season=season, n_valid_px=n_valid,
        vv_peak1_db=p1_db, vv_peak2_db=p2_db, vv_peak_sep_db=sep_db,
        vv_valley_ratio=valley_ratio, is_bimodal=is_bimodal,
        otsu_threshold_db=otsu_t, vv_water_fraction=vv_water_frac,
        sanity_guard_triggered=sanity_triggered,
        vh_guard_available=vh_guard_available,
        vh_otsu_water_median_db=vh_otsu_water_median,
        vh_guard_retention_pct=vh_retention_pct,
        land_mode_db=land_mode_db, water_mode_db=water_mode_db,
        error="",
    )


# ---------------------------------------------------------------------------
# Optional histogram plot
# ---------------------------------------------------------------------------

def save_histogram_plot(item, bbox_wgs84: list, resolution: int, out_dir: Path) -> None:
    """Save a VV+VH histogram PNG for a single scene."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from utils.sar import _preprocess_gcp_warp, _focal_mean_inplace

        ds = _preprocess_gcp_warp(item, bbox_wgs84, resolution, polarisations=("VV", "VH"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(item.id, fontsize=9)

        for ax, pol in zip(axes, ["VV", "VH"]):
            if pol not in ds:
                ax.set_title(f"{pol} — missing")
                continue
            arr = ds[pol].values.copy()
            obs = np.isfinite(arr) & (arr > 0)
            arr_nan = ~obs
            arr[arr_nan] = np.nan
            _focal_mean_inplace(arr, arr_nan, radius=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.log10(arr + 1e-12, out=arr)
                arr *= 10
            vals = arr[obs]
            ax.hist(vals, bins=256, range=(-35, 5), density=True, color="steelblue", alpha=0.7)
            if pol == "VV":
                otsu_t = _otsu_threshold(vals)
                ax.axvline(otsu_t, color="red",    linestyle="--", label=f"Otsu {otsu_t:.1f} dB")
                ax.axvline(VV_FLOOD_THRESHOLD_DB, color="orange", linestyle=":", label=f"Lit. flood {VV_FLOOD_THRESHOLD_DB} dB")
            else:
                ax.axvline(VH_WATER_THRESHOLD_DB, color="red", linestyle="--", label=f"VH guard {VH_WATER_THRESHOLD_DB} dB")
            ax.set_xlabel("dB")
            ax.set_ylabel("density")
            ax.set_title(pol)
            ax.legend(fontsize=8)

        fig.tight_layout()
        out_path = out_dir / f"{item.id}_histogram.png"
        fig.savefig(str(out_path), dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not save plot for %s: %s", item.id, exc)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _ok(cond: bool) -> str:
    return "  OK  " if cond else " WARN "


def _pct(v: float) -> str:
    return f"{v:.1f}%" if np.isfinite(v) else "  n/a"


def _db(v: float) -> str:
    return f"{v:+.1f} dB" if np.isfinite(v) else "   n/a"


def print_report(results: list[SceneResult]) -> None:
    good = [r for r in results if not r.error]
    bad  = [r for r in results if r.error]

    print()
    print("=" * 72)
    print("  S1 INPUT VALIDATION — Mitchell River Catchment")
    print(f"  Scenes processed: {len(good)}  |  failed/skipped: {len(bad)}")
    print("=" * 72)

    if not good:
        print("\n  No scenes could be processed. Check LOCAL_S1_ROOT and scene paths.")
        return

    # ------------------------------------------------------------------
    # Per-scene table
    # ------------------------------------------------------------------
    print()
    print("PER-SCENE SUMMARY")
    print("-" * 72)
    hdr = f"{'Scene':<45} {'Ssn':>3}  {'Otsu':>7}  {'WatFrac':>7}  {'Bimod':>5}  {'VHret':>6}"
    print(hdr)
    print("-" * 72)
    for r in good:
        bimod = "YES" if r.is_bimodal else " no"
        vh_ret = _pct(r.vh_guard_retention_pct) if r.vh_guard_available else "  n/a"
        sg = " [SANITY]" if r.sanity_guard_triggered else ""
        print(f"{r.scene_id:<45} {r.season:>3}  {_db(r.otsu_threshold_db):>10}"
              f"  {_pct(r.vv_water_fraction*100):>7}  {bimod:>5}  {vh_ret:>6}{sg}")
    if bad:
        print()
        for r in bad:
            print(f"  FAILED  {r.scene_id}  ({r.error})")

    # ------------------------------------------------------------------
    # Assumption checks
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("ASSUMPTION CHECKS")
    print("=" * 72)

    # 1. Bimodality
    n_bimodal = sum(1 for r in good if r.is_bimodal)
    pct_bimodal = 100 * n_bimodal / len(good)
    seps = [r.vv_peak_sep_db for r in good if r.is_bimodal and np.isfinite(r.vv_peak_sep_db)]
    vals = [r.vv_valley_ratio for r in good if np.isfinite(r.vv_valley_ratio)]
    print()
    print("ASSUMPTION 1: VV histogram is bimodal (water vs land peaks)")
    print(f"  Expected : two peaks separated by >{BIMODAL_PEAK_SEP_MIN_DB:.0f} dB, "
          f"valley ratio <{BIMODAL_VALLEY_RATIO_MAX:.2f}")
    print(f"  Observed : {n_bimodal}/{len(good)} scenes bimodal ({pct_bimodal:.0f}%)  "
          f"[{_ok(pct_bimodal >= 50)}]")
    if seps:
        print(f"             Peak separation (bimodal scenes): "
              f"min {min(seps):.1f} dB  median {float(np.median(seps)):.1f} dB  max {max(seps):.1f} dB")
    if vals:
        print(f"             Valley ratio (all scenes):        "
              f"min {min(vals):.2f}  median {float(np.median(vals)):.2f}  max {max(vals):.2f}")
    uni_scenes = [r.scene_id for r in good if not r.is_bimodal]
    if uni_scenes:
        print(f"  Unimodal scenes ({len(uni_scenes)}): {', '.join(uni_scenes[:5])}"
              + (" ..." if len(uni_scenes) > 5 else ""))

    # 2. VH water threshold
    vh_results = [r for r in good if r.vh_guard_available]
    print()
    print(f"ASSUMPTION 2: Open water has VH < {VH_WATER_THRESHOLD_DB:.0f} dB (fixed guard)")
    print(f"  Expected : VH median of Otsu-water pixels << {VH_WATER_THRESHOLD_DB:.0f} dB; "
          f"guard retention >70%")
    if vh_results:
        medians = [r.vh_otsu_water_median_db for r in vh_results if np.isfinite(r.vh_otsu_water_median_db)]
        retentions = [r.vh_guard_retention_pct for r in vh_results if np.isfinite(r.vh_guard_retention_pct)]
        if medians:
            med_median = float(np.median(medians))
            print(f"  Observed : VH median @ Otsu-water pixels: "
                  f"min {min(medians):.1f}  median {med_median:.1f}  max {max(medians):.1f} dB  "
                  f"[{_ok(med_median < VH_WATER_THRESHOLD_DB)}]")
        if retentions:
            med_ret = float(np.median(retentions))
            print(f"             VH guard retention rate:        "
                  f"min {min(retentions):.1f}%  median {med_ret:.1f}%  max {max(retentions):.1f}%  "
                  f"[{_ok(med_ret >= 70)}]")
            low_ret = [r.scene_id for r in vh_results
                       if np.isfinite(r.vh_guard_retention_pct) and r.vh_guard_retention_pct < 50]
            if low_ret:
                print(f"  WARNING: VH guard rejecting >50% of Otsu-water in: "
                      f"{', '.join(low_ret[:5])}" + (" ..." if len(low_ret) > 5 else ""))
    else:
        print("  Observed : no scenes with VH data available")

    # 3. Sanity guard (40% water fraction)
    n_sanity = sum(1 for r in good if r.sanity_guard_triggered)
    print()
    print(f"ASSUMPTION 3: VV water fraction sanity guard (<{VV_WATER_FRACTION_LIMIT*100:.0f}%) "
          f"rarely triggers")
    print(f"  Expected : guard triggers on <10% of scenes (dry/unimodal only)")
    pct_sanity = 100 * n_sanity / len(good)
    print(f"  Observed : guard triggered on {n_sanity}/{len(good)} scenes ({pct_sanity:.1f}%)  "
          f"[{_ok(pct_sanity < 15)}]")
    if n_sanity:
        sg_scenes = [r.scene_id for r in good if r.sanity_guard_triggered]
        print(f"             Triggered on: {', '.join(sg_scenes[:5])}"
              + (" ..." if len(sg_scenes) > 5 else ""))
        fracs = [r.vv_water_fraction for r in good if r.sanity_guard_triggered]
        print(f"             Water fractions: {[f'{f*100:.1f}%' for f in fracs[:5]]}")

    # 4. Backscatter calibration range
    land_modes = [r.land_mode_db for r in good if r.is_bimodal and np.isfinite(r.land_mode_db)]
    water_modes = [r.water_mode_db for r in good if r.is_bimodal and np.isfinite(r.water_mode_db)]
    print()
    print("ASSUMPTION 4: DN²/1e6 produces physically plausible backscatter values")
    print(f"  Expected : land mode  {DN2_LAND_MODE_EXPECTED_RANGE[0]:+.0f} to "
          f"{DN2_LAND_MODE_EXPECTED_RANGE[1]:+.0f} dB  |  "
          f"water mode {DN2_WATER_MODE_EXPECTED_RANGE[0]:+.0f} to "
          f"{DN2_WATER_MODE_EXPECTED_RANGE[1]:+.0f} dB")
    if land_modes:
        lm = float(np.median(land_modes))
        in_range = DN2_LAND_MODE_EXPECTED_RANGE[0] <= lm <= DN2_LAND_MODE_EXPECTED_RANGE[1]
        print(f"  Observed : land mode  min {min(land_modes):+.1f}  median {lm:+.1f}  max {max(land_modes):+.1f} dB  "
              f"[{_ok(in_range)}]")
    if water_modes:
        wm = float(np.median(water_modes))
        in_range = DN2_WATER_MODE_EXPECTED_RANGE[0] <= wm <= DN2_WATER_MODE_EXPECTED_RANGE[1]
        print(f"             water mode min {min(water_modes):+.1f}  median {wm:+.1f}  max {max(water_modes):+.1f} dB  "
              f"[{_ok(in_range)}]")

    # 5. Otsu threshold vs literature flood threshold
    otsu_ts = [r.otsu_threshold_db for r in good if not r.sanity_guard_triggered
               and np.isfinite(r.otsu_threshold_db)]
    print()
    print(f"ASSUMPTION 5: Otsu threshold aligns with literature flood threshold (~{VV_FLOOD_THRESHOLD_DB:.0f} dB)")
    print(f"  Expected : Otsu threshold in range −20 to −10 dB for flooded scenes")
    if otsu_ts:
        ot_med = float(np.median(otsu_ts))
        in_range = -20 <= ot_med <= -10
        print(f"  Observed : min {min(otsu_ts):+.1f}  median {ot_med:+.1f}  max {max(otsu_ts):+.1f} dB  "
              f"[{_ok(in_range)}]")
        outliers = [r.scene_id for r in good
                    if np.isfinite(r.otsu_threshold_db) and not r.sanity_guard_triggered
                    and (r.otsu_threshold_db < -25 or r.otsu_threshold_db > -5)]
        if outliers:
            print(f"  Otsu threshold outside −25 to −5 dB: {', '.join(outliers[:5])}")

    # 6. Dry-season reference mask (disabled in production)
    dry_results = [r for r in good if r.season == "dry"]
    if dry_results:
        dry_land_modes = [r.land_mode_db for r in dry_results if np.isfinite(r.land_mode_db)]
        print()
        print(f"ASSUMPTION 6: Dry-season backscatter separable from flood-season water")
        print(f"  Expected : dry-season land mode >> {DRY_MASK_THRESHOLD_DB:.0f} dB "
              f"(reference mask threshold, now disabled)")
        print(f"  Context  : reference mask disabled in production because DN²/1e6 "
              f"produced variable absolute values flagging ~76% of catchment")
        if dry_land_modes:
            dlm = float(np.median(dry_land_modes))
            print(f"  Observed : dry land mode  min {min(dry_land_modes):+.1f}  "
                  f"median {dlm:+.1f}  max {max(dry_land_modes):+.1f} dB")
            gap = dlm - DRY_MASK_THRESHOLD_DB
            print(f"             Separation from mask threshold: {gap:+.1f} dB  "
                  f"[{_ok(gap > 3.0)}]")

    # ------------------------------------------------------------------
    # Deviation summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("DEVIATION SUMMARY  (items to investigate)")
    print("=" * 72)
    issues = []

    if pct_bimodal < 50:
        issues.append(f"  - Only {pct_bimodal:.0f}% of scenes are bimodal — Otsu may be "
                      f"splitting within the land distribution on most scenes")

    if vh_results:
        retentions = [r.vh_guard_retention_pct for r in vh_results if np.isfinite(r.vh_guard_retention_pct)]
        if retentions and float(np.median(retentions)) < 50:
            issues.append(f"  - Median VH guard retention {float(np.median(retentions)):.1f}% — "
                          f"the fixed −20 dB threshold may be rejecting real floodwater")
        if retentions and float(np.median(retentions)) > 95:
            issues.append(f"  - VH guard retention {float(np.median(retentions)):.1f}% — guard has "
                          f"almost no effect; consider whether VH data quality is adequate")

    if pct_sanity > 15:
        issues.append(f"  - Sanity guard triggered on {pct_sanity:.1f}% of scenes — "
                      f"many scenes may be dry / incorrectly calibrated")

    if land_modes:
        lm = float(np.median(land_modes))
        if not (DN2_LAND_MODE_EXPECTED_RANGE[0] <= lm <= DN2_LAND_MODE_EXPECTED_RANGE[1]):
            issues.append(f"  - Land mode median {lm:+.1f} dB is outside expected range "
                          f"{DN2_LAND_MODE_EXPECTED_RANGE} — DN²/1e6 calibration may be off; "
                          f"all fixed dB thresholds will be shifted accordingly")

    if otsu_ts:
        ot_med = float(np.median(otsu_ts))
        if not (-20 <= ot_med <= -10):
            issues.append(f"  - Median Otsu threshold {ot_med:+.1f} dB is outside −20 to −10 dB — "
                          f"thresholds in sar.py may need recalibration against actual data range")

    if not issues:
        print()
        print("  No significant deviations from scientific assumptions found.")
    else:
        print()
        for issue in issues:
            print(issue)

    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Scene discovery and sampling
# ---------------------------------------------------------------------------

def discover_and_sample(
    bbox_wgs84: list,
    year: int,
    n_scenes: int,
    season_filter: str,
) -> list:
    """Return up to n_scenes STAC items from LOCAL_S1_ROOT, stratified by season."""
    import os
    from utils.stac import search_sentinel1, rewrite_hrefs_to_local
    import config

    local_root = os.environ.get("LOCAL_S1_ROOT", "")

    wet_start  = f"{year}-{config.FLOOD_SEASON_START}"
    wet_end    = f"{year}-{config.FLOOD_SEASON_END}"
    dry_start  = f"{year}-10-01"
    dry_end    = f"{year}-11-30"

    endpoint   = config.STAC_ENDPOINT_ELEMENT84
    collection = config.S1_COLLECTION

    season_ranges = []
    if season_filter in ("wet", "both"):
        season_ranges.append(("wet", wet_start, wet_end))
    if season_filter in ("dry", "both"):
        season_ranges.append(("dry", dry_start, dry_end))

    all_items = []
    for season, start, end in season_ranges:
        items = search_sentinel1(bbox=bbox_wgs84, start=start, end=end,
                                 endpoint=endpoint, collection=collection)
        if local_root:
            items = rewrite_hrefs_to_local(items, local_root)
        for item in items:
            all_items.append((season, item))
        print(f"  Found {len(items)} {season}-season scenes")

    if not all_items:
        return []

    # Stratified subsample: interleave seasons
    from itertools import islice
    wet_items = [(s, i) for s, i in all_items if s == "wet"]
    dry_items = [(s, i) for s, i in all_items if s == "dry"]

    # Sample evenly across the time axis (items are chronological)
    def _subsample(lst, n):
        if len(lst) <= n:
            return lst
        step = len(lst) / n
        return [lst[int(i * step)] for i in range(n)]

    n_wet = n_scenes if season_filter == "wet" else n_scenes // 2
    n_dry = n_scenes if season_filter == "dry" else n_scenes - n_wet

    sampled = _subsample(wet_items, n_wet) + _subsample(dry_items, n_dry)
    print(f"  Sampled {len(sampled)} scenes for validation "
          f"({sum(1 for s,_ in sampled if s=='wet')} wet, "
          f"{sum(1 for s,_ in sampled if s=='dry')} dry)")
    return sampled


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-scenes", type=int, default=20,
                        help="Max scenes to sample (default: 20)")
    parser.add_argument("--plots", action="store_true",
                        help="Save per-scene histogram PNGs")
    parser.add_argument("--season", choices=("wet", "dry", "both"), default="both",
                        help="Which season to sample (default: both)")
    parser.add_argument("--resolution", type=int, default=50,
                        help="Processing resolution in metres (default: 50)")
    args = parser.parse_args()

    # Config import requires env vars
    import config
    import geopandas as gpd

    catchment = gpd.read_file(str(config.CATCHMENT_GEOJSON))
    bbox_wgs84 = list(catchment.to_crs("EPSG:4326").total_bounds)

    print(f"\nValidating S1 inputs for year {config.YEAR}")
    print(f"Catchment bbox (WGS84): {[round(v, 3) for v in bbox_wgs84]}")
    print(f"Resolution: {args.resolution} m  |  Max scenes: {args.n_scenes}  |  Season: {args.season}\n")

    sampled = discover_and_sample(
        bbox_wgs84=bbox_wgs84,
        year=config.YEAR,
        n_scenes=args.n_scenes,
        season_filter=args.season,
    )

    if not sampled:
        print("No scenes found. Check STAC endpoint or LOCAL_S1_ROOT.")
        sys.exit(1)

    plot_dir = None
    if args.plots:
        plot_dir = config.OUTPUTS_DIR / str(config.YEAR) / "s1_validation"
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Histogram plots will be saved to: {plot_dir}\n")

    results = []
    for i, (season, item) in enumerate(sampled, 1):
        print(f"  [{i:>2}/{len(sampled)}] {season:>3}  {item.id} ...", end=" ", flush=True)
        r = process_scene(item, bbox_wgs84, season, resolution=args.resolution)
        if r.error:
            print(f"FAILED ({r.error})")
        else:
            bimod = "bimodal" if r.is_bimodal else "unimodal"
            print(f"ok  Otsu={r.otsu_threshold_db:+.1f} dB  WatFrac={r.vv_water_fraction*100:.1f}%  {bimod}")

        results.append(r)

        if args.plots and not r.error and plot_dir:
            save_histogram_plot(item, bbox_wgs84, args.resolution, plot_dir)

    print_report(results)


if __name__ == "__main__":
    main()
