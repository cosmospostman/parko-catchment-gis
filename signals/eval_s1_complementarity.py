"""signals/eval_s1_complementarity.py — S1/S2 complementarity matrix, multi-site.

Answers: which S1 signals add new information beyond which S2 signals, and
does this vary by site?

For every (S2 signal, S1 column) pair per site reports:
  - Spearman rho between their per-pixel-year summary scores
  - Overall AUROC of each signal independently
  - Residual AUROC: the S1 column's AUROC restricted to pixel-years the S2
    signal classifies incorrectly (Youden-optimal threshold)

S1 columns include the four signal dry_means plus two harness-side temporal
descriptors computed here (not in Signal subclasses):
  - vh_dry_std  — within-dry-season VH std (temporal stability)
  - vh_dry_cv   — coefficient of variation = std / |mean| (normalised noise)

These capture the temporal structure a TAM can exploit but a scalar mean
cannot represent. Computing them in the harness keeps signals/s1.py clean.

Performance:
  - Row-group streaming — peak RAM is one row-group, not the full parquet
  - Vectorised aggregation — bincount/lexsort, no Python loop over pixel-years
  - Bounded parallel region loading (MAX_WORKERS) to avoid OOM
  - Youden thresholds cached per S2 signal, not recomputed per S1 column

Usage:
    python signals/eval_s1_complementarity.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signals.eval import SiteSpec
from signals.mavi import MAVISignal
from signals.ndre import NDRESignal, CIRESignal
from signals.ndvi import NDVISignal, NDWISignal, EVISignal
from signals.ndsvi import NDSVISignal
from signals.s1 import VHSignal, VVSignal, VHVVSignal, RVISignal
from tam.core.constants import DRY_DOY_MIN, DRY_DOY_MAX
from utils.training_collector import _region_parquet_path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

S2_SIGNALS = [
    MAVISignal(),
    NDRESignal(),
    CIRESignal(),
    NDVISignal(),
    NDWISignal(),
    EVISignal(),
    NDSVISignal(),
]

S1_SIGNALS = [
    VHSignal(),
    VVSignal(),
    VHVVSignal(),
    RVISignal(),
]

ALL_SIGNALS = S2_SIGNALS + S1_SIGNALS

# Harness-side S1 temporal descriptors — computed here, not in Signal subclasses.
# Each entry: (column_name, label_for_display)
S1_TEMPORAL_COLS = [
    ("vh_dry_std", "vh_dry_std"),
    ("vh_dry_cv",  "vh_dry_cv"),
]

# All S1 columns that appear in the matrix (signal dry_means + temporal descriptors)
S1_COL_NAMES = [f"s1_{s.name}" for s in S1_SIGNALS] + [c for c, _ in S1_TEMPORAL_COLS]

S2_RANK_KEY = "p05"
S1_RANK_KEY = "dry_mean"

SITES = [
    SiteSpec("etna", [
        ("etna_presence_2",  "presence"),
        ("etna_presence_5",  "presence"),
        ("etna_presence_6",  "presence"),
        ("etna_absence_6",   "absence"),
        ("etna_absence_7",   "absence"),
        ("etna_absence_8",   "absence"),
    ]),
    SiteSpec("landsend", [
        ("landsend_sparse_presence_1", "presence"),
        ("landsend_sparse_presence_2", "presence"),
        ("landsend_absence_grass_1",   "absence"),
        ("landsend_absence_riverbed_1","absence"),
    ]),
    SiteSpec("frenchs", [
        ("frenchs_presence_1",               "presence"),
        ("frenchs_presence_2",               "presence"),
        ("frenchs_absence_riparian",         "absence"),
        ("frenchs_absence_riparian_woodland","absence"),
    ]),
    SiteSpec("burdekin", [
        ("burdekin_presence_1", "presence"),
        ("burdekin_absence_4",  "absence"),
        ("burdekin_absence_5",  "absence"),
        ("burdekin_absence_8",  "absence"),
    ]),
]

PRESENCE_MIN_VH_DRY_DB = -21.0
MIN_S2_OBS     = 6
MIN_S1_DRY_OBS = 1
MAX_WORKERS    = 3


# ---------------------------------------------------------------------------
# Vectorised group aggregation
# ---------------------------------------------------------------------------

def _group_p05(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """5th percentile per group via lexsort + searchsorted. NaN for empty groups."""
    valid      = ~np.isnan(values)
    vals_v     = values[valid]
    gidx_v     = group_idx[valid]
    out        = np.full(n_groups, np.nan, dtype="float64")
    if len(vals_v) == 0:
        return out
    order      = np.lexsort((vals_v, gidx_v))
    sv         = vals_v[order]
    sg         = gidx_v[order]
    boundaries = np.searchsorted(sg, np.arange(n_groups))
    ends       = np.searchsorted(sg, np.arange(n_groups), side="right")
    for g in range(n_groups):
        lo, hi = int(boundaries[g]), int(ends[g])
        n = hi - lo
        if n == 0:
            continue
        idx  = 0.05 * (n - 1)
        lo_i = int(idx)
        frac = idx - lo_i
        out[g] = sv[lo + lo_i] * (1 - frac) + sv[lo + lo_i + 1] * frac if lo_i + 1 < n else sv[lo + lo_i]
    return out


def _group_count_valid(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    valid = ~np.isnan(values)
    return np.bincount(group_idx[valid], minlength=n_groups).astype("int32")


def _group_mean(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    valid  = ~np.isnan(values)
    sums   = np.bincount(group_idx[valid], weights=values[valid].astype("float64"), minlength=n_groups)
    counts = np.bincount(group_idx[valid], minlength=n_groups).astype("float64")
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def _group_std(values: np.ndarray, group_idx: np.ndarray, n_groups: int,
               means: np.ndarray) -> np.ndarray:
    """Population std per group given pre-computed means. NaN where n < 2."""
    valid  = ~np.isnan(values)
    vals_v = values[valid].astype("float64")
    gidx_v = group_idx[valid]
    counts = np.bincount(gidx_v, minlength=n_groups).astype("float64")
    sq_dev = (vals_v - means[gidx_v]) ** 2
    sum_sq = np.bincount(gidx_v, weights=sq_dev, minlength=n_groups)
    with np.errstate(invalid="ignore"):
        return np.where(counts >= 2, np.sqrt(sum_sq / counts), np.nan)


# ---------------------------------------------------------------------------
# Per-region loader
# ---------------------------------------------------------------------------

def _load_region(region_id: str, label: str) -> pd.DataFrame | None:
    path = _region_parquet_path(region_id)
    if not path.exists():
        return None

    pf   = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups

    all_pid: list[np.ndarray] = []
    all_yr:  list[np.ndarray] = []
    all_doy: list[np.ndarray] = []
    all_ts:  dict[str, list[np.ndarray]] = {s.name: [] for s in ALL_SIGNALS}
    # Raw linear VH needed for harness-side temporal descriptors
    all_vh_lin: list[np.ndarray] = []

    for rg_idx in range(n_rg):
        print(f"  [{region_id}] row group {rg_idx + 1}/{n_rg} …", flush=True)
        chunk = pf.read_row_group(rg_idx).to_pandas()
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["year"] = chunk["date"].dt.year.astype("int16")
        chunk["doy"]  = chunk["date"].dt.day_of_year.astype("int16")

        all_pid.append(chunk["point_id"].values)
        all_yr.append(chunk["year"].values)
        all_doy.append(chunk["doy"].values)
        for sig in ALL_SIGNALS:
            all_ts[sig.name].append(sig.compute(chunk).values)

        # Raw VH (linear) for S1 rows — NaN elsewhere
        if "vh" in chunk.columns and "source" in chunk.columns:
            vh = np.where(chunk["source"].values == "S1",
                          chunk["vh"].values.astype("float32"), np.nan)
        else:
            vh = np.full(len(chunk), np.nan, dtype="float32")
        all_vh_lin.append(vh)

    pids    = np.concatenate(all_pid)
    yrs     = np.concatenate(all_yr)
    doys    = np.concatenate(all_doy)
    ts      = {name: np.concatenate(arrs).astype("float32")
               for name, arrs in all_ts.items()}
    vh_lin  = np.concatenate(all_vh_lin).astype("float32")

    # Integer group ids for (point_id, year)
    keys = np.array([f"{p}\x00{y}" for p, y in zip(pids, yrs)])
    unique_keys, group_idx = np.unique(keys, return_inverse=True)
    n_groups = len(unique_keys)

    print(f"  [{region_id}] vectorised aggregation over {n_groups} pixel-years …", flush=True)

    rows: dict[str, np.ndarray] = {}

    # --- S2: p05 + n_obs ---
    s2_n_obs = None
    for sig in S2_SIGNALS:
        rows[f"s2_{sig.name}"] = _group_p05(ts[sig.name], group_idx, n_groups)
        n = _group_count_valid(ts[sig.name], group_idx, n_groups)
        if s2_n_obs is None:
            s2_n_obs = n

    # --- S1 signals: dry_mean + dry_n_obs ---
    dry_mask      = (doys >= DRY_DOY_MIN) & (doys <= DRY_DOY_MAX)
    dry_group_idx = group_idx[dry_mask]
    s1_dry_n: dict[str, np.ndarray] = {}
    for sig in S1_SIGNALS:
        dry_ts = ts[sig.name][dry_mask]
        rows[f"s1_{sig.name}"] = _group_mean(dry_ts, dry_group_idx, n_groups)
        s1_dry_n[sig.name]     = _group_count_valid(dry_ts, dry_group_idx, n_groups)

    # --- Harness-side VH temporal descriptors (dry season, linear→dB) ---
    # Convert linear VH to dB, then mask to S1 dry-season rows
    with np.errstate(divide="ignore", invalid="ignore"):
        vh_db = np.where(vh_lin > 0, 10.0 * np.log10(vh_lin), np.nan).astype("float64")
    dry_vh_db     = vh_db[dry_mask]
    dry_vh_n      = _group_count_valid(dry_vh_db, dry_group_idx, n_groups)
    dry_vh_mean   = _group_mean(dry_vh_db, dry_group_idx, n_groups)
    dry_vh_std    = _group_std(dry_vh_db, dry_group_idx, n_groups, dry_vh_mean)
    with np.errstate(invalid="ignore"):
        dry_vh_cv = np.where(np.abs(dry_vh_mean) > 0,
                             dry_vh_std / np.abs(dry_vh_mean), np.nan)
    rows["vh_dry_std"] = dry_vh_std
    rows["vh_dry_cv"]  = dry_vh_cv

    df = pd.DataFrame(rows)
    df["point_id"] = [k.split("\x00")[0] for k in unique_keys]
    df["year"]     = [int(k.split("\x00")[1]) for k in unique_keys]
    df["label"]    = label

    s2_ok = s2_n_obs >= MIN_S2_OBS
    s1_ok = np.zeros(n_groups, dtype=bool)
    for sig in S1_SIGNALS:
        s1_ok |= (s1_dry_n[sig.name] >= MIN_S1_DRY_OBS)
    s1_ok |= (dry_vh_n >= MIN_S1_DRY_OBS)

    df = df[s2_ok & s1_ok].reset_index(drop=True)
    return df if len(df) > 0 else None


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    n_p, n_a = len(pos), len(neg)
    if n_p == 0 or n_a == 0:
        return np.nan
    combined = np.concatenate([pos, neg])
    order    = np.argsort(combined, kind="stable")
    ranks    = np.empty(len(combined), dtype="float64")
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and combined[order[j]] == combined[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    u = ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0
    return float(u / (n_p * n_a))


def _youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    pos_total = int(labels.sum())
    neg_total = len(labels) - pos_total
    if pos_total == 0 or neg_total == 0:
        return float(np.nanmin(scores))
    order         = np.argsort(scores, kind="stable")
    sorted_labels = labels[order]
    cum_pos = np.cumsum(sorted_labels[::-1])[::-1]
    cum_neg = np.cumsum(~sorted_labels[::-1])[::-1]
    tpr = cum_pos / pos_total
    tnr = 1.0 - cum_neg / neg_total
    j   = tpr + tnr - 1.0
    return float(scores[order[int(np.argmax(j))]])


def _residual_auroc(
    s2_scores: np.ndarray,
    s1_scores: np.ndarray,
    labels: np.ndarray,
    s2_threshold: float,
) -> tuple[float, int]:
    s2_pred     = s2_scores >= s2_threshold
    failures    = (labels & ~s2_pred) | (~labels & s2_pred)
    n_fail      = int(failures.sum())
    fail_labels = labels[failures]
    if n_fail < 8 or fail_labels.sum() == 0 or (~fail_labels).sum() == 0:
        return np.nan, n_fail
    fail_s1 = s1_scores[failures]
    return _auroc(fail_s1[fail_labels], fail_s1[~fail_labels]), n_fail


# ---------------------------------------------------------------------------
# Per-site analysis
# ---------------------------------------------------------------------------

def _run_site(site: SiteSpec) -> None:
    print(f"\n{'#' * 72}")
    print(f"# SITE: {site.name}  ({len(site.regions)} regions)")
    print(f"{'#' * 72}")

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_load_region, rid, lbl): rid
            for rid, lbl in site.regions
        }
        for future in as_completed(futures):
            rid    = futures[future]
            result = future.result()
            if result is None:
                print(f"  WARNING: {rid} not found", flush=True)
            else:
                print(f"  loaded {rid}: {len(result)} pixel-years", flush=True)
                frames.append(result)

    if not frames:
        print("  SKIP: no data")
        return

    combined = pd.concat(frames, ignore_index=True)

    is_pres  = combined["label"] == "presence"
    fails_vh = is_pres & (combined["s1_vh_db"] < PRESENCE_MIN_VH_DRY_DB)
    combined = combined[~fails_vh].copy()
    print(f"\n  VH filter: dropped {fails_vh.sum()} presence pixel-years")

    drop_cols = [f"s2_{s.name}" for s in S2_SIGNALS] + S1_COL_NAMES
    combined  = combined.dropna(subset=drop_cols)

    labels = (combined["label"] == "presence").values
    print(f"  Remaining: {labels.sum()} presence, {(~labels).sum()} absence pixel-years\n")

    if labels.sum() < 4 or (~labels).sum() < 4:
        print("  SKIP: insufficient class samples")
        return

    s2_scores_map  = {s.name: combined[f"s2_{s.name}"].values for s in S2_SIGNALS}
    s1_scores_map  = {col: combined[col].values for col in S1_COL_NAMES}
    s1_display_map = {f"s1_{s.name}": s.name for s in S1_SIGNALS}
    s1_display_map.update({c: lbl for c, lbl in S1_TEMPORAL_COLS})

    s2_thresholds = {
        name: _youden_threshold(scores, labels)
        for name, scores in s2_scores_map.items()
    }

    # Overall AUROCs
    print(f"  {'OVERALL AUROC':=<60}")
    for sig in S2_SIGNALS:
        sc = s2_scores_map[sig.name]
        a  = _auroc(sc[labels], sc[~labels])
        print(f"    S2 {sig.name:<14} ({S2_RANK_KEY}):  AUROC={a:.4f}")
    print()
    for col in S1_COL_NAMES:
        sc  = s1_scores_map[col]
        a   = _auroc(sc[labels], sc[~labels])
        lbl = s1_display_map[col]
        print(f"    S1 {lbl:<14}           AUROC={a:.4f}")

    # Complementarity matrix
    col_w = 16
    print()
    print(f"  {'COMPLEMENTARITY MATRIX':=<60}")
    all_s1_labels = [s1_display_map[c] for c in S1_COL_NAMES]
    print(f"  {'':22}" + "".join(f"{lbl:>{col_w}}" for lbl in all_s1_labels))

    for s2 in S2_SIGNALS:
        print(f"  computing {s2.name} …", flush=True)
        s2_sc  = s2_scores_map[s2.name]
        s2_thr = s2_thresholds[s2.name]

        rho_row = f"  {s2.name:<16} rho"
        res_row = f"  {'':16} res"

        for col in S1_COL_NAMES:
            s1_sc  = s1_scores_map[col]
            rho, _ = spearmanr(s2_sc, s1_sc)
            rho_row += f"  {rho:>+{col_w - 2}.3f}"

            rauc, n_fail = _residual_auroc(s2_sc, s1_sc, labels, s2_thr)
            cell = f"{rauc:.3f}({n_fail})" if not np.isnan(rauc) else f"n/a({n_fail})"
            res_row += f"  {cell:>{col_w - 2}}"

        print(rho_row)
        print(res_row)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for site in SITES:
        _run_site(site)


if __name__ == "__main__":
    main()
