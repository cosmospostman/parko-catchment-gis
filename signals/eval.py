"""signals/eval.py — Signal evaluation harness.

Evaluates the discriminative power of a Signal across named site comparisons.
Each site comparison is a set of (region_id, label) pairs drawn from the
training parquets. The harness:

  1. Loads each region's per-observation parquet
  2. Runs signal.compute() to derive the time series
  3. Groups by (point_id, year) and runs signal.summarise() to get scalars
  4. Optionally filters presence pixel-years using the same VH dry-season
     threshold applied during TAM training (presence_min_vh_dry_db), so that
     eval results reflect the pixel-years the model actually trains on
  5. Compares presence vs absence distributions on a chosen summary key
  6. Returns per-site EvalResult with IQR overlap, AUROC, and class stats

Usage
-----
    from signals.ndre import NDRESignal
    from signals.eval import SiteSpec, evaluate

    sites = [
        SiteSpec("arid_clean", [
            ("barcoorah_presence",     "presence"),
            ("lake_mueller_presence",  "presence"),
            ("barcoorah_absence_2",    "absence"),
            ("lake_mueller_absence",   "absence"),
        ]),
        SiteSpec("sparse_stress", [
            ("landsend_sparse_presence_1", "presence"),
            ("landsend_absence_grass_1",   "absence"),
        ]),
    ]

    results = evaluate(NDRESignal(), sites, rank_key="p05")
    for r in results:
        print(r)

    # With presence filter matching TAM training defaults:
    results = evaluate(NDRESignal(), sites, rank_key="p05",
                       presence_min_vh_dry_db=-21.0)
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signals.base import Signal
from tam.core.constants import DRY_DOY_MIN as _DRY_DOY_MIN, DRY_DOY_MAX as _DRY_DOY_MAX
from utils.training_collector import _region_parquet_path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteSpec:
    """A named comparison group of (region_id, label) pairs.

    label must be "presence" or "absence". It is taken from this spec rather
    than the training YAML so that a region can be relabelled for evaluation
    purposes (e.g. testing a false-positive site as a pseudo-presence).
    """
    name: str
    regions: list[tuple[str, str]]   # [(region_id, "presence"|"absence"), ...]


@dataclass
class ClassStats:
    """Distribution statistics for one class within a site."""
    label: str           # "presence" | "absence"
    n_pixels: int        # unique point_ids
    n_pixel_years: int   # (point_id, year) pairs with sufficient obs
    mean: float
    median: float
    p25: float
    p75: float
    iqr: float           # p75 - p25


@dataclass
class EvalResult:
    """Evaluation result for one signal × one site."""
    site: str
    signal: str
    rank_key: str        # which summarise() key was used for ranking

    presence: ClassStats
    absence: ClassStats

    iqr_overlap: float   # fraction of overlap between presence/absence IQRs; 0 = clean separation
    auroc: float         # area under ROC using rank_key score; 0.5 = random

    def __str__(self) -> str:
        sep = "CLEAN" if self.iqr_overlap == 0.0 else f"{self.iqr_overlap:.3f}"
        return (
            f"[{self.site}] {self.signal} ({self.rank_key})"
            f"  IQR_overlap={sep}"
            f"  AUROC={self.auroc:.3f}"
            f"  presence_med={self.presence.median:.4f}"
            f"  absence_med={self.absence.median:.4f}"
            f"  n_pres={self.presence.n_pixel_years}"
            f"  n_abs={self.absence.n_pixel_years}"
        )


# ---------------------------------------------------------------------------
# Presence VH filter
# ---------------------------------------------------------------------------

def _vh_dry_mean(df: pl.DataFrame) -> dict[tuple, float]:
    """Compute mean dry-season VH (dB) per (point_id, year) from S1 rows.

    Returns a dict keyed by (point_id, year). Pixel-years with no S1 dry
    observations are absent — they are not dropped by the filter (no entry = no data,
    not low-VH).
    """
    if "vh" not in df.columns or "source" not in df.columns:
        return {}

    s1 = df.filter(pl.col("source") == "S1")
    if s1.is_empty():
        return {}

    s1 = s1.with_columns([
        pl.col("date").cast(pl.Date).dt.year().alias("year"),
        pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy"),
    ])
    dry = s1.filter(
        (pl.col("doy") >= _DRY_DOY_MIN) & (pl.col("doy") <= _DRY_DOY_MAX)
    )
    if dry.is_empty():
        return {}

    vh_lin = dry["vh"].to_numpy().astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        vh_db = 10.0 * np.log10(np.where(vh_lin > 0, vh_lin, np.nan))
    dry = dry.with_columns(pl.Series("_vh_db", vh_db))

    agg = (
        dry.group_by(["point_id", "year"])
        .agg(pl.col("_vh_db").mean().alias("mean_vh_db"))
    )
    return {(row[0], row[1]): row[2] for row in agg.iter_rows()}


def _apply_presence_vh_filter(
    frame: pl.DataFrame,
    min_vh_db: float,
) -> pl.DataFrame:
    """Drop presence pixel-years whose mean dry-season VH is below min_vh_db.

    Pixel-years with no S1 dry observations (null mean VH) are retained — absence
    of S1 data is not evidence of low backscatter.
    """
    if "_mean_vh_dry_db" not in frame.columns:
        return frame
    is_presence = pl.col("label") == "presence"
    fails = is_presence & (pl.col("_mean_vh_dry_db") < min_vh_db)
    return frame.filter(~fails)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_region(
    region_id: str,
    label: str,
    signal: Signal,
    presence_min_vh_dry_db: float | None = None,
) -> pl.DataFrame | None:
    """Load one region parquet, compute the signal, return a pixel-year summary frame.

    Returns None if the parquet does not exist (region not yet fetched).

    Columns in the returned frame:
        point_id, year, label, <signal.name>_<key> for each summarise() key

    If presence_min_vh_dry_db is set and label is "presence", an additional
    column _mean_vh_dry_db is included so the caller can apply the VH filter.
    """
    path = _region_parquet_path(region_id)
    if not path.exists():
        return None

    pf = pq.ParquetFile(path)
    chunks = [pf.read_row_group(rg) for rg in range(pf.metadata.num_row_groups)]
    df = pl.from_arrow(chunks[0] if len(chunks) == 1 else __import__("pyarrow").concat_tables(chunks))

    # Derive signal time series
    ts = signal.compute(df)

    # Attach year for grouping
    df = df.with_columns([
        pl.Series("_ts", ts.to_numpy()),
        pl.col("date").cast(pl.Date).dt.year().alias("year"),
    ])

    # Pre-compute VH dry-season mean per pixel-year for presence filter
    vh_dry = (
        _vh_dry_mean(df)
        if (label == "presence" and presence_min_vh_dry_db is not None)
        else {}
    )

    # Per pixel-year summarise
    records = []
    for (pid, yr), grp in df.group_by(["point_id", "year"], maintain_order=False):
        ts_grp = grp["_ts"]
        df_grp = grp
        stats = signal.summarise(ts_grp, df_grp)
        if stats["n_obs"] == 0:
            continue
        row = {"point_id": pid, "year": yr, "label": label}
        row.update({f"{signal.name}_{k}": v for k, v in stats.items()})
        if vh_dry:
            row["_mean_vh_dry_db"] = vh_dry.get((pid, yr), np.nan)
        records.append(row)

    if not records:
        return None
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# Discriminability metrics
# ---------------------------------------------------------------------------

def _iqr_overlap(pres_vals: np.ndarray, abs_vals: np.ndarray) -> float:
    """Fraction of IQR overlap between presence and absence distributions.

    Returns the length of the intersection of [p25, p75] intervals divided
    by the length of their union. 0.0 = clean separation; 1.0 = identical.
    Returns NaN if either class has fewer than 4 observations.
    """
    if len(pres_vals) < 4 or len(abs_vals) < 4:
        return np.nan
    p_lo, p_hi = np.percentile(pres_vals, [25, 75])
    a_lo, a_hi = np.percentile(abs_vals,  [25, 75])
    intersection = max(0.0, min(p_hi, a_hi) - max(p_lo, a_lo))
    union = max(p_hi, a_hi) - min(p_lo, a_lo)
    return float(intersection / union) if union > 0 else np.nan


def _auroc(pres_vals: np.ndarray, abs_vals: np.ndarray) -> float:
    """AUROC via sort-based Mann-Whitney U; O(n log n), no sklearn dependency.

    Interprets higher signal value as predicting presence. Returns NaN if
    either class is empty.
    """
    n_p, n_a = len(pres_vals), len(abs_vals)
    if n_p == 0 or n_a == 0:
        return np.nan
    combined = np.concatenate([pres_vals, abs_vals])
    order = np.argsort(combined, kind="stable")
    ranks = np.empty(len(combined), dtype="float64")
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and combined[order[j]] == combined[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    u = ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0
    return float(u / (n_p * n_a))


def _class_stats(label: str, vals: np.ndarray) -> ClassStats:
    if len(vals) == 0:
        return ClassStats(label=label, n_pixels=0, n_pixel_years=0,
                          mean=np.nan, median=np.nan, p25=np.nan,
                          p75=np.nan, iqr=np.nan)
    p25, p75 = np.percentile(vals, [25, 75])
    return ClassStats(
        label=label,
        n_pixels=0,           # filled in by evaluate()
        n_pixel_years=len(vals),
        mean=float(np.mean(vals)),
        median=float(np.median(vals)),
        p25=float(p25),
        p75=float(p75),
        iqr=float(p75 - p25),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate(
    signal: Signal,
    sites: Sequence[SiteSpec],
    rank_key: str = "p05",
    min_obs_per_year: int = 6,
    presence_min_vh_dry_db: float | None = None,
    verbose: bool = True,
) -> list[EvalResult]:
    """Evaluate a signal's discriminative power across one or more site comparisons.

    Parameters
    ----------
    signal:
        The Signal to evaluate.
    sites:
        Named comparison groups; each contains (region_id, label) pairs.
    rank_key:
        Which per-pixel-year summary key to use for IQR overlap and AUROC.
        Must be a key returned by signal.summarise() (e.g. "p05", "std",
        "amplitude"). Default "p05" suits low-percentile dry-season signals.
    min_obs_per_year:
        Minimum n_obs in a pixel-year to include it in the comparison.
        Pixel-years below this threshold are excluded (sparse cloud cover).
    presence_min_vh_dry_db:
        If set, drop presence pixel-years whose mean dry-season VH backscatter
        (dB) is below this threshold, mirroring the TAM training presence filter
        (default −21.0 dB). Pass None (default) to skip filtering and evaluate
        all labeled pixel-years. Pixel-years with no S1 dry observations are
        always retained regardless of this threshold.
    verbose:
        Print progress messages.

    Returns
    -------
    List of EvalResult, one per site.
    """
    score_col = f"{signal.name}_{rank_key}"
    results: list[EvalResult] = []

    for site in sites:
        if verbose:
            print(f"[{site.name}] loading {len(site.regions)} regions …")

        frames: list[pl.DataFrame] = []
        missing: list[str] = []
        with ThreadPoolExecutor(max_workers=len(site.regions)) as pool:
            futures = {
                pool.submit(
                    _load_region, region_id, label, signal,
                    presence_min_vh_dry_db=presence_min_vh_dry_db,
                ): region_id
                for region_id, label in site.regions
            }
            for future in as_completed(futures):
                region_id = futures[future]
                frame = future.result()
                if frame is None:
                    missing.append(region_id)
                else:
                    frames.append(frame)

        if missing and verbose:
            print(f"  WARNING: {len(missing)} region(s) not found: {missing}")

        if not frames:
            if verbose:
                print(f"  SKIP: no data for site {site.name!r}")
            continue

        combined = pl.concat(frames, how="diagonal")

        # Apply presence VH filter — mirrors TAM training presence filter
        if presence_min_vh_dry_db is not None:
            n_before = (combined["label"] == "presence").sum()
            combined = _apply_presence_vh_filter(combined, presence_min_vh_dry_db)
            n_dropped = n_before - (combined["label"] == "presence").sum()
            if verbose and n_dropped > 0:
                print(f"  VH filter (<{presence_min_vh_dry_db} dB): dropped {n_dropped} presence pixel-years")
        if "_mean_vh_dry_db" in combined.columns:
            combined = combined.drop("_mean_vh_dry_db")

        # Apply minimum obs filter
        n_obs_col = f"{signal.name}_n_obs"
        if n_obs_col in combined.columns:
            combined = combined.filter(pl.col(n_obs_col) >= min_obs_per_year)

        if score_col not in combined.columns:
            raise ValueError(
                f"rank_key {rank_key!r} not in summarise() output. "
                f"Available columns: {[c for c in combined.columns if c.startswith(signal.name)]}"
            )

        pres = combined.filter(pl.col("label") == "presence")
        abs_ = combined.filter(pl.col("label") == "absence")

        pres_vals = pres[score_col].drop_nulls().to_numpy()
        pres_vals = pres_vals[~np.isnan(pres_vals)]
        abs_vals  = abs_[score_col].drop_nulls().to_numpy()
        abs_vals  = abs_vals[~np.isnan(abs_vals)]

        pres_stats = _class_stats("presence", pres_vals)
        abs_stats  = _class_stats("absence",  abs_vals)
        pres_stats.n_pixels = pres["point_id"].n_unique()
        abs_stats.n_pixels  = abs_["point_id"].n_unique()

        result = EvalResult(
            site=site.name,
            signal=signal.name,
            rank_key=rank_key,
            presence=pres_stats,
            absence=abs_stats,
            iqr_overlap=_iqr_overlap(pres_vals, abs_vals),
            auroc=_auroc(pres_vals, abs_vals),
        )
        results.append(result)

        if verbose:
            print(f"  {result}")

    return results
