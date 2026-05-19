"""signals/eval.py — Signal evaluation harness.

Evaluates the discriminative power of a Signal across named site comparisons.
Each site comparison is a set of (region_id, label) pairs drawn from the
training parquets. The harness:

  1. Loads each region's per-observation parquet
  2. Runs signal.compute() to derive the time series
  3. Groups by (point_id, year) and runs signal.summarise() to get scalars
  4. Compares presence vs absence distributions on a chosen summary key
  5. Returns per-site EvalResult with IQR overlap, AUROC, and class stats

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
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signals.base import Signal
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
# Loading
# ---------------------------------------------------------------------------

def _load_region(region_id: str, label: str, signal: Signal) -> pd.DataFrame | None:
    """Load one region parquet, compute the signal, return a pixel-year summary frame.

    Returns None if the parquet does not exist (region not yet fetched).

    Columns in the returned frame:
        point_id, year, label, <signal.name>_<key> for each summarise() key
    """
    path = _region_parquet_path(region_id)
    if not path.exists():
        return None

    pf = pq.ParquetFile(path)
    chunks: list[pd.DataFrame] = []
    for rg in range(pf.metadata.num_row_groups):
        chunks.append(pf.read_row_group(rg).to_pandas())
    df = pd.concat(chunks, ignore_index=True)

    # Derive signal time series
    ts = signal.compute(df)

    # Attach year for grouping
    df["_ts"] = ts
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Per pixel-year summarise
    records = []
    for (pid, yr), grp in df.groupby(["point_id", "year"], sort=False):
        stats = signal.summarise(grp["_ts"], grp)
        if stats["n_obs"] == 0:
            continue
        row = {"point_id": pid, "year": yr, "label": label}
        row.update({f"{signal.name}_{k}": v for k, v in stats.items()})
        records.append(row)

    if not records:
        return None
    return pd.DataFrame(records)


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
    """AUROC using the Mann-Whitney U statistic (exact, no sklearn dependency).

    Interprets higher signal value as predicting presence. Returns NaN if
    either class is empty.
    """
    n_p, n_a = len(pres_vals), len(abs_vals)
    if n_p == 0 or n_a == 0:
        return np.nan
    # Count pairs where presence > absence (ties count as 0.5)
    u = 0.0
    for pv in pres_vals:
        u += np.sum(pv > abs_vals) + 0.5 * np.sum(pv == abs_vals)
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

        frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for region_id, label in site.regions:
            frame = _load_region(region_id, label, signal)
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

        combined = pd.concat(frames, ignore_index=True)

        # Apply minimum obs filter
        n_obs_col = f"{signal.name}_n_obs"
        if n_obs_col in combined.columns:
            combined = combined[combined[n_obs_col] >= min_obs_per_year]

        if score_col not in combined.columns:
            raise ValueError(
                f"rank_key {rank_key!r} not in summarise() output. "
                f"Available columns: {[c for c in combined.columns if c.startswith(signal.name)]}"
            )

        pres = combined[combined["label"] == "presence"]
        abs_ = combined[combined["label"] == "absence"]

        pres_vals = pres[score_col].dropna().values
        abs_vals  = abs_[score_col].dropna().values

        pres_stats = _class_stats("presence", pres_vals)
        abs_stats  = _class_stats("absence",  abs_vals)
        pres_stats.n_pixels = pres["point_id"].nunique()
        abs_stats.n_pixels  = abs_["point_id"].nunique()

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
