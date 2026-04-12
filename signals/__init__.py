"""signals — Parkinsonia spectral/temporal signal abstractions.

Exports
-------
QualityParams         — shared quality-filter dataclass (used by all signals)
NirCvSignal           — dry-season NIR inter-annual CV
RecPSignal            — wet/dry seasonal NDVI amplitude
RedEdgeSignal         — red-edge ratio annual floor
SwirSignal            — SWIR moisture index annual floor
FloweringSignal       — flowering flash detection (DOY anomaly algorithm)
extract_parko_features — convenience function: all four tabular features in one pass

Usage
-----
from utils.location import get
from signals import NirCvSignal, QualityParams

loc    = get("longreach")
df     = pd.read_parquet(loc.parquet_path())
result = NirCvSignal().diagnose(df, loc, out_dir=Path("outputs/longreach-nir-cv"))
print(result["separability"])

# Site-specific params loaded from location YAML:
from signals._shared import load_signal_params
params = load_signal_params(loc, "nir_cv")
result = NirCvSignal(params).diagnose(df, loc)

# All four tabular features:
from signals import extract_parko_features
features = extract_parko_features(df, loc)
# → DataFrame with [point_id, lon, lat, nir_cv, rec_p, re_p10, swir_p10]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# Shared quality params — exported at package level
# ---------------------------------------------------------------------------

@dataclass
class QualityParams:
    """Quality-filter parameters shared by all signal Params dataclasses."""
    scl_purity_min: float = 0.5
    min_obs_per_year: int = 10
    min_obs_dry: int = 5


# ---------------------------------------------------------------------------
# Signal classes — imported after QualityParams to avoid circular imports
# ---------------------------------------------------------------------------

from signals.nir_cv import NirCvSignal                        # noqa: E402
from signals.wet_dry_amp import RecPSignal                    # noqa: E402
from signals.red_edge import RedEdgeSignal                    # noqa: E402
from signals.swir import SwirSignal                           # noqa: E402
from signals.flowering import FloweringSignal                 # noqa: E402
from signals.recession import RecessionSensitivitySignal      # noqa: E402
from signals.greenup import GreenupTimingSignal               # noqa: E402
from signals.tuning import sweep_signal                       # noqa: E402


# ---------------------------------------------------------------------------
# Convenience: all four tabular features in one pass
# ---------------------------------------------------------------------------

def extract_parko_features(
    pixel_df: pd.DataFrame,
    loc: object,
    nir_cv_params: NirCvSignal.Params | None = None,
    rec_p_params: RecPSignal.Params | None = None,
    red_edge_params: RedEdgeSignal.Params | None = None,
    swir_params: SwirSignal.Params | None = None,
) -> pd.DataFrame:
    """Compute all four tabular signal features and join into one per-pixel table.

    FloweringSignal is excluded — it requires a separate algorithm and is run
    independently via ``FloweringSignal().compute(df, loc)``.

    Site-specific param overrides are loaded automatically from the location's
    YAML (``signals:`` section) unless explicit Params are passed in.

    Parameters
    ----------
    pixel_df:
        Raw observation parquet loaded for this location.
    loc:
        ``utils.location.Location``.
    nir_cv_params, rec_p_params, red_edge_params, swir_params:
        Optional per-signal Params overrides. If None, site params are loaded
        from the location YAML via ``load_signal_params``.

    Returns
    -------
    DataFrame with columns ``[point_id, lon, lat, nir_cv, rec_p, re_p10, swir_p10]``.
    """
    from signals._shared import load_signal_params

    if nir_cv_params is None:
        nir_cv_params = load_signal_params(loc, "nir_cv")
    if rec_p_params is None:
        rec_p_params = load_signal_params(loc, "rec_p")
    if red_edge_params is None:
        red_edge_params = load_signal_params(loc, "red_edge")
    if swir_params is None:
        swir_params = load_signal_params(loc, "swir")

    nir_stats = NirCvSignal(nir_cv_params).compute(pixel_df, loc)
    rec_stats = RecPSignal(rec_p_params).compute(pixel_df, loc)
    re_stats = RedEdgeSignal(red_edge_params).compute(pixel_df, loc)
    swir_stats = SwirSignal(swir_params).compute(pixel_df, loc)

    result = (
        nir_stats[["point_id", "lon", "lat", "nir_cv"]]
        .merge(rec_stats[["point_id", "rec_p"]], on="point_id", how="outer")
        .merge(re_stats[["point_id", "re_p10"]], on="point_id", how="outer")
        .merge(swir_stats[["point_id", "swir_p10"]], on="point_id", how="outer")
    )

    # Fill lon/lat from any signal that has them (outer join may leave gaps)
    for stats_df in [rec_stats, re_stats, swir_stats]:
        missing = result["lon"].isna()
        if missing.any():
            coords = stats_df[["point_id", "lon", "lat"]].set_index("point_id")
            result.loc[missing, "lon"] = result.loc[missing, "point_id"].map(coords["lon"])
            result.loc[missing, "lat"] = result.loc[missing, "point_id"].map(coords["lat"])

    return result[["point_id", "lon", "lat", "nir_cv", "rec_p", "re_p10", "swir_p10"]]


__all__ = [
    "QualityParams",
    "NirCvSignal",
    "RecPSignal",
    "RedEdgeSignal",
    "SwirSignal",
    "FloweringSignal",
    "RecessionSensitivitySignal",
    "GreenupTimingSignal",
    "extract_parko_features",
    "sweep_signal",
]
