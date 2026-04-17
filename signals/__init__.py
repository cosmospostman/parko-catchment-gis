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
from pathlib import Path
from typing import Union

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
from signals.greenup import GreenupTimingSignal, GreenupShiftSignal  # noqa: E402
from signals.integral import NdviIntegralSignal               # noqa: E402
from signals.tuning import sweep_signal                       # noqa: E402


# ---------------------------------------------------------------------------
# Convenience: all four tabular features in one pass
# ---------------------------------------------------------------------------

def extract_parko_features(
    pixel_df: Union[pd.DataFrame, Path],
    loc: object,
    nir_cv_params: NirCvSignal.Params | None = None,
    rec_p_params: RecPSignal.Params | None = None,
    red_edge_params: RedEdgeSignal.Params | None = None,
    swir_params: SwirSignal.Params | None = None,
    ndvi_integral_params: NdviIntegralSignal.Params | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    calibration_path: Path | None = None,
) -> pd.DataFrame:
    """Compute all tabular signal features and join into one per-pixel table.

    FloweringSignal is excluded — it requires a separate algorithm and is run
    independently via ``FloweringSignal().compute(df, loc)``.

    Site-specific param overrides are loaded automatically from the location's
    YAML (``signals:`` section) unless explicit Params are passed in.

    Parameters
    ----------
    pixel_df:
        Raw observation parquet loaded for this location, or a Path to the
        parquet file (preferred for large files — avoids loading into RAM).
    loc:
        ``utils.location.Location``.
    nir_cv_params, rec_p_params, red_edge_params, swir_params, ndvi_integral_params:
        Optional per-signal Params overrides. If None, site params are loaded
        from the location YAML via ``load_signal_params``.
    bbox:
        Optional ``(lon_min, lat_min, lon_max, lat_max)`` spatial filter.
        When provided and ``pixel_df`` is a Path, only rows within this bbox
        are loaded into memory before feature extraction — use this when the
        parquet covers a much larger area than the pixels you actually need
        (e.g. extracting training pixels from a small sub-bbox inside a large
        scene parquet).  The in-memory DataFrame path is used, so the parquet
        does not need to be pixel-sorted.
    calibration_path:
        Optional path to a tile-harmonisation correction table (produced by
        ``utils.tile_harmonisation.calibrate``).  When provided, per-(tile,
        band, year) scale factors are applied before computing derived
        features.  Ignored on the in-memory DataFrame path (sub-bbox use).
        Pass ``loc.calibration_path()`` for automatic discovery.

    Returns
    -------
    DataFrame with columns
    ``[point_id, lon, lat, nir_cv, rec_p, re_p10, swir_p10, ndvi_integral]``.
    """
    from signals._shared import load_signal_params, compute_features_chunked

    if nir_cv_params is None:
        nir_cv_params = load_signal_params(loc, "nir_cv")
    if rec_p_params is None:
        rec_p_params = load_signal_params(loc, "rec_p")
    if red_edge_params is None:
        red_edge_params = load_signal_params(loc, "red_edge")
    if swir_params is None:
        swir_params = load_signal_params(loc, "swir")
    if ndvi_integral_params is None:
        ndvi_integral_params = load_signal_params(loc, "ndvi_integral")

    # When given a Path + bbox, scan row-by-row-group, filter to bbox, then use
    # the in-memory DataFrame path — avoids sorting a huge parquet when only a
    # small sub-region is needed (e.g. training pixels from a tiny sub-bbox).
    if isinstance(pixel_df, Path) and bbox is not None:
        import pyarrow.parquet as pq
        import polars as pl
        lon_min, lat_min, lon_max, lat_max = bbox
        pf = pq.ParquetFile(pixel_df)
        n_rg = pf.metadata.num_row_groups
        print(
            f"  [bbox-filter] scanning {pixel_df.name} ({n_rg} row groups) "
            f"for bbox {bbox} ..."
        )
        parts_pd = []
        for rg_idx in range(n_rg):
            chunk = pl.from_arrow(pf.read_row_groups([rg_idx]))
            filtered = chunk.filter(
                (pl.col("lon") >= lon_min) & (pl.col("lon") <= lon_max) &
                (pl.col("lat") >= lat_min) & (pl.col("lat") <= lat_max)
            )
            if not filtered.is_empty():
                parts_pd.append(filtered.to_pandas())
        pixel_df = pd.concat(parts_pd, ignore_index=True) if parts_pd else pd.DataFrame()
        n_px = pixel_df["point_id"].nunique() if len(pixel_df) else 0
        print(f"  [bbox-filter] {n_px:,} pixels retained")
        # pixel_df is now a DataFrame — fall through to the per-signal path below

    # When given a Path (full scene), use the memory-efficient chunked path.
    # When given a DataFrame (small/pre-filtered set), use the per-signal path.
    if isinstance(pixel_df, Path):
        from signals._shared import ensure_pixel_sorted
        pixel_df = ensure_pixel_sorted(pixel_df)

        base, integral_yearly_df = compute_features_chunked(
            path=pixel_df,
            scl_purity_min=nir_cv_params.quality.scl_purity_min,
            dry_months=loc.dry_months,
            min_obs_per_year=nir_cv_params.quality.min_obs_per_year,
            min_obs_dry=nir_cv_params.quality.min_obs_dry,
            re_floor_percentile=red_edge_params.floor_percentile,
            swir_floor_percentile=swir_params.floor_percentile,
            year_from=year_from,
            year_to=year_to,
            smooth_days=ndvi_integral_params.smooth_days,
            compute_ndvi_integral=True,
            calibration_path=calibration_path,
        )
        integral_stats = NdviIntegralSignal(ndvi_integral_params).compute(
            pixel_df=None, loc=loc, _per_year=integral_yearly_df,
        )
        return base.merge(integral_stats[["point_id", "ndvi_integral"]], on="point_id", how="left")

    if year_from is not None or year_to is not None:
        yr = pixel_df["date"].dt.year
        if year_from is not None:
            pixel_df = pixel_df[yr >= year_from]
        if year_to is not None:
            pixel_df = pixel_df[yr <= year_to]

    # NOTE: tile harmonisation corrections are NOT applied on this in-memory
    # path.  Corrections are only applied inside compute_features_chunked()
    # (the Path branch above).  This path is used for pre-filtered sub-bbox
    # DataFrames (training pixels), not full scenes, so the omission is
    # intentional — not a bug.
    nir_stats = NirCvSignal(nir_cv_params).compute(pixel_df, loc)
    rec_stats = RecPSignal(rec_p_params).compute(pixel_df, loc)
    re_stats = RedEdgeSignal(red_edge_params).compute(pixel_df, loc)
    swir_stats = SwirSignal(swir_params).compute(pixel_df, loc)
    integral_stats = NdviIntegralSignal(ndvi_integral_params).compute(pixel_df, loc)

    result = (
        nir_stats[["point_id", "lon", "lat", "nir_cv"]]
        .merge(rec_stats[["point_id", "rec_p"]], on="point_id", how="outer")
        .merge(re_stats[["point_id", "re_p10"]], on="point_id", how="outer")
        .merge(swir_stats[["point_id", "swir_p10"]], on="point_id", how="outer")
        .merge(integral_stats[["point_id", "ndvi_integral"]], on="point_id", how="outer")
    )

    # Fill lon/lat from any signal that has them (outer join may leave gaps)
    for stats_df in [rec_stats, re_stats, swir_stats, integral_stats]:
        missing = result["lon"].isna()
        if missing.any():
            coords = stats_df[["point_id", "lon", "lat"]].set_index("point_id")
            result.loc[missing, "lon"] = result.loc[missing, "point_id"].map(coords["lon"])
            result.loc[missing, "lat"] = result.loc[missing, "point_id"].map(coords["lat"])

    return result[["point_id", "lon", "lat", "nir_cv", "rec_p", "re_p10", "swir_p10", "ndvi_integral"]]


__all__ = [
    "QualityParams",
    "NirCvSignal",
    "RecPSignal",
    "RedEdgeSignal",
    "SwirSignal",
    "FloweringSignal",
    "RecessionSensitivitySignal",
    "GreenupTimingSignal",
    "GreenupShiftSignal",
    "NdviIntegralSignal",
    "extract_parko_features",
    "sweep_signal",
]
