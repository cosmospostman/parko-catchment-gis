"""signals/integral.py — Annual NDVI integral signal.

Metric: mean annual smoothed NDVI across all DOYs in the year, averaged over
reliable years.  Captures total ``duration × magnitude`` of greenness — a
quantity neither ``rec_p`` (amplitude) nor ``nir_cv`` (dry-season variability)
measures directly.

Primary features
----------------
ndvi_integral        — mean of the smoothed NDVI curve across all DOYs,
                       averaged over reliable pixel-years
ndvi_integral_cv     — CV of per-year mean NDVI across reliable years
                       (lower = more consistent annual greenness)

Mechanistic interpretation
--------------------------
Parkinsonia's photosynthetic bark maintains a non-zero NDVI floor even during
complete leaf drop, so the curve never collapses to zero.  The integral is
therefore high for Parkinsonia relative to grasses and semi-deciduous shrubs
that spike briefly and crash — even when their ``rec_p`` amplitudes are similar.

The "large integral" framing comes directly from the literature on Parkinsonia
phenological discrimination: P. aculeata occupies more of the phenological
space throughout the year, producing a high-mean, low-amplitude waveform.

Relationship to existing signals
---------------------------------
``rec_p``  measures peak − floor amplitude.  High floor contributes to both
           ``rec_p`` and ``ndvi_integral``, but two pixels with identical
           ``rec_p`` can differ substantially in integral depending on curve
           width.  The integral is not redundant — it carries independent
           information about curve shape.

``nir_cv`` measures inter-annual dry-season NIR variability — a different axis
           entirely.  Expected correlation is low.

Note on ``ndvi_integral_cv``
----------------------------
This is the inter-annual CV of *mean annual NDVI*, not to be confused with
``nir_cv`` which is the inter-annual CV of *dry-season NIR*.  Both are
consistency metrics but on different bands, seasons, and aggregation windows.

Implementation note
-------------------
The integral is computed as ``mean(ndvi_smooth)`` across all observations in
the calendar year — equivalent to the area under the curve divided by year
length.  This normalisation makes pixel-years with different observation counts
comparable: the smoothed curve already interpolates across cloud gaps, so the
mean is cloud-insensitive.

Reliable years are defined by ``is_sparse_year`` from the curve frame (set by
``annual_ndvi_curve`` / ``annual_ndvi_curve_chunked`` when observations per year
fall below ``QualityParams.min_obs_per_year``).  No separate obs-counting is
needed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from signals import QualityParams
from signals._shared import load_and_filter, annual_ndvi_curve


class NdviIntegralSignal:
    """Annual NDVI integral (mean smoothed NDVI) signal."""

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)
        smooth_days: int = 30       # must match the curve used as input
        min_years: int = 3          # minimum reliable years for per-pixel stats

    def __init__(self, params: "NdviIntegralSignal.Params | None" = None) -> None:
        self.params = params or NdviIntegralSignal.Params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _integral_per_year(self, curve: "pl.DataFrame | Path") -> pd.DataFrame:
        """Per-pixel per-year mean smoothed NDVI.

        Reliable years are those where ``is_sparse_year`` is False — consistent
        with the definition used by GreenupTimingSignal and RecessionSensitivitySignal.

        ``curve`` may be a ``pl.DataFrame`` or a ``Path`` to a parquet file
        written by ``annual_ndvi_curve_chunked``.  When a ``Path`` is given
        the groupby is collected with streaming so the full observation-level
        curve is never loaded into RAM.

        Returns a pandas DataFrame with columns:
            point_id, year, ndvi_mean, n_obs, reliable
        """
        if isinstance(curve, Path):
            base = pl.scan_parquet(curve)
            use_streaming = True
        else:
            base = curve.lazy()
            use_streaming = False

        agg_lazy = (
            base
            .group_by(["point_id", "year"])
            .agg([
                pl.col("ndvi_smooth").mean().alias("ndvi_mean"),
                pl.len().alias("n_obs"),
                pl.col("is_sparse_year").first().alias("is_sparse_year"),
            ])
            .with_columns(
                (~pl.col("is_sparse_year")).alias("reliable")
            )
            .drop("is_sparse_year")
        )

        if use_streaming:
            result = agg_lazy.collect(engine="streaming")
        else:
            result = agg_lazy.collect()

        return result.to_pandas()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        pixel_df: pd.DataFrame | None,
        loc: object,
        _df: "pl.DataFrame | None" = None,
        _curve: "pl.DataFrame | Path | None" = None,
    ) -> pd.DataFrame:
        """Compute NDVI integral features per pixel.

        Parameters
        ----------
        pixel_df:
            Raw observation parquet loaded for this location, or None if
            ``_curve`` is provided.
        loc:
            ``utils.location.Location``.
        _df:
            Optional pre-filtered Polars DataFrame (output of
            ``load_and_filter``).  If provided, skips the filter step.
        _curve:
            Optional pre-computed NDVI curve.  May be a ``pl.DataFrame``
            (output of ``annual_ndvi_curve``) or a ``Path`` to a parquet
            written by ``annual_ndvi_curve_chunked``.  When a ``Path`` is
            given the groupby uses streaming so the full curve is never
            loaded into RAM.

        Returns
        -------
        DataFrame with columns:
            ``[point_id, lon, lat, ndvi_integral, ndvi_integral_cv,
               n_years, n_reliable_years]``.

        ``ndvi_integral`` and ``ndvi_integral_cv`` are NaN for pixels with
        fewer than ``min_years`` reliable years.
        """
        p = self.params

        if _curve is not None:
            curve = _curve
            if isinstance(curve, Path):
                coords = (
                    pl.scan_parquet(curve)
                    .select(["point_id", "lon", "lat"])
                    .unique("point_id")
                    .collect()
                    .to_pandas()
                )
            else:
                coords = curve.select(["point_id", "lon", "lat"]).unique("point_id").to_pandas()
        else:
            df = _df if _df is not None else load_and_filter(pixel_df, p.quality.scl_purity_min)
            coords = df.select(["point_id", "lon", "lat"]).unique("point_id").to_pandas()
            curve = annual_ndvi_curve(df, p.smooth_days, p.quality.min_obs_per_year)

        per_year = self._integral_per_year(curve)
        per_year_pl = pl.from_pandas(per_year)

        min_years = p.min_years

        n_years_pl = (
            per_year_pl
            .group_by("point_id")
            .agg(pl.len().alias("n_years"))
        )

        reliable_pl = per_year_pl.filter(pl.col("reliable"))
        stats_pl = (
            reliable_pl
            .group_by("point_id")
            .agg([
                pl.col("ndvi_mean").mean().alias("ndvi_integral_mean"),
                pl.col("ndvi_mean").std().alias("ndvi_integral_std"),
                pl.col("ndvi_mean").count().alias("n_reliable_years"),
            ])
            .with_columns([
                pl.when(pl.col("n_reliable_years") >= min_years)
                  .then(pl.col("ndvi_integral_mean"))
                  .otherwise(pl.lit(None, dtype=pl.Float64))
                  .alias("ndvi_integral"),
                pl.when(
                    (pl.col("n_reliable_years") >= min_years) &
                    (pl.col("ndvi_integral_mean") > 0)
                )
                  .then(pl.col("ndvi_integral_std") / pl.col("ndvi_integral_mean"))
                  .otherwise(pl.lit(None, dtype=pl.Float64))
                  .alias("ndvi_integral_cv"),
            ])
            .drop(["ndvi_integral_mean", "ndvi_integral_std"])
            .join(n_years_pl, on="point_id", how="left")
        )

        # Pixels with zero reliable years won't appear in reliable_pl — fill
        # via n_years_pl so every pixel gets a row.
        stats_pl = (
            n_years_pl
            .join(stats_pl.drop("n_years"), on="point_id", how="left")
            .with_columns(
                pl.col("n_reliable_years").fill_null(0),
            )
        )

        coords_pl = pl.from_pandas(coords)
        stats_pl = stats_pl.join(coords_pl, on="point_id", how="left")

        col_order = [
            "point_id", "lon", "lat",
            "ndvi_integral", "ndvi_integral_cv",
            "n_years", "n_reliable_years",
        ]
        return stats_pl.select(col_order).to_pandas()

    def compute_from_path(
        self,
        path: Path,
        loc: object,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> pd.DataFrame:
        """Memory-efficient compute for large pixel-sorted parquets.

        Uses ``annual_ndvi_curve_chunked`` to build the NDVI curve row-group
        by row-group (peak RAM ≈ one row group), writes it to a temp file, then
        aggregates with streaming.  Never loads the full dataset into RAM.

        Parameters
        ----------
        path:
            Path to a pixel-sorted parquet file.
        loc:
            ``utils.location.Location``.
        year_from, year_to:
            Optional inclusive year bounds applied before curve computation.

        Returns
        -------
        Same columns as ``compute()``.
        """
        import tempfile
        from signals._shared import annual_ndvi_curve_chunked

        p = self.params

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            curve_path = Path(f.name)

        try:
            annual_ndvi_curve_chunked(
                sorted_parquet_path=path,
                out_path=curve_path,
                smooth_days=p.smooth_days,
                min_obs_per_year=p.quality.min_obs_per_year,
                scl_purity_min=p.quality.scl_purity_min,
                year_from=year_from,
                year_to=year_to,
            )
            return self.compute(pixel_df=None, loc=loc, _curve=curve_path)
        finally:
            curve_path.unlink(missing_ok=True)

    def diagnose(
        self,
        pixel_df: pd.DataFrame | None,
        loc: object,
        out_dir: Path | None = None,
        _curve: "pl.DataFrame | Path | None" = None,
    ) -> dict:
        """Compute signal and write standard diagnostic figures.

        Figures written:
          - map_ndvi_integral.png     — spatial map of ndvi_integral
          - map_ndvi_integral_cv.png  — spatial map of ndvi_integral_cv
          - distributions.png         — histograms split by presence/absence

        Returns
        -------
        dict with keys: ``signal``, ``site``, ``n_pixels``,
        ``presence_median``, ``absence_median``, ``separability``, ``figures``.
        """
        from signals.diagnostics import (
            plot_signal_map,
            plot_distributions,
            separability_score,
            _resolve_classes,
        )

        stats = self.compute(pixel_df, loc, _curve=_curve)

        if out_dir is None:
            _root = Path(__file__).resolve().parent.parent
            out_dir = _root / "outputs" / f"{loc.id}-ndvi-integral"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        presence_ids, absence_ids = _resolve_classes(stats, loc)
        sep = separability_score(stats, "ndvi_integral", presence_ids, absence_ids)

        figures = []

        fig = plot_signal_map(
            stats, "ndvi_integral", loc,
            title=(
                f"{loc.name} — Mean annual NDVI (integral)\n"
                "higher = more sustained greenness across the year"
            ),
            out_path=out_dir / "map_ndvi_integral.png",
            colormap="YlGn",
        )
        if fig is not None:
            figures.append(out_dir / "map_ndvi_integral.png")

        fig = plot_signal_map(
            stats, "ndvi_integral_cv", loc,
            title=(
                f"{loc.name} — NDVI integral CV\n"
                "lower = more consistent inter-annual mean greenness"
            ),
            out_path=out_dir / "map_ndvi_integral_cv.png",
            colormap="YlOrRd",
        )
        if fig is not None:
            figures.append(out_dir / "map_ndvi_integral_cv.png")

        fig = plot_distributions(
            stats, "ndvi_integral", loc,
            presence_ids=presence_ids,
            absence_ids=absence_ids,
            out_path=out_dir / "distributions.png",
        )
        if fig is not None:
            figures.append(out_dir / "distributions.png")

        return {
            "signal": "ndvi_integral",
            "site": loc.id,
            "n_pixels": len(stats),
            "presence_median": (
                stats.loc[stats["point_id"].isin(presence_ids), "ndvi_integral"].median()
                if presence_ids is not None else None
            ),
            "absence_median": (
                stats.loc[stats["point_id"].isin(absence_ids), "ndvi_integral"].median()
                if absence_ids is not None else None
            ),
            "separability": sep,
            "figures": figures,
        }
