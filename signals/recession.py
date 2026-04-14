"""signals/recession.py — Dry-season recession sensitivity signal.

Metric: how much does a pixel's dry-season NDVI recession slope vary with
the magnitude of the preceding wet season (proxied by peak NDWI)?

Primary features
----------------
recession_slope       — mean dry-season NDVI slope across years (less
                        negative = more persistent canopy)
recession_slope_cv    — CV of per-year slopes (lower = more consistent)
recession_sensitivity — Pearson r between per-year slope and per-year
                        wet-season peak NDWI (near zero = decoupled from
                        rainfall; negative = steep recession in dry years)

Mechanistic interpretation
--------------------------
Parkinsonia's deep roots sustain canopy through dry season regardless of
wet-season magnitude → shallow, consistent recession slope → near-zero
sensitivity.

Grassland tracks rainfall → steep recession after poor wet seasons, slower
after good ones → large CV, negative sensitivity.

Perennial riparian has permanent water access → shallow recession, but
also near-zero sensitivity (no year-to-year variation to respond to).
The two near-zero classes may still be separated by recession_slope level
or by GreenupTimingSignal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from signals import QualityParams
from signals._shared import load_and_filter, annual_ndvi_curve


class RecessionSensitivitySignal:
    """Dry-season recession slope and moisture-sensitivity signal."""

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)
        recession_start_month: int = 4   # April — post-wet flush ends
        recession_end_month: int = 9     # September — mid dry season
        wet_start_month: int = 12        # December — wet season onset
        wet_end_month: int = 3           # March — wet season close
        smooth_days: int = 30
        min_recession_obs: int = 5       # per pixel-year in recession window
        min_wet_obs: int = 3             # per pixel-year in wet window
        min_years: int = 3               # minimum valid years for sensitivity

    def __init__(self, params: RecessionSensitivitySignal.Params | None = None) -> None:
        self.params = params or RecessionSensitivitySignal.Params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wet_ndwi_peak(self, df: pl.DataFrame) -> pd.DataFrame:
        """Per-pixel per-year peak NDWI in the wet-season window.

        The wet season straddles the year boundary (Dec Y-1 → Mar Y).
        We assign the wet season to the calendar year of its *end* (Mar Y),
        so December observations are shifted forward one year.
        """
        p = self.params

        df = df.with_columns([
            ((pl.col("B03") - pl.col("B11")) / (pl.col("B03") + pl.col("B11"))).alias("ndwi"),
        ])

        # Assign wet-season year: Dec belongs to the following year
        df = df.with_columns([
            pl.when(pl.col("month") == 12)
              .then(pl.col("year") + 1)
              .otherwise(pl.col("year"))
              .alias("wet_year"),
        ])

        if p.wet_start_month <= p.wet_end_month:
            wet_months = list(range(p.wet_start_month, p.wet_end_month + 1))
        else:
            # wraps year boundary
            wet_months = list(range(p.wet_start_month, 13)) + list(range(1, p.wet_end_month + 1))

        wet_df = df.filter(pl.col("month").is_in(wet_months))

        stats = (
            wet_df.group_by(["point_id", "wet_year"])
            .agg([
                pl.col("ndwi").max().alias("ndwi_peak"),
                pl.col("ndwi").count().alias("_n_wet"),
            ])
            .filter(pl.col("_n_wet") >= p.min_wet_obs)
            .rename({"wet_year": "year"})
            .drop("_n_wet")
        )
        return stats.to_pandas()

    def _recession_slopes(self, curve: "pl.DataFrame | Path") -> pd.DataFrame:
        """OLS slope of smoothed NDVI vs DOY within the recession window.

        Returns one row per (point_id, year) with columns:
            point_id, year, recession_slope, n_recession_obs
        Years with fewer than min_recession_obs are excluded.

        OLS is computed entirely in Polars via sufficient-statistic aggregation —
        no Python UDF, no list columns, fully vectorised across all groups.
        slope = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)

        ``curve`` may be a ``pl.DataFrame`` or a ``Path`` to a parquet file
        written by ``annual_ndvi_curve_chunked``.  When a ``Path`` is given,
        the groupby is collected with ``engine='streaming'`` to avoid loading
        the full 267M-row curve into RAM.
        """
        p = self.params
        rec_months = list(range(p.recession_start_month, p.recession_end_month + 1))

        if isinstance(curve, Path):
            rec = (
                pl.scan_parquet(curve)
                .filter(pl.col("month").is_in(rec_months) & ~pl.col("is_sparse_year"))
            )
            use_streaming = True
        else:
            rec = curve.filter(
                pl.col("month").is_in(rec_months) & ~pl.col("is_sparse_year")
            )
            use_streaming = False

        # Collect sufficient statistics per (pixel, year) — streaming-compatible.
        # OLS slope = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)  — no list columns needed.
        doy_f = pl.col("doy").cast(pl.Float64)
        ndvi_f = pl.col("ndvi_smooth").cast(pl.Float64)

        agg_lazy = (
            rec.group_by(["point_id", "year"])
            .agg([
                pl.len().alias("n_obs"),
                doy_f.sum().alias("sum_x"),
                (doy_f * doy_f).sum().alias("sum_x2"),
                ndvi_f.sum().alias("sum_y"),
                (doy_f * ndvi_f).sum().alias("sum_xy"),
            ])
            .filter(pl.col("n_obs") >= p.min_recession_obs)
        )

        if use_streaming:
            agg = agg_lazy.collect(engine="streaming")
        else:
            agg = agg_lazy.collect() if isinstance(agg_lazy, pl.LazyFrame) else agg_lazy

        n = pl.col("n_obs").cast(pl.Float64)
        slopes = (
            agg.with_columns([
                pl.when((n * pl.col("sum_x2") - pl.col("sum_x") ** 2) > 0)
                  .then(
                      (n * pl.col("sum_xy") - pl.col("sum_x") * pl.col("sum_y")) /
                      (n * pl.col("sum_x2") - pl.col("sum_x") ** 2)
                  )
                  .otherwise(pl.lit(None, dtype=pl.Float64))
                  .alias("recession_slope"),
            ])
            .drop(["sum_x", "sum_x2", "sum_y", "sum_xy"])
            .rename({"n_obs": "n_recession_obs"})
        )

        return slopes.to_pandas()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
        _df: pl.DataFrame | None = None,
        _curve: "pl.DataFrame | Path | None" = None,
    ) -> pd.DataFrame:
        """Compute recession sensitivity features per pixel.

        Parameters
        ----------
        pixel_df:
            Raw observation parquet loaded for this location.
        loc:
            ``utils.location.Location``.
        _df:
            Optional pre-filtered Polars DataFrame (output of
            ``load_and_filter``). If provided, skips the filter step.
        _curve:
            Optional pre-computed NDVI curve.  May be a ``pl.DataFrame``
            (output of ``annual_ndvi_curve``) or a ``Path`` to a parquet
            written by ``annual_ndvi_curve_chunked``.  When a ``Path`` is
            given the recession-slope groupby uses streaming so the full
            267M-row curve is never loaded into RAM.

        Returns
        -------
        DataFrame with columns:
            ``[point_id, lon, lat, recession_slope, recession_slope_cv,
               recession_sensitivity, n_years]``.

        ``recession_sensitivity`` is NaN for pixels with fewer than
        ``params.min_years`` valid year pairs or negligible NDWI range
        across years (< 0.02).
        """
        p = self.params
        df = _df if _df is not None else load_and_filter(pixel_df, p.quality.scl_purity_min)

        coords = (
            df.select(["point_id", "lon", "lat"])
            .unique("point_id")
            .to_pandas()
        )

        curve = _curve if _curve is not None else annual_ndvi_curve(df, p.smooth_days, p.quality.min_obs_per_year)
        slopes_df = self._recession_slopes(curve)
        ndwi_df = self._wet_ndwi_peak(df)

        # Join slopes with wet-season NDWI peak on (point_id, year)
        paired_pl = pl.from_pandas(
            slopes_df.merge(ndwi_df, on=["point_id", "year"], how="inner")
        ).drop_nulls(["recession_slope", "ndwi_peak"])

        # Per-pixel aggregations — mean and CV are pure Polars (vectorised).
        # Pearson r uses struct.map_elements: one Python call per pixel (not per
        # pixel-year), guarded by min_years and min NDWI range checks.
        min_years = p.min_years
        ndwi_range_min = 0.02

        def _pearson_r(row) -> float:
            s = np.asarray(row["_slopes_list"], dtype=float)
            n = np.asarray(row["_ndwi_list"], dtype=float)
            if len(s) < min_years:
                return float("nan")
            if n.max() - n.min() < ndwi_range_min:
                return float("nan")
            denom = np.std(s) * np.std(n)
            if denom == 0:
                return float("nan")
            return float(np.corrcoef(s, n)[0, 1])

        stats_pl = (
            paired_pl
            .group_by("point_id")
            .agg([
                pl.col("recession_slope").mean().alias("recession_slope"),
                (pl.col("recession_slope").std() / pl.col("recession_slope").mean().abs())
                  .alias("recession_slope_cv"),
                pl.col("recession_slope").alias("_slopes_list"),
                pl.col("ndwi_peak").alias("_ndwi_list"),
                pl.len().alias("n_years"),
            ])
            .with_columns([
                pl.struct(["_slopes_list", "_ndwi_list"])
                  .map_elements(_pearson_r, return_dtype=pl.Float64)
                  .alias("recession_sensitivity"),
            ])
            .drop(["_slopes_list", "_ndwi_list"])
        )

        stats = stats_pl.to_pandas()
        stats = stats.merge(coords, on="point_id", how="left")

        col_order = [
            "point_id", "lon", "lat",
            "recession_slope", "recession_slope_cv", "recession_sensitivity",
            "n_years",
        ]
        return stats[col_order]

    def diagnose(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
        out_dir: Path | None = None,
    ) -> dict:
        """Compute signal and write standard diagnostic figures.

        Figures written:
          - map_recession_slope.png    — spatial map of recession_slope
          - map_recession_sensitivity.png — spatial map of recession_sensitivity
          - distributions.png          — histograms split by presence/absence

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

        stats = self.compute(pixel_df, loc)

        if out_dir is None:
            _root = Path(__file__).resolve().parent.parent
            out_dir = _root / "outputs" / f"{loc.id}-recession"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        presence_ids, absence_ids = _resolve_classes(stats, loc)
        sep = separability_score(stats, "recession_slope", presence_ids, absence_ids)

        figures = []

        fig = plot_signal_map(
            stats, "recession_slope", loc,
            title=(
                f"{loc.name} — Dry-season NDVI recession slope\n"
                "less negative = more persistent canopy"
            ),
            out_path=out_dir / "map_recession_slope.png",
            colormap="RdYlGn",
        )
        if fig is not None:
            figures.append(out_dir / "map_recession_slope.png")

        fig = plot_signal_map(
            stats, "recession_sensitivity", loc,
            title=(
                f"{loc.name} — Recession sensitivity (r: slope vs NDWI peak)\n"
                "near zero = decoupled from rainfall; negative = rain-dependent"
            ),
            out_path=out_dir / "map_recession_sensitivity.png",
            colormap="RdBu",
        )
        if fig is not None:
            figures.append(out_dir / "map_recession_sensitivity.png")

        fig = plot_distributions(
            stats, "recession_slope", loc,
            presence_ids=presence_ids,
            absence_ids=absence_ids,
            out_path=out_dir / "distributions.png",
        )
        if fig is not None:
            figures.append(out_dir / "distributions.png")

        return {
            "signal": "recession_slope",
            "site": loc.id,
            "n_pixels": len(stats),
            "presence_median": (
                stats.loc[stats["point_id"].isin(presence_ids), "recession_slope"].median()
                if presence_ids is not None else None
            ),
            "absence_median": (
                stats.loc[stats["point_id"].isin(absence_ids), "recession_slope"].median()
                if absence_ids is not None else None
            ),
            "separability": sep,
            "figures": figures,
        }
