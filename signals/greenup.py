"""signals/greenup.py — Green-up peak timing signal.

Metric: day-of-year (DOY) at which annual NDVI peaks, and how consistent
that timing is across years.

Primary features
----------------
peak_doy       — mean DOY of annual NDVI peak across reliable years
peak_doy_cv    — CV of peak DOY across years (lower = more consistent)
greenup_rate   — mean OLS slope of NDVI on the rising limb from the
                 pre-peak trough to the peak (lower confidence — sparse
                 wet-season observations; interpret cautiously)

Mechanistic interpretation
--------------------------
Parkinsonia's deep roots buffer its flush timing from rainfall variability.
The existing red-edge and NDVI contrast analyses show a consistent March–April
peak across all six Longreach years. Grasses track monsoon onset more closely,
so their peak DOY varies more year to year (higher CV) and may differ in
mean timing.

Caveat: the wet season (Dec–Feb) is the cloud-heavy period at this latitude.
Peak DOY estimates are noisier for years with fewer clean wet-season
acquisitions. Years with fewer than ``min_wet_obs`` clean observations in the
search window are flagged via ``n_reliable_years``; per-pixel estimates are
based only on reliable years.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from signals import QualityParams
from signals._shared import load_and_filter, annual_ndvi_curve


class GreenupTimingSignal:
    """Green-up peak timing and consistency signal."""

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)
        search_start_month: int = 11   # November — allow early green-up
        search_end_month: int = 5      # May — allow late peak
        smooth_days: int = 30
        min_wet_obs: int = 5           # minimum obs in search window to trust peak estimate
        min_years: int = 3             # minimum reliable years for per-pixel stats
        trough_window_days: int = 60   # look-back from peak to find pre-peak trough

    def __init__(self, params: GreenupTimingSignal.Params | None = None) -> None:
        self.params = params or GreenupTimingSignal.Params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _peak_doy_per_year(self, curve: pl.DataFrame) -> pl.DataFrame:
        """Per-pixel per-year peak DOY within the search window.

        The search window wraps the year boundary (Nov Y → May Y+1). We
        assign the peak to the calendar year of its *end* (the year
        containing May), so November and December observations are shifted
        forward one year.

        Returns Polars DataFrame with columns:
            point_id, year, peak_doy, n_search_obs, greenup_rate, reliable
        """
        p = self.params

        if p.search_start_month <= p.search_end_month:
            search_months = list(range(p.search_start_month, p.search_end_month + 1))
        else:
            search_months = (
                list(range(p.search_start_month, 13)) +
                list(range(1, p.search_end_month + 1))
            )

        # Shift months outside search_end_month forward one year so the window
        # is contiguous (Nov/Dec → year+1).
        search = (
            curve
            .filter(pl.col("month").is_in(search_months))
            .with_columns(
                pl.when(pl.col("month") <= p.search_end_month)
                  .then(pl.col("year"))
                  .otherwise(pl.col("year") + 1)
                  .alias("search_year")
            )
        )

        # Per-(point_id, search_year): count obs, argmax NDVI → peak_doy + peak_date
        # Reliable = n_obs >= min_wet_obs.
        # greenup_rate: OLS slope on rising limb (trough_window_days before peak).
        # map_elements is called once per pixel-year group (not per row).
        trough_days = p.trough_window_days
        min_wet = p.min_wet_obs

        def _group_stats(row) -> dict:
            doys = np.asarray(row["doy"], dtype=float)
            ndvi = np.asarray(row["ndvi_smooth"], dtype=float)
            dates = np.asarray(row["date_i"], dtype=np.int64)  # days since epoch
            n_obs = len(doys)

            if n_obs < min_wet:
                return {
                    "peak_doy": float("nan"),
                    "greenup_rate": float("nan"),
                    "reliable": False,
                }

            idx_peak = int(np.argmax(ndvi))
            peak_doy = float(doys[idx_peak])
            peak_date_i = dates[idx_peak]
            trough_cutoff_i = peak_date_i - trough_days  # days

            mask = (dates >= trough_cutoff_i) & (dates <= peak_date_i)
            greenup_rate = float("nan")
            if mask.sum() >= 3:
                x = doys[mask]
                y = ndvi[mask]
                xm = x.mean()
                denom = ((x - xm) ** 2).sum()
                if denom > 0:
                    greenup_rate = float(((x - xm) * (y - y.mean())).sum() / denom)

            return {
                "peak_doy": peak_doy,
                "greenup_rate": greenup_rate,
                "reliable": True,
            }

        # Convert date to integer days for cheap arithmetic inside map_elements
        result = (
            search
            .with_columns(
                (pl.col("date").cast(pl.Date).cast(pl.Int32)).alias("date_i")
            )
            .sort(["point_id", "search_year", "date"])
            .group_by(["point_id", "search_year"])
            .agg([
                pl.col("doy").alias("doy"),
                pl.col("ndvi_smooth").alias("ndvi_smooth"),
                pl.col("date_i").alias("date_i"),
                pl.len().alias("n_search_obs"),
            ])
            .with_columns(
                pl.struct(["doy", "ndvi_smooth", "date_i"])
                  .map_elements(_group_stats, return_dtype=pl.Struct({
                      "peak_doy": pl.Float64,
                      "greenup_rate": pl.Float64,
                      "reliable": pl.Boolean,
                  }))
                  .alias("_stats")
            )
            .unnest("_stats")
            .drop(["doy", "ndvi_smooth", "date_i"])
            .rename({"search_year": "year"})
        )

        return result.to_pandas()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
        _df: pl.DataFrame | None = None,
        _curve: pl.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute green-up timing features per pixel.

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
            Optional pre-computed NDVI curve (output of
            ``annual_ndvi_curve``). If provided, skips the smoothing step.

        Returns
        -------
        DataFrame with columns:
            ``[point_id, lon, lat, peak_doy, peak_doy_cv, greenup_rate,
               n_years, n_reliable_years]``.

        ``peak_doy`` and ``peak_doy_cv`` are computed from reliable years
        only. ``greenup_rate`` is lower confidence (sparse wet-season obs).
        """
        p = self.params
        df = _df if _df is not None else load_and_filter(pixel_df, p.quality.scl_purity_min)

        coords = (
            df.select(["point_id", "lon", "lat"])
            .unique("point_id")
            .to_pandas()
        )

        curve = _curve if _curve is not None else annual_ndvi_curve(df, p.smooth_days, p.quality.min_obs_per_year)
        per_year = self._peak_doy_per_year(curve)  # pandas DataFrame
        per_year_pl = pl.from_pandas(per_year)

        min_years = p.min_years

        # n_years counts all pixel-year windows (reliable + unreliable)
        n_years_pl = (
            per_year_pl
            .group_by("point_id")
            .agg(pl.len().alias("n_years"))
        )

        # Stats from reliable years only
        reliable_pl = per_year_pl.filter(pl.col("reliable"))
        stats_pl = (
            reliable_pl
            .group_by("point_id")
            .agg([
                pl.col("peak_doy").mean().alias("peak_doy_mean"),
                pl.col("peak_doy").std().alias("peak_doy_std"),
                pl.col("peak_doy").count().alias("n_reliable_years"),
                pl.col("greenup_rate").drop_nulls().mean().alias("greenup_rate"),
            ])
            .with_columns([
                pl.when(pl.col("n_reliable_years") >= min_years)
                  .then(pl.col("peak_doy_mean"))
                  .otherwise(pl.lit(None))
                  .alias("peak_doy"),
                pl.when(
                    (pl.col("n_reliable_years") >= min_years) &
                    (pl.col("peak_doy_mean") != 0)
                  )
                  .then(pl.col("peak_doy_std") / pl.col("peak_doy_mean"))
                  .otherwise(pl.lit(None))
                  .alias("peak_doy_cv"),
                pl.when(pl.col("n_reliable_years") >= min_years)
                  .then(pl.col("greenup_rate"))
                  .otherwise(pl.lit(None))
                  .alias("greenup_rate"),
            ])
            .drop(["peak_doy_mean", "peak_doy_std"])
            .join(n_years_pl, on="point_id", how="left")
        )

        # Pixels with zero reliable years won't appear in reliable_pl — join
        # back via n_years_pl (which covers all pixels) so the output is complete.
        # n_years_pl already dropped n_years from stats_pl join above, so no suffix clash.
        stats_pl = (
            n_years_pl
            .join(stats_pl.drop("n_years"), on="point_id", how="left")
            .with_columns(
                pl.col("n_reliable_years").fill_null(0),
            )
        )

        coords_pl = df.select(["point_id", "lon", "lat"]).unique("point_id")
        stats_pl = stats_pl.join(coords_pl, on="point_id", how="left")

        col_order = [
            "point_id", "lon", "lat",
            "peak_doy", "peak_doy_cv", "greenup_rate",
            "n_years", "n_reliable_years",
        ]
        return stats_pl.select(col_order).to_pandas()

    def diagnose(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
        out_dir: Path | None = None,
    ) -> dict:
        """Compute signal and write standard diagnostic figures.

        Figures written:
          - map_peak_doy.png      — spatial map of peak_doy
          - map_peak_doy_cv.png   — spatial map of peak_doy_cv
          - distributions.png     — histograms split by presence/absence

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
            out_dir = _root / "outputs" / f"{loc.id}-greenup"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        presence_ids, absence_ids = _resolve_classes(stats, loc)
        sep = separability_score(stats, "peak_doy", presence_ids, absence_ids)

        figures = []

        fig = plot_signal_map(
            stats, "peak_doy", loc,
            title=(
                f"{loc.name} — Mean annual NDVI peak DOY\n"
                "earlier = sooner post-wet flush"
            ),
            out_path=out_dir / "map_peak_doy.png",
            colormap="plasma",
        )
        if fig is not None:
            figures.append(out_dir / "map_peak_doy.png")

        fig = plot_signal_map(
            stats, "peak_doy_cv", loc,
            title=(
                f"{loc.name} — Peak DOY coefficient of variation\n"
                "lower = more consistent inter-annual timing"
            ),
            out_path=out_dir / "map_peak_doy_cv.png",
            colormap="YlOrRd",
        )
        if fig is not None:
            figures.append(out_dir / "map_peak_doy_cv.png")

        fig = plot_distributions(
            stats, "peak_doy", loc,
            presence_ids=presence_ids,
            absence_ids=absence_ids,
            out_path=out_dir / "distributions.png",
        )
        if fig is not None:
            figures.append(out_dir / "distributions.png")

        return {
            "signal": "peak_doy",
            "site": loc.id,
            "n_pixels": len(stats),
            "presence_median": (
                stats.loc[stats["point_id"].isin(presence_ids), "peak_doy"].median()
                if presence_ids is not None else None
            ),
            "absence_median": (
                stats.loc[stats["point_id"].isin(absence_ids), "peak_doy"].median()
                if absence_ids is not None else None
            ),
            "separability": sep,
            "figures": figures,
        }
