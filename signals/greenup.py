"""signals/greenup.py — Green-up peak timing signal.

Metric: day-of-year (DOY) at which annual NDVI peaks, and how consistent
that timing is across years.

Primary features
----------------
peak_doy           — mean DOY of annual NDVI peak across reliable years
peak_doy_cv        — CV of peak DOY across years (lower = more consistent)

Relative greenup shift features (GreenupShiftSignal)
-----------------------------------------------------
peak_doy_shift     — mean per-year offset of pixel peak DOY from same-year
                     neighbourhood median (negative = earlier than neighbours)
peak_doy_shift_sd  — SD of that offset across years (lower = stable offset)

Mechanistic interpretation
--------------------------
Parkinsonia's deep roots buffer its flush timing from rainfall variability.
The existing red-edge and NDVI contrast analyses show a consistent March–April
peak across all six Longreach years. Grasses track monsoon onset more closely,
so their peak DOY varies more year to year (higher CV) and may differ in
mean timing.

The relative shift signal (GreenupShiftSignal) controls for inter-annual
flood/rainfall timing by computing each pixel's peak offset *within its year*
relative to a local neighbourhood median. Pixels that consistently peak before
their neighbourhood (negative shift) are more likely Parkinsonia, which is
buffered from flood timing by its deep root system.

Neighbourhood gating: pixels whose mean annual NDVI amplitude (rec_p proxy:
per-year NDVI p90−p10, averaged) falls below the scene-wide 10th percentile
are excluded from neighbourhood median calculations to prevent bare-soil and
water pixels contaminating the reference.

Note on distance approximation: neighbourhood radius is specified in metres
and converted to degrees using a constant lat/lon scale at the scene centre
(cos-corrected longitude). Accurate to ~1% across the ~8 km Longreach scene;
may need revisiting for larger or higher-latitude scenes.

Caveat: the wet season (Dec–Feb) is the cloud-heavy period at this latitude.
Peak DOY estimates are noisier for years with fewer clean acquisitions.
Years with fewer than ``min_wet_obs`` observations are flagged via
``n_reliable_years``; per-pixel estimates are based only on reliable years.
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
        smooth_days: int = 30
        min_wet_obs: int = 5           # minimum obs per year to trust peak estimate
        min_years: int = 3             # minimum reliable years for per-pixel stats

    def __init__(self, params: GreenupTimingSignal.Params | None = None) -> None:
        self.params = params or GreenupTimingSignal.Params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _peak_doy_per_year(self, curve: "pl.DataFrame | Path") -> pl.DataFrame:
        """Per-pixel per-year peak DOY across all observations in the calendar year.

        Uses pure scalar aggregations — no list columns, no sort, no Python UDF —
        so the groupby is fully streaming-compatible.

        ``curve`` may be a ``pl.DataFrame`` or a ``Path`` to a parquet file
        written by ``annual_ndvi_curve_chunked``.  When a ``Path`` is given,
        the groupby is collected with streaming so the full observation-level
        curve is never loaded into RAM.

        Returns Polars DataFrame with columns:
            point_id, year, peak_doy, n_search_obs, reliable
        """
        p = self.params
        min_wet = p.min_wet_obs

        if isinstance(curve, Path):
            base = pl.scan_parquet(curve)
            use_streaming = True
        else:
            base = curve.lazy()
            use_streaming = False

        # peak_doy: DOY at which ndvi_smooth is maximised within the year.
        # Expressed as scalar aggs — streaming-compatible, no list columns.
        agg_lazy = (
            base
            .group_by(["point_id", "year"])
            .agg([
                pl.len().alias("n_search_obs"),
                pl.col("doy")
                  .filter(pl.col("ndvi_smooth") == pl.col("ndvi_smooth").max())
                  .first()
                  .cast(pl.Float64)
                  .alias("peak_doy"),
            ])
            .with_columns(
                pl.col("n_search_obs").ge(min_wet).alias("reliable")
            )
            .with_columns(
                pl.when(pl.col("reliable"))
                  .then(pl.col("peak_doy"))
                  .otherwise(pl.lit(None, dtype=pl.Float64))
                  .alias("peak_doy")
            )
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
        pixel_df: pd.DataFrame,
        loc: object,
        _df: pl.DataFrame | None = None,
        _curve: "pl.DataFrame | Path | None" = None,
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
            Optional pre-computed NDVI curve.  May be a ``pl.DataFrame``
            (output of ``annual_ndvi_curve``) or a ``Path`` to a parquet
            written by ``annual_ndvi_curve_chunked``.  When a ``Path`` is
            given the peak-DOY groupby uses streaming so the full
            observation-level curve is never loaded into RAM.

        Returns
        -------
        DataFrame with columns:
            ``[point_id, lon, lat, peak_doy, peak_doy_cv,
               n_years, n_reliable_years]``.

        ``peak_doy`` and ``peak_doy_cv`` are computed from reliable years only.
        """
        p = self.params

        if _curve is not None:
            curve = _curve
            # Extract coords from the curve parquet (has lon/lat) — avoids
            # touching df_filt during the expensive streaming collect.
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

        coords_pl = pl.from_pandas(coords)
        stats_pl = stats_pl.join(coords_pl, on="point_id", how="left")

        col_order = [
            "point_id", "lon", "lat",
            "peak_doy", "peak_doy_cv",
            "n_years", "n_reliable_years",
        ]
        return stats_pl.select(col_order).to_pandas()



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


# ---------------------------------------------------------------------------
# GreenupShiftSignal — relative greenup shift from neighbourhood
# ---------------------------------------------------------------------------

class GreenupShiftSignal:
    """Relative green-up shift: per-pixel offset from same-year neighbourhood median peak DOY.

    Algorithm
    ---------
    The neighbourhood median is a scene-level property, so it is pre-computed
    for every gated pixel in one vectorised pass and stored as a
    ``(point_id, year) → neighbourhood_median_peak_doy`` lookup table.
    Individual pixel shifts are then pure table lookups + subtraction — no
    repeated spatial queries per pixel or per year.

    Build step (``build_lookup``):
      1. Apply amplitude gate: exclude pixels below the scene-wide
         ``amp_gate_percentile`` of mean annual NDVI amplitude from the
         neighbour pool.
      2. Build a cKDTree over gated pixels (once).
      3. Query all gated pixels' neighbours within radius R (one vectorised
         ``query_ball_point`` call).
      4. For each year: gather each gated pixel's neighbours' peak DOYs as a
         ragged array, compute the median → store as a flat DataFrame indexed
         by ``(point_id, year)``.

    Compute step (``compute``):
      5. Merge ``per_year`` (pixel own peak DOY) against the lookup on
         ``(point_id, year)``.
      6. Offset = own peak DOY − neighbourhood median.
      7. Aggregate across reliable years → ``peak_doy_shift`` (mean),
         ``peak_doy_shift_sd`` (SD).

    The lookup is cached on the instance after ``build_lookup`` so that
    parameter sweeps that only vary ``min_years`` can skip the spatial step.
    Re-call ``build_lookup`` when ``radius_m`` or ``amp_gate_percentile`` change.

    See module docstring for notes on the distance approximation.
    """

    @dataclass
    class Params:
        quality: "QualityParams" = field(default_factory=lambda: __import__("signals", fromlist=["QualityParams"]).QualityParams())
        smooth_days: int = 30
        min_wet_obs: int = 5
        min_years: int = 3
        radius_m: float = 250.0            # neighbourhood search radius in metres
        amp_gate_percentile: float = 0.10  # exclude pixels below this scene-wide amplitude percentile

    def __init__(self, params: "GreenupShiftSignal.Params | None" = None) -> None:
        self.params = params or GreenupShiftSignal.Params()
        self._lookup: pd.DataFrame | None = None  # (point_id, year) → nbr_median

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deg_per_metre(centre_lat_deg: float) -> tuple[float, float]:
        """Return (deg_per_m_lat, deg_per_m_lon) at a given latitude.

        Uses a spherical Earth approximation (R = 6_371_000 m).
        Accurate to < 0.3% across Australia.
        """
        import math
        R = 6_371_000.0
        deg_per_m_lat = 180.0 / (math.pi * R)
        deg_per_m_lon = deg_per_m_lat / math.cos(math.radians(centre_lat_deg))
        return deg_per_m_lat, deg_per_m_lon

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_lookup(
        self,
        per_year: pd.DataFrame,
        coords: pd.DataFrame,
        amp_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Pre-compute neighbourhood median peak DOY for every gated pixel × year.

        The result is cached on ``self._lookup`` and returned.  Call this once
        per (radius_m, amp_gate_percentile) combination; re-use across
        ``compute()`` calls that only vary ``min_years``.

        Parameters
        ----------
        per_year:
            Per-pixel per-year peak DOY table (output of
            ``GreenupTimingSignal._peak_doy_per_year()``).
        coords:
            DataFrame with columns ``point_id, lon, lat`` (one row per pixel,
            covering the full scene — not just labelled pixels).
        amp_series:
            Series indexed by ``point_id`` giving mean annual NDVI amplitude
            (e.g. ``rec_p`` from the ranking CSV).  Pixels below the
            scene-wide ``amp_gate_percentile`` are excluded from the neighbour
            pool.  If None, all pixels in ``coords`` are eligible neighbours.

        Returns
        -------
        DataFrame with columns ``point_id, year, nbr_median`` — one row per
        gated pixel × year that has at least one neighbour with a reliable
        peak DOY.
        """
        from scipy.spatial import cKDTree

        p          = self.params
        coords_idx = coords.set_index("point_id")
        centre_lat = float(coords_idx["lat"].mean())
        deg_lat, deg_lon = self._deg_per_metre(centre_lat)

        # ---- Amplitude gate -----------------------------------------------
        if amp_series is not None:
            amp_thresh = float(amp_series.quantile(p.amp_gate_percentile))
            gated_mask = amp_series >= amp_thresh
            gated_ids  = amp_series[gated_mask].index
        else:
            gated_ids = coords_idx.index

        gated_coords = coords_idx.loc[gated_ids, ["lon", "lat"]]
        gated_xy = np.column_stack([
            gated_coords["lon"].values / deg_lon,
            gated_coords["lat"].values / deg_lat,
        ])
        gated_pids = gated_coords.index.to_numpy()  # positional index into gated_xy

        # ---- Spatial index (once) ------------------------------------------
        # Build tree over gated pixels.  The full neighbour matrix at
        # 10 m/pixel × 500 m radius ≈ 6 900 neighbours/pixel × 576 k pixels
        # would be ~15 GB — too large to materialise.  Instead: build the tree
        # once and process pixels in chunks.  Per chunk of CHUNK_PIX pixels:
        #   - query_ball_point → ragged neighbour lists for those pixels only
        #   - flatten to (flat_nbrs_chunk, pixel_labels_chunk) CSR arrays
        #   - per year: gather DOY values, self-null, pandas groupby median
        # Memory per chunk: CHUNK_PIX × ~6 900 nbrs × 8 B ≈ CHUNK_PIX × 55 kB.
        # At CHUNK_PIX = 2 000 → ~110 MB per chunk → acceptable.
        CHUNK_PIX = 2_000
        n_pix     = len(gated_pids)
        tree      = cKDTree(gated_xy)

        # ---- Pre-year data setup ------------------------------------------
        reliable   = per_year[per_year["reliable"] & per_year["point_id"].isin(set(gated_ids))]
        pid_to_pos = {pid: i for i, pid in enumerate(gated_pids)}
        years      = sorted(reliable["year"].unique())

        # Build doy_arr for each year up front (small: n_pix × 8 B × n_years)
        doy_by_year: dict[int, np.ndarray] = {}
        for yr, grp in reliable.groupby("year"):
            doy_arr  = np.full(n_pix, np.nan)
            pos_vals = np.fromiter(
                (pid_to_pos[pid] for pid in grp["point_id"] if pid in pid_to_pos),
                dtype=int,
            )
            doy_vals = grp.loc[grp["point_id"].isin(pid_to_pos), "peak_doy"].to_numpy(dtype=float)
            doy_arr[pos_vals] = doy_vals
            doy_by_year[int(yr)] = doy_arr

        # nbr_medians_by_year[yr][i] = neighbourhood median for pixel i, year yr
        nbr_medians_by_year: dict[int, np.ndarray] = {yr: np.full(n_pix, np.nan) for yr in years}

        # ---- Chunked neighbour query + grouped median ----------------------
        # For each chunk: query neighbours once, then per year compute medians
        # only for pixels that have a reliable DOY that year (skip the rest).
        # This avoids querying/computing neighbours for pixels that will never
        # produce a valid lookup entry.
        #
        # Within each (chunk, year) pass: pixels without a reliable DOY are
        # filtered out before building the CSR → the groupby operates on a
        # strictly smaller set, proportional to fraction of pixels with data.
        for chunk_start in range(0, n_pix, CHUNK_PIX):
            chunk_end  = min(chunk_start + CHUNK_PIX, n_pix)
            chunk_size = chunk_end - chunk_start
            chunk_xy   = gated_xy[chunk_start:chunk_end]

            # nbrs[j] = list of indices into gated_pids for pixel (chunk_start+j)
            nbrs = tree.query_ball_point(chunk_xy, r=p.radius_m, workers=-1)

            # Pre-build a fixed flat index for all chunk pixels (used below)
            local_counts_all = np.array([len(n) for n in nbrs], dtype=np.int64)
            total_all        = int(local_counts_all.sum())
            if total_all == 0:
                continue

            flat_nbrs_all  = np.empty(total_all, dtype=np.int32)
            global_src_all = np.repeat(
                np.arange(chunk_start, chunk_end, dtype=np.int32),
                local_counts_all.astype(np.int32),
            )
            pos = 0
            for j, nbr in enumerate(nbrs):
                flat_nbrs_all[pos:pos + len(nbr)] = nbr
                pos += len(nbr)
            del nbrs  # free the ragged list

            self_mask_all    = flat_nbrs_all == global_src_all
            # pixel_labels_all is constant for this chunk — compute once.
            pixel_labels_all = np.repeat(
                np.arange(chunk_size, dtype=np.int32),
                local_counts_all.astype(np.int32),
            )

            # Per year: restrict to pixels that have a reliable DOY this year.
            for yr in years:
                doy_arr  = doy_by_year[yr]
                has_own_chunk = ~np.isnan(doy_arr[chunk_start:chunk_end])
                if not has_own_chunk.any():
                    continue  # no pixels in this chunk have data this year

                # Local indices (within chunk) of pixels with a reliable DOY.
                active_local = np.where(has_own_chunk)[0].astype(np.int32)
                # Which edges belong to active pixels?
                edge_mask   = np.isin(pixel_labels_all, active_local)

                flat_nbrs_c    = flat_nbrs_all[edge_mask]
                global_src_c   = global_src_all[edge_mask]
                pixel_labels_c = pixel_labels_all[edge_mask]
                self_mask_c    = self_mask_all[edge_mask]

                if flat_nbrs_c.size == 0:
                    continue

                flat_doys              = doy_arr[flat_nbrs_c].copy()
                flat_doys[self_mask_c] = np.nan

                med_ser       = (
                    pd.Series(flat_doys, dtype=float)
                    .groupby(pixel_labels_c)
                    .median()
                )
                # med_ser indexed by local j; write results back at global positions
                for local_j, med_val in med_ser.items():
                    nbr_medians_by_year[yr][chunk_start + local_j] = med_val

        # ---- Collect records ----------------------------------------------
        records = []
        for yr in years:
            doy_arr     = doy_by_year[yr]
            nbr_medians = nbr_medians_by_year[yr]
            has_own     = ~np.isnan(doy_arr)
            has_nbrs    = ~np.isnan(nbr_medians)
            valid_mask  = has_own & has_nbrs
            yr_int      = int(yr)
            for i in np.flatnonzero(valid_mask):
                records.append({
                    "point_id":   gated_pids[i],
                    "year":       yr_int,
                    "nbr_median": float(nbr_medians[i]),
                })

        lookup = pd.DataFrame(records, columns=["point_id", "year", "nbr_median"])
        self._lookup = lookup
        return lookup

    def compute(
        self,
        per_year: pd.DataFrame,
        coords: pd.DataFrame,
        amp_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Compute relative greenup shift features per pixel.

        Calls ``build_lookup`` automatically if it has not been called yet (or
        if ``self._lookup`` is None).  Call ``build_lookup`` explicitly before
        sweeping ``min_years`` to avoid redundant spatial work.

        Parameters
        ----------
        per_year:
            Per-pixel per-year peak DOY table — must cover the full scene so
            that neighbourhood medians are representative.
        coords:
            DataFrame with columns ``point_id, lon, lat`` (full scene).
        amp_series:
            Mean annual NDVI amplitude indexed by ``point_id``.  Passed to
            ``build_lookup`` if the lookup has not been built yet.

        Returns
        -------
        DataFrame with columns:
            ``point_id, lon, lat, peak_doy_shift, peak_doy_shift_sd, n_shift_years``.

        ``peak_doy_shift`` is NaN for pixels with fewer than ``min_years``
        reliable offset estimates.
        """
        if self._lookup is None:
            self.build_lookup(per_year, coords, amp_series)

        p      = self.params
        lookup = self._lookup  # point_id, year, nbr_median

        # Merge own peak DOY with neighbourhood median on (point_id, year)
        reliable = per_year[per_year["reliable"]][["point_id", "year", "peak_doy"]]
        merged   = reliable.merge(lookup, on=["point_id", "year"], how="inner")
        merged["offset"] = merged["peak_doy"] - merged["nbr_median"]

        # Aggregate across years
        agg = (
            merged
            .groupby("point_id")["offset"]
            .agg(
                peak_doy_shift=("mean"),
                peak_doy_shift_sd=("std"),
                n_shift_years=("count"),
            )
            .reset_index()
        )
        # Mask pixels below min_years
        insufficient = agg["n_shift_years"] < p.min_years
        agg.loc[insufficient, ["peak_doy_shift", "peak_doy_shift_sd"]] = np.nan

        # All pixels in coords get a row (NaN if no valid years)
        all_pids = coords[["point_id"]].copy()
        result   = all_pids.merge(agg, on="point_id", how="left")
        result["n_shift_years"] = result["n_shift_years"].fillna(0).astype(int)

        coords_idx = coords.set_index("point_id")[["lon", "lat"]]
        result = result.merge(
            coords_idx.reset_index(), on="point_id", how="left"
        )
        return result[["point_id", "lon", "lat",
                        "peak_doy_shift", "peak_doy_shift_sd", "n_shift_years"]]
