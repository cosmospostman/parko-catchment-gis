"""signals/wet_dry_amp.py — Wet/dry seasonal NDVI amplitude signal.

Primary metric: rec_p = mean annual (p90 − p10) NDVI across years.
Window-free — no fixed wet/dry month windows needed for feature computation.
Higher rec_p → deeper seasonal swing → Parkinsonia signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import polars as pl

from signals import QualityParams
from signals._shared import load_and_filter


class RecPSignal:
    """Wet/dry seasonal NDVI amplitude signal."""

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)

    def __init__(self, params: RecPSignal.Params | None = None) -> None:
        self.params = params or RecPSignal.Params()

    def compute(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
    ) -> pd.DataFrame:
        """Compute rec_p per pixel.

        Returns
        -------
        DataFrame with columns ``[point_id, lon, lat, rec_p, rec_p_std, n_years]``.
        """
        p = self.params
        df = load_and_filter(
            pixel_df, p.quality.scl_purity_min,
            load_cols=["point_id", "lon", "lat", "date", "scl_purity", "B08", "B04"],
        )

        coords = (
            df.select(["point_id", "lon", "lat"])
            .unique("point_id")
            .to_pandas()
        )

        df = df.with_columns([
            ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))).alias("ndvi")
        ])

        # Annual (p90 − p10) amplitude per (pixel, year), filtering sparse years
        annual = (
            df.group_by(["point_id", "year"])
            .agg([
                pl.col("ndvi").quantile(0.90, interpolation="linear").alias("ndvi_p90"),
                pl.col("ndvi").quantile(0.10, interpolation="linear").alias("ndvi_p10"),
                pl.col("ndvi").count().alias("_n_obs"),
            ])
            .filter(pl.col("_n_obs") >= p.quality.min_obs_per_year)
            .with_columns([
                (pl.col("ndvi_p90") - pl.col("ndvi_p10")).alias("rec_p")
            ])
        )

        stats = (
            annual.group_by("point_id")
            .agg([
                pl.col("rec_p").mean(),
                pl.col("rec_p").std().alias("rec_p_std"),
                pl.col("rec_p").count().alias("n_years"),
            ])
            .to_pandas()
        )

        stats = stats.merge(coords, on="point_id", how="left")

        return stats[["point_id", "lon", "lat", "rec_p", "rec_p_std", "n_years"]]

        self,
        pixel_df: pd.DataFrame,
        loc: object,
        out_dir: Path | None = None,
    ) -> dict:
        """Compute signal and write standard diagnostic figures.

        Returns
        -------
        dict with keys: ``signal``, ``site``, ``n_pixels``,
        ``presence_median``, ``absence_median``, ``separability``, ``figures``.
        """
        from signals.diagnostics import plot_signal_map, plot_distributions, separability_score, _resolve_classes

        stats = self.compute(pixel_df, loc)

        if out_dir is None:
            from pathlib import Path as _Path
            _root = _Path(__file__).resolve().parent.parent
            out_dir = _root / "outputs" / f"{loc.id}-rec-p"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        presence_ids, absence_ids = _resolve_classes(stats, loc)
        sep = separability_score(stats, "rec_p", presence_ids, absence_ids)

        figures = []
        fig = plot_signal_map(
            stats, "rec_p", loc,
            title=f"{loc.name} — NDVI seasonal amplitude (rec_p)\nhigher = deeper wet/dry swing",
            out_path=out_dir / "map.png",
            colormap="YlOrRd",
        )
        if fig is not None:
            figures.append(out_dir / "map.png")

        fig = plot_distributions(
            stats, "rec_p", loc,
            presence_ids=presence_ids, absence_ids=absence_ids,
            out_path=out_dir / "distributions.png",
        )
        if fig is not None:
            figures.append(out_dir / "distributions.png")

        return {
            "signal": "rec_p",
            "site": loc.id,
            "n_pixels": len(stats),
            "presence_median": stats.loc[stats["point_id"].isin(presence_ids), "rec_p"].median() if presence_ids is not None else None,
            "absence_median": stats.loc[stats["point_id"].isin(absence_ids), "rec_p"].median() if absence_ids is not None else None,
            "separability": sep,
            "figures": figures,
        }
