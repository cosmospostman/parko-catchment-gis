"""signals/red_edge.py — Red-edge ratio annual floor signal.

Metric: B07/B05 ratio — proxy for active chlorophyll retention.
Primary metric: re_p10 = mean annual 10th-percentile B07/B05 across years.
Higher re_p10 → more retained chlorophyll in dry season → Parkinsonia signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import polars as pl

from signals import QualityParams
from signals._shared import load_and_filter, annual_percentile


class RedEdgeSignal:
    """Red-edge ratio annual 10th-percentile floor signal."""

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)
        floor_percentile: float = 0.10

    def __init__(self, params: RedEdgeSignal.Params | None = None) -> None:
        self.params = params or RedEdgeSignal.Params()

    def compute(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
    ) -> pd.DataFrame:
        """Compute re_p10 per pixel.

        Returns
        -------
        DataFrame with columns ``[point_id, lon, lat, re_p10, n_years]``.
        """
        p = self.params
        df = load_and_filter(
            pixel_df, p.quality.scl_purity_min,
            load_cols=["point_id", "lon", "lat", "date", "scl_purity", "B07", "B05"],
        )

        coords = (
            df.select(["point_id", "lon", "lat"])
            .unique("point_id")
            .to_pandas()
        )

        df = df.with_columns([
            (pl.col("B07") / pl.col("B05")).alias("re_ratio")
        ])

        fp = p.floor_percentile
        fp_int = int(round(fp * 100))
        stats = annual_percentile(df, "re_ratio", fp, p.quality.min_obs_per_year)
        stats = stats.rename(columns={f"re_ratio_p{fp_int}": "re_p10"})

        stats = stats.merge(coords, on="point_id", how="left")

        return stats[["point_id", "lon", "lat", "re_p10", "n_years"]]

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
            out_dir = _root / "outputs" / f"{loc.id}-red-edge"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        presence_ids, absence_ids = _resolve_classes(stats, loc)
        sep = separability_score(stats, "re_p10", presence_ids, absence_ids)

        figures = []
        fig = plot_signal_map(
            stats, "re_p10", loc,
            title=f"{loc.name} — Red-edge ratio annual p10 (re_p10)\nhigher = retained chlorophyll floor",
            out_path=out_dir / "map.png",
            colormap="YlGn",
        )
        if fig is not None:
            figures.append(out_dir / "map.png")

        fig = plot_distributions(
            stats, "re_p10", loc,
            presence_ids=presence_ids, absence_ids=absence_ids,
            out_path=out_dir / "distributions.png",
        )
        if fig is not None:
            figures.append(out_dir / "distributions.png")

        return {
            "signal": "re_p10",
            "site": loc.id,
            "n_pixels": len(stats),
            "presence_median": stats.loc[stats["point_id"].isin(presence_ids), "re_p10"].median() if presence_ids is not None else None,
            "absence_median": stats.loc[stats["point_id"].isin(absence_ids), "re_p10"].median() if absence_ids is not None else None,
            "separability": sep,
            "figures": figures,
        }
