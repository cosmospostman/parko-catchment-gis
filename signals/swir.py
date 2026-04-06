"""signals/swir.py — SWIR moisture index annual floor signal.

Metric: (B08 − B11) / (B08 + B11) — SWIR moisture index.
Primary metric: swir_p10 = mean annual 10th-percentile across years.
Higher swir_p10 → sustained canopy moisture in dry season → Parkinsonia signature.

Note: swir_p10 is correlated with re_p10 (Pearson r ≈ 0.73 at Longreach).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from signals import QualityParams
from signals._shared import load_and_filter, annual_percentile


class SwirSignal:
    """SWIR moisture index annual 10th-percentile floor signal."""

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)
        riparian_percentile: float = 0.90

    def __init__(self, params: SwirSignal.Params | None = None) -> None:
        self.params = params or SwirSignal.Params()

    def compute(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
    ) -> pd.DataFrame:
        """Compute swir_p10 per pixel.

        Returns
        -------
        DataFrame with columns ``[point_id, lon, lat, swir_p10, n_years]``.
        """
        p = self.params
        df = load_and_filter(pixel_df, p.quality.scl_purity_min)
        df["swir_mi"] = (df["B08"] - df["B11"]) / (df["B08"] + df["B11"])

        stats = annual_percentile(df, "swir_mi", 0.10, p.quality.min_obs_per_year)
        stats = stats.rename(columns={"swir_mi_p10": "swir_p10"})

        coords = df[["point_id", "lon", "lat"]].drop_duplicates("point_id")
        stats = stats.merge(coords, on="point_id", how="left")

        return stats[["point_id", "lon", "lat", "swir_p10", "n_years"]]

    def diagnose(
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
            out_dir = _root / "outputs" / f"{loc.id}-swir"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        presence_ids, absence_ids = _resolve_classes(stats, loc)
        sep = separability_score(stats, "swir_p10", presence_ids, absence_ids)

        figures = []
        fig = plot_signal_map(
            stats, "swir_p10", loc,
            title=f"{loc.name} — SWIR moisture index annual p10 (swir_p10)\nhigher = sustained canopy moisture",
            out_path=out_dir / "map.png",
            colormap="Blues",
        )
        if fig is not None:
            figures.append(out_dir / "map.png")

        fig = plot_distributions(
            stats, "swir_p10", loc,
            presence_ids=presence_ids, absence_ids=absence_ids,
            out_path=out_dir / "distributions.png",
        )
        if fig is not None:
            figures.append(out_dir / "distributions.png")

        return {
            "signal": "swir_p10",
            "site": loc.id,
            "n_pixels": len(stats),
            "presence_median": stats.loc[stats["point_id"].isin(presence_ids), "swir_p10"].median() if presence_ids is not None else None,
            "absence_median": stats.loc[stats["point_id"].isin(absence_ids), "swir_p10"].median() if absence_ids is not None else None,
            "separability": sep,
            "figures": figures,
        }
