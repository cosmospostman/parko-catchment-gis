"""signals/nir_cv.py — Dry-season NIR inter-annual coefficient of variation.

Lower nir_cv → more year-to-year stability → consistent evergreen cover.
Parkinsonia: low nir_cv. Grasses: high nir_cv.

Primary metric: nir_cv = std(annual dry-season median B08) / mean(annual dry-season median B08)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl

from signals import QualityParams
from signals._shared import load_and_filter, dry_season_cv


class NirCvSignal:
    """Dry-season NIR inter-annual coefficient of variation signal."""

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)

    def __init__(self, params: NirCvSignal.Params | None = None) -> None:
        self.params = params or NirCvSignal.Params()

    def compute(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
    ) -> pd.DataFrame:
        """Compute nir_cv per pixel.

        Parameters
        ----------
        pixel_df:
            Raw observation parquet loaded for this location.
        loc:
            ``utils.location.Location`` — ``dry_months`` sourced from here.

        Returns
        -------
        DataFrame with columns ``[point_id, lon, lat, nir_mean, nir_std, nir_cv, n_years]``.
        """
        p = self.params
        df = load_and_filter(
            pixel_df, p.quality.scl_purity_min,
            load_cols=["point_id", "lon", "lat", "date", "scl_purity", "B08"],
        )

        coords = (
            df.select(["point_id", "lon", "lat"])
            .unique("point_id")
            .to_pandas()
        )

        stats = dry_season_cv(df, "B08", loc.dry_months, p.quality.min_obs_dry)
        stats = stats.rename(columns={
            "B08_mean": "nir_mean",
            "B08_std": "nir_std",
            "B08_cv": "nir_cv",
        })

        stats = stats.merge(coords, on="point_id", how="left")

        col_order = ["point_id", "lon", "lat", "nir_mean", "nir_std", "nir_cv", "n_years"]
        return stats[col_order]

        self,
        pixel_df: pd.DataFrame,
        loc: object,
        out_dir: Path | None = None,
    ) -> dict:
        """Compute signal and write standard diagnostic figures.

        Parameters
        ----------
        pixel_df:
            Raw observation parquet loaded for this location.
        loc:
            ``utils.location.Location``.
        out_dir:
            If given, figures are written here. Defaults to
            ``outputs/<loc.id>-nir-cv/``.

        Returns
        -------
        dict with keys: ``signal``, ``site``, ``n_pixels``,
        ``presence_median``, ``absence_median``, ``separability``,
        ``figures``.
        """
        from signals.diagnostics import (
            plot_signal_map,
            plot_distributions,
            separability_score,
            _resolve_classes,
        )

        stats = self.compute(pixel_df, loc)

        if out_dir is None:
            from pathlib import Path as _Path
            _root = _Path(__file__).resolve().parent.parent
            out_dir = _root / "outputs" / f"{loc.id}-nir-cv"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        presence_ids, absence_ids = _resolve_classes(stats, loc)

        sep = separability_score(stats, "nir_cv", presence_ids, absence_ids)

        figures = []
        fig = plot_signal_map(
            stats, "nir_cv", loc,
            title=f"{loc.name} — NIR CV (dry-season, inter-annual)\nlower = more stable = persistent canopy",
            out_path=out_dir / "map.png",
            colormap="RdYlGn_r",
        )
        if fig is not None:
            figures.append(out_dir / "map.png")

        fig = plot_distributions(
            stats, "nir_cv", loc,
            presence_ids=presence_ids,
            absence_ids=absence_ids,
            out_path=out_dir / "distributions.png",
        )
        if fig is not None:
            figures.append(out_dir / "distributions.png")

        result = {
            "signal": "nir_cv",
            "site": loc.id,
            "n_pixels": len(stats),
            "presence_median": stats.loc[stats["point_id"].isin(presence_ids), "nir_cv"].median() if presence_ids is not None else None,
            "absence_median": stats.loc[stats["point_id"].isin(absence_ids), "nir_cv"].median() if absence_ids is not None else None,
            "separability": sep,
            "figures": figures,
        }
        return result
