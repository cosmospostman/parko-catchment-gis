"""signals/flowering.py — Flowering flash detection signal.

Algorithm: DOY-binned z-score anomaly + haze masking + contrast gating.

Primary metric: fi_p90_cg = mean annual 90th-percentile FI_by z-score,
restricted to dates where scene-level contrast (infestation z minus extension z)
was positive that year.

This signal has a DIFFERENT interface from the four tabular signals. It requires
both an infestation population and an extension population to be defined in
loc.sub_bboxes (role='presence' and role='absence' respectively), because the
contrast-gating step compares the two populations.

Primary index: FI_by = (B03 + B04) / (B02 + B08) — suppresses bare-soil false positives.
Higher fi_p90_cg → more isolated flowering anomalies → Parkinsonia signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from signals import QualityParams
from signals._shared import load_and_filter as _load_and_filter


# Five flowering indices — (column_name, description)
_INDICES = [
    ("FI_rg",   "FI_rg  = (B03+B04)/B08"),
    ("FI_r",    "FI_r   = B04/B08"),
    ("FI_by",   "FI_by  = (B03+B04)/(B02+B08)"),
    ("dNDVI",   "dNDVI  = −(B08−B04)/(B08+B04)"),
    ("FI_swir", "FI_swir = B11/B08"),
]
_PRIMARY = "FI_by"


class FloweringSignal:
    """Flowering flash detection signal.

    Uses within-pixel DOY-binned z-score anomalies with scene-level haze masking
    and inter-population contrast gating to isolate transient flowering events.

    Presence/absence class boundaries are sourced from loc.sub_bboxes.
    """

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)
        doy_bin_days: int = 14
        haze_b02_anom_max: float = 0.010
        peak_percentile: int = 75
        riparian_nir_percentile: int = 90
        min_pixel_obs: int = 10

    def __init__(self, params: FloweringSignal.Params | None = None) -> None:
        self.params = params or FloweringSignal.Params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_haze_mask(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Return acquisition dates that pass the scene-level B02 haze filter."""
        p = self.params
        doy_bin_days = p.doy_bin_days
        haze_max = p.haze_b02_anom_max

        scene = df.groupby("date")["B02"].mean().reset_index()
        scene["doy"] = scene["date"].dt.dayofyear
        scene["doy_bin"] = ((scene["doy"] - 1) // doy_bin_days) * doy_bin_days + 1
        b2_baseline = scene.groupby("doy_bin")["B02"].median().rename("B02_base")
        scene = scene.join(b2_baseline, on="doy_bin")
        scene["B02_anom"] = scene["B02"] - scene["B02_base"]

        clean_dates = scene.loc[scene["B02_anom"] <= haze_max, "date"]
        return pd.DatetimeIndex(clean_dates)

    def _add_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["FI_rg"]   = (df["B03"] + df["B04"]) / df["B08"]
        df["FI_r"]    = df["B04"] / df["B08"]
        df["FI_by"]   = (df["B03"] + df["B04"]) / (df["B02"] + df["B08"])
        df["dNDVI"]   = -(df["B08"] - df["B04"]) / (df["B08"] + df["B04"])
        df["FI_swir"] = df["B11"] / df["B08"]
        index_cols = [name for name, _ in _INDICES]
        df[index_cols] = df[index_cols].replace([np.inf, -np.inf], np.nan)
        return df

    def _compute_pixel_anomalies(self, pop_df: pd.DataFrame) -> pd.DataFrame:
        """Per-pixel DOY-binned z-score anomalies for a single population."""
        doy_bin_days = self.params.doy_bin_days
        pop_df = pop_df.copy()
        pop_df["doy"] = pop_df["date"].dt.dayofyear
        pop_df["doy_bin"] = ((pop_df["doy"] - 1) // doy_bin_days) * doy_bin_days + 1
        pop_df["year"] = pop_df["date"].dt.year

        index_cols = [name for name, _ in _INDICES]

        baseline = (
            pop_df.groupby(["point_id", "doy_bin"])[index_cols]
            .median()
            .reset_index()
            .rename(columns={c: f"{c}_base" for c in index_cols})
        )
        pixel_std = (
            pop_df.groupby("point_id")[index_cols]
            .std()
            .reset_index()
            .rename(columns={c: f"{c}_std" for c in index_cols})
        )

        pop_df = pop_df.merge(baseline, on=["point_id", "doy_bin"], how="left")
        pop_df = pop_df.merge(pixel_std, on="point_id", how="left")

        for name in index_cols:
            std_safe = pop_df[f"{name}_std"].where(pop_df[f"{name}_std"] > 1e-6, np.nan)
            pop_df[f"{name}_z"] = (pop_df[name] - pop_df[f"{name}_base"]) / std_safe

        return pop_df

    def _build_contrast_ts(
        self,
        inf_anomalies: pd.DataFrame,
        ext_anomalies: pd.DataFrame,
    ) -> pd.DataFrame:
        """Per-date contrast: mean infestation FI_by z minus mean extension z."""
        z_col = f"{_PRIMARY}_z"
        inf_scene = inf_anomalies.groupby("date")[z_col].mean().rename("inf_z")
        ext_scene = ext_anomalies.groupby("date")[z_col].mean().rename("ext_z")
        ct = pd.concat([inf_scene, ext_scene], axis=1).reset_index()
        ct["contrast"] = ct["inf_z"] - ct["ext_z"]
        ct["year"] = ct["date"].dt.year
        ct = ct.sort_values("date").reset_index(drop=True)
        return ct

    def _build_pixel_p90(
        self,
        inf_anomalies: pd.DataFrame,
        ext_anomalies: pd.DataFrame,
        ct: pd.DataFrame,
    ) -> pd.DataFrame:
        """Per-pixel annual p90 FI_by z-score, unrestricted and contrast-gated."""
        z_col = f"{_PRIMARY}_z"
        contrast_pos_dates = set(ct.loc[ct["contrast"] > 0, "date"])

        rows = []
        for pop_label, pop_df in [("infestation", inf_anomalies), ("extension", ext_anomalies)]:
            pop_df = pop_df.copy()
            pop_df["year"] = pop_df["date"].dt.year

            annual_p90 = (
                pop_df.groupby(["point_id", "year"])[z_col]
                .quantile(0.90)
                .reset_index()
                .rename(columns={z_col: "fi_p90_year"})
            )
            pixel_p90 = (
                annual_p90.groupby("point_id")["fi_p90_year"]
                .mean()
                .reset_index()
                .rename(columns={"fi_p90_year": "fi_p90"})
            )

            cg_df = pop_df[pop_df["date"].isin(contrast_pos_dates)]
            if len(cg_df) > 0:
                annual_cg = (
                    cg_df.groupby(["point_id", "year"])[z_col]
                    .quantile(0.90)
                    .reset_index()
                    .rename(columns={z_col: "fi_p90_cg_year"})
                )
                pixel_cg = (
                    annual_cg.groupby("point_id")["fi_p90_cg_year"]
                    .mean()
                    .reset_index()
                    .rename(columns={"fi_p90_cg_year": "fi_p90_cg"})
                )
                pixel_p90 = pixel_p90.merge(pixel_cg, on="point_id", how="left")
            else:
                pixel_p90["fi_p90_cg"] = np.nan

            pixel_p90["population"] = pop_label
            rows.append(pixel_p90)

        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
    ) -> pd.DataFrame:
        """Compute fi_p90_cg per pixel.

        Presence pixels are identified from ``loc.sub_bboxes`` (role='presence');
        absence pixels from role='absence'. Both populations are required for
        contrast gating.

        Parameters
        ----------
        pixel_df:
            Raw observation parquet loaded for this location.
        loc:
            ``utils.location.Location``.

        Returns
        -------
        DataFrame with columns ``[point_id, lon, lat, fi_p90, fi_p90_cg, population, n_obs]``.
        """
        p = self.params

        # SCL quality filter
        df = _load_and_filter(pixel_df, p.quality.scl_purity_min)

        # Min per-pixel obs filter
        obs_per_pixel = df.groupby("point_id").size()
        valid_pixels = obs_per_pixel[obs_per_pixel >= p.min_pixel_obs].index
        df = df[df["point_id"].isin(valid_pixels)].copy()

        # Haze filter
        clean_dates = self._build_haze_mask(df)
        df = df[df["date"].isin(clean_dates)].copy()

        # Compute indices
        df = self._add_indices(df)

        # Assign population labels from loc.sub_bboxes
        presence_bbox = None
        absence_bbox = None
        for sub in loc.sub_bboxes.values():
            if sub.role == "presence":
                presence_bbox = sub.bbox
            elif sub.role == "absence":
                absence_bbox = sub.bbox

        coords = df[["point_id", "lat", "lon"]].drop_duplicates("point_id")

        if presence_bbox is not None:
            lon_min, lat_min, lon_max, lat_max = presence_bbox
            inf_ids = coords.loc[
                coords["lon"].between(lon_min, lon_max) &
                coords["lat"].between(lat_min, lat_max),
                "point_id",
            ]
        else:
            # Fallback: no presence bbox defined — all pixels are "extension"
            inf_ids = pd.Index([])

        df["population"] = np.where(df["point_id"].isin(inf_ids), "infestation", "extension")

        # DOY anomalies per population
        inf_anomalies = self._compute_pixel_anomalies(df[df["population"] == "infestation"])
        ext_anomalies = self._compute_pixel_anomalies(df[df["population"] == "extension"])

        # Contrast time series + p90 metrics
        ct = self._build_contrast_ts(inf_anomalies, ext_anomalies)
        p90 = self._build_pixel_p90(inf_anomalies, ext_anomalies, ct)

        # Attach coordinates
        p90 = p90.merge(coords, on="point_id", how="left")

        # Attach observation count
        n_obs = df.groupby("point_id").size().reset_index(name="n_obs")
        p90 = p90.merge(n_obs, on="point_id", how="left")

        col_order = ["point_id", "lon", "lat", "fi_p90", "fi_p90_cg", "population", "n_obs"]
        available = [c for c in col_order if c in p90.columns]
        return p90[available]

    def diagnose(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
        out_dir: Path | None = None,
    ) -> dict:
        """Compute signal and write standard diagnostic figures.

        Figures written (if out_dir provided):
          - map.png           — fi_p90_cg spatial scatter on WMS
          - distributions.png — fi_p90_cg histogram split by presence/absence
          - anomaly_profile.png — mean DOY anomaly profile per population

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
        )

        stats = self.compute(pixel_df, loc)

        if out_dir is None:
            _root = Path(__file__).resolve().parent.parent
            out_dir = _root / "outputs" / f"{loc.id}-flowering"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        presence_ids = stats.loc[stats["population"] == "infestation", "point_id"]
        absence_ids = stats.loc[stats["population"] == "extension", "point_id"]

        sep = separability_score(stats, "fi_p90_cg", presence_ids, absence_ids)

        figures = []

        fig = plot_signal_map(
            stats, "fi_p90_cg", loc,
            title=f"{loc.name} — Flowering flash fi_p90_cg\nhigher = isolated flowering anomaly events",
            out_path=out_dir / "map.png",
            colormap="YlOrRd",
        )
        if fig is not None:
            figures.append(out_dir / "map.png")

        fig = plot_distributions(
            stats, "fi_p90_cg", loc,
            presence_ids=presence_ids,
            absence_ids=absence_ids,
            out_path=out_dir / "distributions.png",
        )
        if fig is not None:
            figures.append(out_dir / "distributions.png")

        # DOY anomaly profile figure
        try:
            fig = self._plot_anomaly_profile(pixel_df, loc, out_dir / "anomaly_profile.png")
            if fig is not None:
                figures.append(out_dir / "anomaly_profile.png")
        except Exception as exc:
            print(f"  WARNING: anomaly_profile figure failed ({exc})", flush=True)

        return {
            "signal": "fi_p90_cg",
            "site": loc.id,
            "n_pixels": len(stats),
            "presence_median": stats.loc[stats["population"] == "infestation", "fi_p90_cg"].median(),
            "absence_median": stats.loc[stats["population"] == "extension", "fi_p90_cg"].median(),
            "separability": sep,
            "figures": figures,
        }

    def _plot_anomaly_profile(
        self,
        pixel_df: pd.DataFrame,
        loc: object,
        out_path: Path | None = None,
    ) -> object | None:
        """Mean FI_by z-score per DOY bin, split by population."""
        import matplotlib.pyplot as plt

        p = self.params

        df = _load_and_filter(pixel_df, p.quality.scl_purity_min)
        obs_per_pixel = df.groupby("point_id").size()
        valid_pixels = obs_per_pixel[obs_per_pixel >= p.min_pixel_obs].index
        df = df[df["point_id"].isin(valid_pixels)].copy()
        clean_dates = self._build_haze_mask(df)
        df = df[df["date"].isin(clean_dates)].copy()
        df = self._add_indices(df)

        # Assign populations
        coords = df[["point_id", "lat", "lon"]].drop_duplicates("point_id")
        presence_bbox = next(
            (sub.bbox for sub in loc.sub_bboxes.values() if sub.role == "presence"), None
        )
        if presence_bbox is not None:
            lon_min, lat_min, lon_max, lat_max = presence_bbox
            inf_ids = coords.loc[
                coords["lon"].between(lon_min, lon_max) &
                coords["lat"].between(lat_min, lat_max),
                "point_id",
            ]
        else:
            inf_ids = pd.Index([])
        df["population"] = np.where(df["point_id"].isin(inf_ids), "infestation", "extension")

        inf_anom = self._compute_pixel_anomalies(df[df["population"] == "infestation"])
        ext_anom = self._compute_pixel_anomalies(df[df["population"] == "extension"])

        z_col = f"{_PRIMARY}_z"
        doy_bin_days = p.doy_bin_days

        def _profile(anom_df: pd.DataFrame) -> pd.DataFrame:
            anom_df = anom_df.copy()
            anom_df["doy"] = anom_df["date"].dt.dayofyear
            anom_df["doy_bin"] = ((anom_df["doy"] - 1) // doy_bin_days) * doy_bin_days + 1
            return anom_df.groupby("doy_bin")[z_col].mean()

        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            inf_prof = _profile(inf_anom)
            ext_prof = _profile(ext_anom)
            ax.plot(inf_prof.index, inf_prof.values, color="darkorange", linewidth=1.8, label="Infestation")
            ax.plot(ext_prof.index, ext_prof.values, color="steelblue", linewidth=1.8, label="Extension")
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xlabel("Day of year (bin start)", fontsize=9)
            ax.set_ylabel(f"Mean {z_col} (z-score)", fontsize=9)
            ax.set_title(f"{loc.name} — FI_by DOY anomaly profile by population", fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()

            if out_path is not None:
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            return fig
        except Exception:
            return None
