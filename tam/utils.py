from __future__ import annotations

from pathlib import Path

import pandas as pd


def label_pixels(features_df: pd.DataFrame, train_loc) -> pd.DataFrame:
    """Assign is_presence from labeled regions or a Location's sub_bboxes.

    Accepts either:
    - A Location object (uses sub_bboxes with role "presence"/"absence")
    - A list of TrainingRegion objects (uses bbox + label "presence"/"absence")

    Returns a copy of features_df with an is_presence column (True / False / NaN).
    Pixels outside any labelled bbox get NaN — scored but not trained on.
    """
    df = features_df.copy()
    df["is_presence"] = pd.NA

    if isinstance(train_loc, list):
        for region in train_loc:
            lon_min, lat_min, lon_max, lat_max = region.bbox
            mask = (
                df["lon"].between(lon_min, lon_max) &
                df["lat"].between(lat_min, lat_max)
            )
            if region.label == "presence":
                df.loc[mask, "is_presence"] = True
            elif region.label == "absence":
                df.loc[mask, "is_presence"] = False
    else:
        for sub in train_loc.sub_bboxes.values():
            lon_min, lat_min, lon_max, lat_max = sub.bbox
            mask = (
                df["lon"].between(lon_min, lon_max) &
                df["lat"].between(lat_min, lat_max)
            )
            if sub.role == "presence":
                df.loc[mask, "is_presence"] = True
            elif sub.role == "absence":
                df.loc[mask, "is_presence"] = False

    return df


def summarise(
    scored_df: pd.DataFrame,
    loc,
    *,
    show_scene_percentiles: bool = True,
    prob_col: str = "prob_tam",
) -> None:
    """Print per-class probability statistics."""
    print(f"\n{'='*60}")
    print(f"Site: {loc.name}  ({len(scored_df):,} pixels)")
    print(f"{'='*60}")

    labelled = scored_df[scored_df["is_presence"].notna()]
    if not labelled.empty:
        print("\nProbability by class (mean / median / std):")
        for val, label in [(True, "Presence"), (False, "Absence")]:
            sub = labelled[labelled["is_presence"] == val][prob_col]
            if not sub.empty:
                print(f"  {label:10s}  mean={sub.mean():.3f}  median={sub.median():.3f}  std={sub.std():.3f}")

    if show_scene_percentiles:
        all_scored = scored_df[prob_col].dropna()
        print(f"\nFull scene  ({len(all_scored):,} scored pixels):")
        print(f"  mean={all_scored.mean():.3f}  median={all_scored.median():.3f}  std={all_scored.std():.3f}")
        for pct in (75, 90, 95):
            print(f"  p{pct}={all_scored.quantile(pct/100):.3f}")


def save_pixel_ranking(
    scored_df: pd.DataFrame,
    out_path: Path,
    features: list[str],
) -> None:
    """Write the scored pixel table sorted by rank to a CSV file."""
    prob_cols = [c for c in scored_df.columns if c.startswith("prob_")]
    cols = ["point_id", "lon", "lat", "is_presence"] + prob_cols + ["rank"] + features
    scored_df[[c for c in cols if c in scored_df.columns]].sort_values("rank").to_csv(
        out_path, index=False, float_format="%.4f"
    )
    print(f"Saved: {out_path}")
