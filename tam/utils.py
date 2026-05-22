from __future__ import annotations

from pathlib import Path

import polars as pl


def label_pixels(features_df: pl.DataFrame, train_loc) -> pl.DataFrame:
    """Assign is_presence from labeled regions or a Location's sub_bboxes.

    Accepts either:
    - A Location object (uses sub_bboxes with role "presence"/"absence")
    - A list of TrainingRegion objects (uses bbox + label "presence"/"absence")

    Returns features_df with an is_presence column (True / False / None).
    Pixels outside any labelled bbox get None — scored but not trained on.
    """
    # Build (bbox, value) pairs from whichever input type we received.
    if isinstance(train_loc, list):
        regions_vals = [
            (region.bbox, True if region.label == "presence" else False)
            for region in train_loc
            if region.label in ("presence", "absence")
        ]
    else:
        regions_vals = [
            (sub.bbox, True if sub.role == "presence" else False)
            for sub in train_loc.sub_bboxes.values()
            if sub.role in ("presence", "absence")
        ]

    # Start with null and overwrite per region using vectorised Polars expressions.
    result = pl.lit(None, dtype=pl.Boolean)
    for bbox, val in regions_vals:
        lon_min, lat_min, lon_max, lat_max = bbox
        in_bbox = (
            pl.col("lon").is_between(lon_min, lon_max) &
            pl.col("lat").is_between(lat_min, lat_max)
        )
        result = pl.when(in_bbox).then(pl.lit(val)).otherwise(result)

    return features_df.with_columns(result.alias("is_presence"))


def summarise(
    scored_df: pl.DataFrame,
    loc,
    *,
    show_scene_percentiles: bool = True,
    prob_col: str = "prob_tam",
) -> None:
    """Print per-class probability statistics."""
    print(f"\n{'='*60}")
    print(f"Site: {loc.name}  ({len(scored_df):,} pixels)")
    print(f"{'='*60}")

    labelled = scored_df.filter(pl.col("is_presence").is_not_null())
    if len(labelled) > 0:
        print("\nProbability by class (mean / median / std):")
        for val, label in [(True, "Presence"), (False, "Absence")]:
            sub = labelled.filter(pl.col("is_presence") == val)[prob_col].drop_nulls()
            if len(sub) > 0:
                print(f"  {label:10s}  mean={sub.mean():.3f}  median={sub.median():.3f}  std={sub.std():.3f}")

    if show_scene_percentiles:
        all_scored = scored_df[prob_col].drop_nulls()
        print(f"\nFull scene  ({len(all_scored):,} scored pixels):")
        print(f"  mean={all_scored.mean():.3f}  median={all_scored.median():.3f}  std={all_scored.std():.3f}")
        for pct in (75, 90, 95):
            print(f"  p{pct}={all_scored.quantile(pct/100):.3f}")


def save_pixel_ranking(
    scored_df: pl.DataFrame,
    out_path: Path,
    features: list[str],
) -> None:
    """Write the scored pixel table sorted by rank to a CSV file."""
    prob_cols = [c for c in scored_df.columns if c.startswith("prob_")]
    extra = [f for f in features if f not in prob_cols]
    cols = ["point_id", "lon", "lat", "is_presence"] + prob_cols + ["rank"] + extra
    out = scored_df.select([c for c in cols if c in scored_df.columns]).sort("rank")
    out.write_csv(out_path, float_precision=4)
    print(f"Saved: {out_path}")
