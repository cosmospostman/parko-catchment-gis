"""
Cluster Quaids 2025 high-score (>0.9) pixels by dry-season spectral features
and write a score-CSV where prob_tam encodes the cluster as a discrete value
spread across [0, 1] so each cluster renders as a distinct colour in the UI.

Non-FP pixels (score ≤ 0.9) are omitted — only the cluster map is written.

Output: outputs/scores/quaids/spectral_heatmap.csv
Columns: point_id, lon, lat, prob_tam  (+ ndvi_dry, ndwi_dry, mavi_dry, vh_db_dry)

Usage:
    python utils/diag_quaids_spectral_heatmap.py [--n-clusters N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.constants import add_spectral_indices, ensure_float32_bands

DRY_DOY_MIN = 121   # May 1
DRY_DOY_MAX = 304   # Oct 31

PIXEL_PARQUETS = [
    ROOT / "data/pixels/quaids/2025/55KBB.parquet",
    ROOT / "data/pixels/quaids/2025/55KCB.parquet",
]
SCORE_CSV = ROOT / "outputs/scores/quaids/tam-v10-0.875.csv"
OUT_CSV   = ROOT / "outputs/scores/quaids/spectral_heatmap.csv"

# Discrete prob_tam values assigned to each cluster (spread across colourmap).
# k clusters → evenly spaced in (0, 1] so they render as distinct colours.
def cluster_prob_values(k: int) -> list[float]:
    return [round((i + 1) / k, 4) for i in range(k)]


def load_pixels() -> pl.DataFrame:
    return add_spectral_indices(ensure_float32_bands(pl.concat(
        [pl.read_parquet(p) for p in PIXEL_PARQUETS],
        how="diagonal_relaxed",
    )))


def dry_season_means(df: pl.DataFrame) -> pl.DataFrame:
    if "doy" not in df.columns:
        df = df.with_columns(pl.col("date").dt.ordinal_day().alias("doy"))

    dry_s2 = (
        df.filter(
            (pl.col("source") == "S2")
            & pl.col("doy").is_between(DRY_DOY_MIN, DRY_DOY_MAX)
            & pl.col("scl_purity").is_not_null()
            & (pl.col("scl_purity") >= 0.5)
        )
        .group_by(["point_id", "lon", "lat"])
        .agg(
            pl.col("NDVI").mean().alias("ndvi_dry"),
            pl.col("NDWI").mean().alias("ndwi_dry"),
            pl.col("MAVI").mean().alias("mavi_dry"),
        )
    )

    dry_s1 = (
        df.filter(
            (pl.col("source") == "S1")
            & pl.col("doy").is_between(DRY_DOY_MIN, DRY_DOY_MAX)
            & pl.col("vh").is_not_null()
        )
        .with_columns((10.0 * pl.col("vh").log(base=10.0)).alias("vh_db"))
        .filter(pl.col("vh_db").is_finite())
        .group_by("point_id")
        .agg(pl.col("vh_db").mean().alias("vh_db_dry"))
    )

    return dry_s2.join(dry_s1, on="point_id", how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=5)
    args = parser.parse_args()
    k = args.n_clusters

    print("Loading pixel parquets ...")
    df = load_pixels()
    print(f"  {len(df):,} rows  ({df['point_id'].n_unique():,} pixels)")

    print("Computing dry-season means ...")
    summary = dry_season_means(df)

    print("Loading scores ...")
    scores = (
        pl.read_csv(SCORE_CSV)
        .filter(pl.col("prob_tam").is_not_null())
        .with_columns(pl.col("prob_tam").cast(pl.Float64))
        .filter(pl.col("prob_tam").is_not_nan())
        .select(["point_id", "prob_tam"])
    )

    scores_hi = scores.filter(pl.col("prob_tam") > 0.9)
    hi = summary.join(scores_hi.select("point_id"), on="point_id", how="inner").drop_nulls(
        subset=["ndvi_dry", "mavi_dry", "vh_db_dry"]
    )
    print(f"  {len(hi):,} high-score pixels with full spectral data")

    # Cluster on NDVI, MAVI, VH(dB) — the most discriminative dry-season features
    feats = hi.select(["ndvi_dry", "mavi_dry", "vh_db_dry"]).to_numpy()
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    print(f"Clustering into {k} groups (k-means) ...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(feats_scaled)

    # Sort cluster labels by mean NDVI so colour order is meaningful
    # (low NDVI = bare/grass → cluster 0 = dark; high NDVI = woodland → cluster k-1 = bright)
    cluster_ndvi_means = [feats[labels == c, 0].mean() for c in range(k)]
    rank = np.argsort(cluster_ndvi_means)           # rank[i] = original cluster with i-th lowest NDVI
    remap = np.empty(k, dtype=int)
    for new_label, old_label in enumerate(rank):
        remap[old_label] = new_label
    labels_ranked = remap[labels]

    probs = cluster_prob_values(k)
    prob_vals = np.array([probs[l] for l in labels_ranked])

    out = hi.with_columns(
        pl.Series("prob_tam", prob_vals),
    ).select(["point_id", "lon", "lat", "ndvi_dry", "ndwi_dry", "mavi_dry", "vh_db_dry", "prob_tam"])

    out = out.with_columns([
        pl.col("lon").round(6),
        pl.col("lat").round(6),
        pl.col("ndvi_dry").round(4),
        pl.col("ndwi_dry").round(4),
        pl.col("mavi_dry").round(4),
        pl.col("vh_db_dry").round(3),
    ]).sort(["lat", "lon"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUT_CSV)
    print(f"\nWritten {len(out):,} rows → {OUT_CSV}")

    print(f"\nCluster summary (prob_tam → colour):")
    centres = scaler.inverse_transform(km.cluster_centers_)
    for new_label in range(k):
        old_label = rank[new_label]
        ndvi_c, mavi_c, vh_c = centres[old_label]
        n = (labels_ranked == new_label).sum()
        print(f"  cluster {new_label+1}  prob={probs[new_label]}  n={n:6,}  "
              f"NDVI={ndvi_c:.3f}  MAVI={mavi_c:.3f}  VH={vh_c:.1f} dB")


if __name__ == "__main__":
    main()
