"""utils/site_similarity.py — Site-level similarity and separability metrics.

Loads a global_features_cache.parquet produced by a training run and computes:
  1. Site-to-site presence similarity (cosine distance matrix)
  2. Within-site presence/absence separability (Bhattacharyya distance)

Outputs:
  site_similarity.csv        — long-form pairwise cosine distances
  site_separability.csv      — per-site separability scores
  site_similarity.png        — clustermap heatmap of distance matrix
  site_separability.png      — bar chart of separability scores

Usage:
    python utils/site_similarity.py \
        --cache outputs/sweep_zscore_etna_landsend/lr5e-05_dm64/global_features_cache.parquet \
        --out outputs/sweep_zscore_etna_landsend/site_similarity
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _point_site_class(pid: str) -> tuple[str, str]:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return (m.group(1), m.group(2)) if m else (pid, "unknown")


def _drop_all_nan_features(df: pd.DataFrame) -> pd.DataFrame:
    all_nan = df.columns[df.isna().all()]
    if len(all_nan):
        logger.warning("Dropping all-NaN features (likely S1-only run): %s", list(all_nan))
    return df.drop(columns=all_nan)


def _bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two 1-D samples, scalar."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    var_pool = (var_a + var_b) / 2.0
    if var_pool < 1e-12:
        return 0.0
    term1 = 0.25 * ((mu_a - mu_b) ** 2) / var_pool
    term2 = 0.5 * np.log(var_pool / np.sqrt(max(var_a * var_b, 1e-24)))
    return term1 + term2


def compute_separability(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Per-site mean Bhattacharyya distance between presence and absence pixels."""
    rows = []
    sites = feat_df["site"].unique()
    for site in sorted(sites):
        pres = feat_df[(feat_df["site"] == site) & (feat_df["cls"] == "presence")]
        abse = feat_df[(feat_df["site"] == site) & (feat_df["cls"] == "absence")]
        n_pres = len(pres)
        n_abse = len(abse)
        if n_pres == 0 or n_abse == 0:
            rows.append({"site": site, "separability": float("nan"),
                         "n_presence": n_pres, "n_absence": n_abse})
            continue
        feature_cols = [c for c in feat_df.columns if c not in ("site", "cls")]
        dists = []
        for col in feature_cols:
            a = pres[col].dropna().values
            b = abse[col].dropna().values
            d = _bhattacharyya(a, b)
            if np.isfinite(d):
                dists.append(d)
        sep = float(np.mean(dists)) if dists else float("nan")
        rows.append({"site": site, "separability": sep,
                     "n_presence": n_pres, "n_absence": n_abse})
    return pd.DataFrame(rows).sort_values("separability", ascending=False, na_position="last")


def compute_presence_similarity(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine distance between site presence-median vectors."""
    from scipy.spatial.distance import cosine

    feature_cols = [c for c in feat_df.columns if c not in ("site", "cls")]
    pres = feat_df[feat_df["cls"] == "presence"]
    sites_with_pres = sorted(pres["site"].unique())

    # Median feature vector per site
    medians = (
        pres.groupby("site")[feature_cols]
        .median()
        .reindex(sites_with_pres)
    )

    # Z-score across sites so features are on equal footing
    mu = medians.mean()
    sd = medians.std().replace(0, 1)
    normed = (medians - mu) / sd

    rows = []
    for i, sa in enumerate(sites_with_pres):
        for j, sb in enumerate(sites_with_pres):
            if i >= j:
                continue
            va = normed.loc[sa].values
            vb = normed.loc[sb].values
            finite = np.isfinite(va) & np.isfinite(vb)
            if finite.sum() < 2:
                dist = float("nan")
            else:
                dist = cosine(va[finite], vb[finite])
            rows.append({"site_a": sa, "site_b": sb, "cosine_distance": dist})

    return pd.DataFrame(rows), medians, sites_with_pres


def _plot_heatmap(sites: list[str], dist_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    if len(sites) < 2:
        logger.warning("Too few sites for heatmap — skipping")
        return

    # Build symmetric matrix
    mat = pd.DataFrame(np.zeros((len(sites), len(sites))), index=sites, columns=sites)
    for _, row in dist_df.iterrows():
        v = row["cosine_distance"] if np.isfinite(row["cosine_distance"]) else 0.0
        mat.loc[row["site_a"], row["site_b"]] = v
        mat.loc[row["site_b"], row["site_a"]] = v

    # Hierarchical clustering to reorder rows/cols
    condensed = squareform(mat.values, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    ordered = mat.iloc[order, :].iloc[:, order]

    n = len(sites)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))
    im = ax.imshow(ordered.values, cmap="YlOrRd", vmin=0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ordered.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ordered.index, fontsize=9)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{ordered.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if ordered.values[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Cosine distance")
    ax.set_title("Site presence similarity (cosine distance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Heatmap: %s", out_path)


def _plot_separability(sep_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = sep_df.dropna(subset=["separability"])
    if plot_df.empty:
        logger.warning("No separability scores to plot")
        return

    fig, ax = plt.subplots(figsize=(7, max(3, len(plot_df) * 0.4)))
    bars = ax.barh(plot_df["site"], plot_df["separability"], color="steelblue")
    ax.set_xlabel("Mean Bhattacharyya distance (presence vs absence)")
    ax.set_title("Within-site separability")
    ax.invert_yaxis()
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"p={int(row['n_presence'])} a={int(row['n_absence'])}",
                va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close("all")
    logger.info("Separability chart: %s", out_path)


def _print_summary(sep_df: pd.DataFrame, medians: pd.DataFrame) -> None:
    print("\n=== Within-site separability ===")
    print(sep_df.to_string(index=False, float_format="%.3f"))

    print("\n=== Site presence medians (standardised features) ===")
    mu = medians.mean()
    sd = medians.std().replace(0, 1)
    normed = ((medians - mu) / sd).round(2)
    print(normed.to_string())


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True,
                        help="Path to global_features_cache.parquet")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same dir as cache)")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Feature subset (default: all non-NaN)")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        logger.error("Cache not found: %s", cache_path)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else cache_path.parent / "site_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cache_path)
    df = _drop_all_nan_features(df)

    if args.features:
        missing = [f for f in args.features if f not in df.columns]
        if missing:
            logger.error("Requested features not in cache: %s", missing)
            sys.exit(1)
        df = df[args.features]

    # Attach site and class labels
    df = df.copy()
    df["site"], df["cls"] = zip(*[_point_site_class(pid) for pid in df.index])

    # --- Separability ---
    sep_df = compute_separability(df)
    sep_path = out_dir / "site_separability.csv"
    sep_df.to_csv(sep_path, index=False)
    logger.info("Separability CSV: %s", sep_path)

    # --- Similarity ---
    dist_df, medians, sites_with_pres = compute_presence_similarity(df)
    sim_path = out_dir / "site_similarity.csv"
    dist_df.to_csv(sim_path, index=False)
    logger.info("Similarity CSV: %s", sim_path)

    # --- Print ---
    _print_summary(sep_df, medians)

    # --- Plots ---
    _plot_heatmap(sites_with_pres, dist_df, out_dir / "site_similarity.png")
    _plot_separability(sep_df, out_dir / "site_separability.png")

    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
