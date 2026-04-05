"""analysis/ala_flower_flash.py — ALA-guided yellow flower flash search.

Identifies tightly clustered, recent Parkinsonia aculeata occurrence records
from the ALA cache, then fetches the minimal Sentinel-2 imagery needed to
look for the flowering spectral signal at those locations.

Pipeline
--------
1. Load ALA occurrences (from cache or fresh fetch).
2. Filter to recent records (--recency-cutoff).
3. DBSCAN cluster in projected space (--eps-m, --min-samples).
4. Score clusters by density; take top N (--top-n).
5. Generate a regular grid of sample points within each cluster bbox.
6. STAC search + chip fetch (flowering bands only).
7. Extract observations → quality-score → waveform features.
8. Print ranked table to stdout; write CSV to cache dir.

Usage
-----
    python analysis/ala_flower_flash.py \\
        --cache-dir /path/to/cache \\
        --working-dir /path/to/working \\
        --top-n 3

All options have sensible defaults — the script can run with only
--cache-dir and --working-dir.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAND_ALIAS: dict[str, str] = {
    "B05": "rededge1",
    "B07": "rededge3",
    "B08": "nir",
    "B11": "swir16",
    "SCL": "scl",
    "AOT": "aot",
}

# Bands required by flowering_index + quality pipeline
FLASH_BANDS: list[str] = ["B05", "B07", "B08", "B11", "SCL", "AOT"]

STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

PROJECTED_CRS = "EPSG:7855"  # GDA2020 / MGA Zone 55 (metres)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1–4: cluster selection
# ---------------------------------------------------------------------------

def load_occurrences(cache_dir: Path, species: str) -> gpd.GeoDataFrame:
    """Load ALA occurrences from cache; fetch fresh if absent."""
    gpkg = cache_dir / "ala_occurrences.gpkg"
    if gpkg.exists():
        logger.info("Loading ALA occurrences from cache: %s", gpkg)
        return gpd.read_file(gpkg)

    logger.info("Cache miss — fetching ALA occurrences for '%s'", species)
    # Import here to avoid circular dependency; fetch_ala_occurrences is
    # a standalone script that also exposes fetch_all() as a library function.
    from analysis.fetch_ala_occurrences import fetch_all
    gdf = fetch_all(species)
    gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(gpkg, driver="GPKG")
    logger.info("Written: %s", gpkg)
    return gdf


def select_clusters(
    gdf: gpd.GeoDataFrame,
    recency_cutoff: str,
    eps_m: float,
    min_samples: int,
    top_n: int,
    bbox_buffer_deg: float,
) -> list[dict]:
    """Return top-N cluster dicts ordered by density (desc).

    Each dict has keys:
        cluster_id, n_points, centroid_lon, centroid_lat,
        bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max,
        bbox_area_ha, density, recency_score,
        search_bbox (expanded by bbox_buffer_deg)
    """
    # --- Recency filter ---------------------------------------------------
    if "eventDate" in gdf.columns:
        gdf = gdf.copy()
        gdf["_ed"] = gdf["eventDate"].str[:10]  # keep YYYY-MM-DD prefix
        gdf = gdf[gdf["_ed"] >= recency_cutoff]
        logger.info(
            "After recency filter (>= %s): %d records", recency_cutoff, len(gdf)
        )

    if len(gdf) == 0:
        logger.error("No records after recency filter — loosen --recency-cutoff")
        return []

    # --- Project to metres -----------------------------------------------
    proj = gdf.to_crs(PROJECTED_CRS)
    coords = np.column_stack([proj.geometry.x, proj.geometry.y])

    # --- DBSCAN -----------------------------------------------------------
    db = DBSCAN(eps=eps_m, min_samples=min_samples, algorithm="ball_tree",
                metric="euclidean")
    labels = db.fit_predict(coords)
    gdf = gdf.copy()
    gdf["_cluster"] = labels
    gdf["_x_m"] = coords[:, 0]
    gdf["_y_m"] = coords[:, 1]

    unique_labels = set(labels) - {-1}
    logger.info("DBSCAN found %d clusters (noise points: %d)",
                len(unique_labels), int(np.sum(labels == -1)))

    if not unique_labels:
        logger.error("No clusters found — try lowering --eps-m or --min-samples")
        return []

    # --- Score clusters --------------------------------------------------
    recency_split = "2020-01-01"
    clusters = []
    for cid in unique_labels:
        subset = gdf[gdf["_cluster"] == cid]
        lons = subset.geometry.x.values
        lats = subset.geometry.y.values

        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())

        # bbox area in hectares via Transformer to metres
        t = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
        x0, y0 = t.transform(lon_min, lat_min)
        x1, y1 = t.transform(lon_max, lat_max)
        width_m = max(abs(x1 - x0), 1.0)
        height_m = max(abs(y1 - y0), 1.0)
        bbox_area_ha = (width_m * height_m) / 10_000

        n = len(subset)
        density = n / bbox_area_ha

        if "_ed" in subset.columns:
            recency_score = float((subset["_ed"] >= recency_split).mean())
        else:
            recency_score = 0.0

        centroid_lon = float(lons.mean())
        centroid_lat = float(lats.mean())

        # Expand bbox for location uncertainty
        search_bbox = [
            lon_min - bbox_buffer_deg,
            lat_min - bbox_buffer_deg,
            lon_max + bbox_buffer_deg,
            lat_max + bbox_buffer_deg,
        ]

        clusters.append({
            "cluster_id": int(cid),
            "n_points": n,
            "centroid_lon": centroid_lon,
            "centroid_lat": centroid_lat,
            "bbox_lon_min": lon_min,
            "bbox_lat_min": lat_min,
            "bbox_lon_max": lon_max,
            "bbox_lat_max": lat_max,
            "bbox_area_ha": bbox_area_ha,
            "density": density,
            "recency_score": recency_score,
            "search_bbox": search_bbox,
        })

    clusters.sort(key=lambda c: c["density"], reverse=True)
    top = clusters[:top_n]
    logger.info("Top %d clusters by density:", len(top))
    for c in top:
        logger.info(
            "  cluster=%d  n=%d  density=%.2f pts/ha  bbox_area=%.1f ha  "
            "centroid=(%.4f, %.4f)  recency=%.2f",
            c["cluster_id"], c["n_points"], c["density"], c["bbox_area_ha"],
            c["centroid_lon"], c["centroid_lat"], c["recency_score"],
        )
    return top


# ---------------------------------------------------------------------------
# Step 5: generate sample points within a cluster bbox
# ---------------------------------------------------------------------------

def make_grid_points(
    cluster: dict,
    spacing_m: float = 100.0,
) -> list[tuple[str, float, float]]:
    """Generate a regular grid of sample points within the cluster search bbox.

    Points are spaced spacing_m metres apart in the projected CRS, then
    back-projected to EPSG:4326. The 5×5 chip window covers ±2 pixels (~±20 m)
    around each point, so a 100 m grid provides ~5× coverage of any true peak.

    Returns list of (point_id, lon, lat).
    """
    lon_min, lat_min, lon_max, lat_max = cluster["search_bbox"]
    cid = cluster["cluster_id"]

    to_proj = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    to_geo = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)

    x0, y0 = to_proj.transform(lon_min, lat_min)
    x1, y1 = to_proj.transform(lon_max, lat_max)

    xs = np.arange(x0, x1, spacing_m)
    ys = np.arange(y0, y1, spacing_m)

    points: list[tuple[str, float, float]] = []
    for i, xi in enumerate(xs):
        for j, yj in enumerate(ys):
            lon, lat = to_geo.transform(xi, yj)
            pid = f"c{cid}_{i:03d}_{j:03d}"
            points.append((pid, float(lon), float(lat)))

    logger.info("Cluster %d: generated %d grid points (%.0f m spacing)",
                cid, len(points), spacing_m)
    return points


# ---------------------------------------------------------------------------
# Steps 6–7: fetch + extract + waveform
# ---------------------------------------------------------------------------

def fetch_observations(
    cluster: dict,
    points: list[tuple[str, float, float]],
    working_dir: Path,
    stac_start: str,
    stac_end: str,
    cloud_max: int,
    obs_cache: Path | None,
) -> "pd.DataFrame":
    """STAC search → chip fetch → extract → raw observation DataFrame.

    If obs_cache exists, load from it and skip all network/disk I/O.
    If obs_cache is given but doesn't exist, write it after extraction.

    Columns: point_id, date, B05, B07, B08, B11, scl_purity, aot, vza, sza
    """
    import pandas as pd

    if obs_cache is not None and obs_cache.exists():
        logger.info("Loading observation cache: %s", obs_cache)
        return pd.read_parquet(obs_cache)

    PROJECT_ROOT = Path(__file__).parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from analysis.primitives.quality import ArchiveStats, score_observation
    from analysis.timeseries.extraction import extract_observations
    from stage0.chip_store import DiskChipStore
    from stage0.fetch import fetch_chips
    from utils.stac import search_sentinel2

    cid = cluster["cluster_id"]
    inputs_dir = working_dir / f"flash_inputs_c{cid}"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cluster %d: STAC search %s → %s", cid, stac_start, stac_end)
    items = search_sentinel2(
        bbox=cluster["search_bbox"],
        start=stac_start,
        end=stac_end,
        cloud_cover_max=cloud_max,
        endpoint=STAC_ENDPOINT,
        collection=S2_COLLECTION,
    )
    if not items:
        logger.warning("Cluster %d: no STAC items found", cid)
        return pd.DataFrame()

    logger.info("Cluster %d: fetching chips for %d items × %d points",
                cid, len(items), len(points))
    asyncio.run(fetch_chips(
        points=points,
        items=items,
        bands=FLASH_BANDS,
        window_px=5,
        inputs_dir=inputs_dir,
        scl_filter=True,
        band_alias=BAND_ALIAS,
    ))

    store = DiskChipStore(inputs_dir)
    extract_bands = ["B05", "B07", "B08", "B11"]
    raw_obs = extract_observations(items, points, store, bands=extract_bands)
    logger.info("Cluster %d: %d raw observations", cid, len(raw_obs))

    if not raw_obs:
        logger.warning("Cluster %d: no usable observations after SCL filter", cid)
        return pd.DataFrame()

    try:
        archive_stats = ArchiveStats.from_observations(raw_obs)
        scored_obs = [score_observation(o, archive_stats) for o in raw_obs]
    except ValueError:
        logger.debug("Cluster %d: skipping greenness_z scoring (no B04)", cid)
        scored_obs = raw_obs

    rows = []
    for obs in scored_obs:
        rows.append({
            "point_id":   obs.point_id,
            "date":       obs.date,
            "B05":        obs.bands.get("B05", float("nan")),
            "B07":        obs.bands.get("B07", float("nan")),
            "B08":        obs.bands.get("B08", float("nan")),
            "B11":        obs.bands.get("B11", float("nan")),
            "scl_purity": obs.quality.scl_purity,
            "aot":        obs.quality.aot,
            "vza":        obs.quality.view_zenith,
            "sza":        obs.quality.sun_zenith,
        })

    df = pd.DataFrame(rows)

    if obs_cache is not None:
        obs_cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(obs_cache, index=False)
        logger.info("Observation cache written: %s", obs_cache)

    return df


def analyse_observations(
    df: "pd.DataFrame",
    points: list[tuple[str, float, float]],
    cluster_id: int,
    anomaly: bool = False,
) -> list[dict]:
    """Compute flowering signal from cached observation DataFrame.

    Two modes:
      anomaly=False (default): quality-weighted peak must exceed FLOWERING_THRESHOLD.
      anomaly=True: detect relative seasonal rise above each point's own baseline.
        Signal = peak_in_window - median_outside_window. No absolute threshold needed.
    """
    import pandas as pd
    from analysis.constants import FLOWERING_WINDOW, FLOWERING_THRESHOLD
    from analysis.primitives.indices import flowering_index

    if df.empty:
        return []

    point_coords: dict[str, tuple[float, float]] = {p: (lon, lat) for p, lon, lat in points}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df["quality"] = (df["scl_purity"] * df["aot"] * df["vza"] * df["sza"])
    df["raw_index"] = df.apply(
        lambda r: flowering_index({"B05": r.B05, "B07": r.B07, "B08": r.B08, "B11": r.B11}),
        axis=1,
    )
    df["weighted_index"] = df["raw_index"] * df["quality"]

    in_window = df["doy"].between(FLOWERING_WINDOW[0], FLOWERING_WINDOW[1])

    results = []
    for pid, grp in df.groupby("point_id"):
        win = grp[in_window]
        out = grp[~in_window]

        if win.empty:
            continue

        if anomaly:
            baseline = out["raw_index"].median() if len(out) >= 3 else 0.0
            win_peak_raw = win["raw_index"].max()
            signal = win_peak_raw - baseline
            peak_row = win.loc[win["raw_index"].idxmax()]
            peak_value = float(signal)
            peak_doy = int(peak_row["doy"])
            years_detected = float(win["year"].nunique())
            if signal <= 0:
                continue
        else:
            # Original absolute threshold on quality-weighted index
            by_year = win.groupby("year")["weighted_index"].max()
            detected = by_year[by_year >= FLOWERING_THRESHOLD]
            if detected.empty:
                continue
            peak_value = float(detected.max())
            peak_year = detected.idxmax()
            peak_doy = int(win[win["year"] == peak_year]["weighted_index"].idxmax())
            peak_doy = int(win.loc[win[win["year"] == peak_year]["weighted_index"].idxmax(), "doy"])
            years_detected = float(len(detected))

        lon, lat = point_coords.get(pid, (float("nan"), float("nan")))
        results.append({
            "cluster_id":     cluster_id,
            "point_id":       pid,
            "lon":            lon,
            "lat":            lat,
            "peak_value":     peak_value,
            "peak_doy":       float(peak_doy),
            "years_detected": years_detected,
        })

    logger.info("Cluster %d: %d points returned signal (%s mode)",
                cluster_id, len(results), "anomaly" if anomaly else "threshold")
    return results


def run_flash_search(
    cluster: dict,
    points: list[tuple[str, float, float]],
    working_dir: Path,
    stac_start: str,
    stac_end: str,
    cloud_max: int,
    obs_cache: Path | None = None,
    anomaly: bool = False,
) -> list[dict]:
    """STAC search → chip fetch → extract → signal detection → result dicts."""
    df = fetch_observations(
        cluster=cluster,
        points=points,
        working_dir=working_dir,
        stac_start=stac_start,
        stac_end=stac_end,
        cloud_max=cloud_max,
        obs_cache=obs_cache,
    )
    return analyse_observations(df, points, cluster["cluster_id"], anomaly=anomaly)


# ---------------------------------------------------------------------------
# Step 8: report
# ---------------------------------------------------------------------------

def _doy_to_mmdd(doy: float) -> str:
    """Convert a day-of-year float to a MM-DD string for the current year."""
    try:
        d = date(date.today().year, 1, 1).replace(
            year=date.today().year
        )
        from datetime import timedelta
        d = datetime(date.today().year, 1, 1) + timedelta(days=int(doy) - 1)
        return d.strftime("%b %d")
    except Exception:
        return f"DOY {int(doy)}"


def report(all_results: list[dict], cache_dir: Path, run_id: str) -> None:
    """Print ranked table to stdout and write CSV."""
    from analysis.constants import FLOWERING_THRESHOLD

    if not all_results:
        print("No flowering signal detected in any cluster.")
        return

    all_results.sort(key=lambda r: r["peak_value"], reverse=True)
    strong_threshold = FLOWERING_THRESHOLD * 2

    # --- CSV ---
    out_csv = cache_dir / f"ala_flower_flash_{run_id}.csv"
    fieldnames = [
        "cluster_id", "point_id", "lon", "lat",
        "peak_value", "peak_doy", "spike_duration",
        "peak_doy_mean", "peak_doy_sd", "years_detected",
    ]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    logger.info("Written: %s", out_csv)

    # --- Stdout table ---
    print(f"\n{'='*76}")
    print(f"  Yellow Flower Flash — ALA candidate clusters   run_id={run_id}")
    print(f"{'='*76}")
    header = f"  {'point_id':<18} {'cluster':>7} {'peak':>6} {'DOY':>4} {'~date':>7} {'yrs':>4}  flag"
    print(header)
    print(f"  {'-'*70}")
    for r in all_results:
        flag = "**" if r["peak_value"] >= strong_threshold else (
            "*" if r["peak_value"] >= FLOWERING_THRESHOLD else ""
        )
        print(
            f"  {r['point_id']:<18} {r['cluster_id']:>7} "
            f"{r['peak_value']:>6.3f} {int(r['peak_doy']):>4} "
            f"{_doy_to_mmdd(r['peak_doy']):>7} "
            f"{int(r['years_detected']):>4}  {flag}"
        )

    print(f"\n  ** peak >= {strong_threshold:.2f} (strong candidate)   * peak >= {FLOWERING_THRESHOLD:.2f} (threshold)")

    # --- Cluster summaries ---
    by_cluster: dict[int, list] = defaultdict(list)
    for r in all_results:
        by_cluster[r["cluster_id"]].append(r["peak_value"])

    print(f"\n{'Cluster summary':}")
    print(f"  {'cluster':>7}  {'n_pts':>5}  {'median_peak':>11}  {'max_peak':>8}  {'above_thresh':>12}")
    print(f"  {'-'*54}")
    for cid, peaks in sorted(by_cluster.items()):
        peaks_arr = sorted(peaks, reverse=True)
        median_p = sorted(peaks)[len(peaks) // 2]
        max_p = peaks_arr[0]
        frac = sum(1 for p in peaks if p >= FLOWERING_THRESHOLD) / len(peaks)
        print(f"  {cid:>7}  {len(peaks):>5}  {median_p:>11.3f}  {max_p:>8.3f}  {frac:>11.0%}")

    print(f"\nFull results written to: {out_csv}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Search for the yellow flower flash at ALA occurrence clusters."
    )
    p.add_argument("--cache-dir", required=True, type=Path,
                   help="Directory containing ala_occurrences.gpkg (or where it will be written)")
    p.add_argument("--working-dir", required=True, type=Path,
                   help="Directory for staged chips (sub-dirs created per cluster)")
    p.add_argument("--species", default="Parkinsonia aculeata",
                   help="ALA species query string (default: 'Parkinsonia aculeata')")
    p.add_argument("--recency-cutoff", default="2015-01-01",
                   help="Only use records on or after this date (YYYY-MM-DD)")
    p.add_argument("--eps-m", type=float, default=500.0,
                   help="DBSCAN search radius in metres (default: 500)")
    p.add_argument("--min-samples", type=int, default=3,
                   help="DBSCAN minimum cluster size (default: 3)")
    p.add_argument("--top-n", type=int, default=3,
                   help="Number of top clusters to process (default: 3)")
    p.add_argument("--bbox-buffer-deg", type=float, default=0.005,
                   help="Degrees to expand cluster bbox for location uncertainty (~500 m)")
    p.add_argument("--grid-spacing-m", type=float, default=100.0,
                   help="Sample point grid spacing in metres (default: 100)")
    p.add_argument("--stac-start", default="2019-01-01",
                   help="STAC search start date (default: 2019-01-01)")
    p.add_argument("--stac-end", default=date.today().isoformat(),
                   help="STAC search end date (default: today)")
    p.add_argument("--cloud-max", type=int, default=30,
                   help="Maximum cloud cover %% for STAC filter (default: 30)")
    p.add_argument("--run-id", default=None,
                   help="Run identifier for output filenames (default: timestamp)")
    p.add_argument("--clusters-only", action="store_true",
                   help="Stop after writing cluster CSV — skip imagery fetch")
    p.add_argument("--bbox", default=None,
                   help="Explicit bbox 'lon_min,lat_min,lon_max,lat_max' — bypasses DBSCAN cluster selection")
    p.add_argument("--obs-cache", default=None, type=Path,
                   help="Path to parquet file for caching raw observations (load if exists, write if not)")
    p.add_argument("--anomaly", action="store_true",
                   help="Use seasonal anomaly detection instead of absolute threshold")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("rasterio.session").setLevel(logging.WARNING)

    args = _parse_args()
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.working_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load ALA data (skipped when --bbox provided)
    if args.bbox:
        lon_min, lat_min, lon_max, lat_max = [float(x) for x in args.bbox.split(",")]
        clusters = [{
            "cluster_id": 0,
            "n_points": 0,
            "centroid_lon": (lon_min + lon_max) / 2,
            "centroid_lat": (lat_min + lat_max) / 2,
            "bbox_lon_min": lon_min, "bbox_lat_min": lat_min,
            "bbox_lon_max": lon_max, "bbox_lat_max": lat_max,
            "bbox_area_ha": 0,
            "density": 0, "recency_score": 0,
            "search_bbox": [lon_min, lat_min, lon_max, lat_max],
        }]
        logger.info("Using explicit bbox: %s", args.bbox)
    else:
        gdf = load_occurrences(args.cache_dir, args.species)
        logger.info("Loaded %d ALA records", len(gdf))

        # Steps 2–4: cluster and select top-N
        clusters = select_clusters(
            gdf,
            recency_cutoff=args.recency_cutoff,
            eps_m=args.eps_m,
            min_samples=args.min_samples,
            top_n=args.top_n,
            bbox_buffer_deg=args.bbox_buffer_deg,
        )

        if not clusters:
            sys.exit(1)

    # Write cluster summary CSV
    cluster_csv = args.cache_dir / f"ala_clusters_{run_id}.csv"
    cluster_fields = [
        "cluster_id", "n_points", "centroid_lon", "centroid_lat",
        "bbox_lon_min", "bbox_lat_min", "bbox_lon_max", "bbox_lat_max",
        "bbox_area_ha", "density", "recency_score",
    ]
    with open(cluster_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cluster_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(clusters)
    logger.info("Cluster summary written: %s", cluster_csv)

    if args.clusters_only:
        logger.info("--clusters-only: stopping after cluster selection")
        return

    # Steps 5–7: for each cluster, generate points + fetch + extract
    all_results: list[dict] = []
    for cluster in clusters:
        points = make_grid_points(cluster, spacing_m=args.grid_spacing_m)
        if not points:
            continue
        obs_cache = args.obs_cache
        if obs_cache is not None and len(clusters) > 1:
            # Disambiguate cache path per cluster when multiple clusters run
            obs_cache = obs_cache.with_stem(f"{obs_cache.stem}_c{cluster['cluster_id']}")
        results = run_flash_search(
            cluster=cluster,
            points=points,
            working_dir=args.working_dir,
            stac_start=args.stac_start,
            stac_end=args.stac_end,
            cloud_max=args.cloud_max,
            obs_cache=obs_cache,
            anomaly=args.anomaly,
        )
        all_results.extend(results)

    # Step 8: report
    report(all_results, args.cache_dir, run_id)


if __name__ == "__main__":
    main()
