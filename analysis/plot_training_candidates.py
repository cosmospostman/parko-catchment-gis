"""
plot_training_candidates.py — plot Parkinsonia sightings across NT and western QLD
at point level to help select a training region bbox.

Reads the already-fetched GeoPackage from fetch_ala_australia.py.

Usage:
    python analysis/plot_training_candidates.py

Output:
    outputs/australia_occurrences/training_candidates.png
"""
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

GPKG_PATH = Path(__file__).parent.parent / "outputs" / "australia_occurrences" / "ala_australia_occurrences.gpkg"
OUT_PATH  = Path(__file__).parent.parent / "outputs" / "australia_occurrences" / "training_candidates.png"

# Zoom extent: NT + western QLD
PLOT_EXTENT = [128.0, -22.0, 148.0, -10.0]

# Mitchell catchment bbox (for reference)
MITCHELL_BBOX = [141.3453505, -18.23350524, 145.51819104, -14.92188926]

# Candidate training regions — edit/add as needed
CANDIDATES = [
    {
        "label": "NT — Katherine / Daly River",
        "bbox": [130.0, -16.5, 133.5, -13.5],
        "color": "#e6550d",
    },
    {
        "label": "NT — McArthur River",
        "bbox": [135.5, -19.0, 138.5, -16.5],
        "color": "#756bb1",
    },
]

logger = logging.getLogger(__name__)


def count_in_bbox(gdf: gpd.GeoDataFrame, bbox: list[float]) -> int:
    minx, miny, maxx, maxy = bbox
    return len(gdf.cx[minx:maxx, miny:maxy])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    if not GPKG_PATH.exists():
        logger.error("GeoPackage not found: %s — run fetch_ala_australia.py first", GPKG_PATH)
        sys.exit(1)

    gdf = gpd.read_file(GPKG_PATH)
    logger.info("Loaded %d records from %s", len(gdf), GPKG_PATH)

    # Clip to plot extent for faster rendering
    minx, miny, maxx, maxy = PLOT_EXTENT
    gdf_clip = gdf.cx[minx:maxx, miny:maxy]
    logger.info("Records in plot extent: %d", len(gdf_clip))

    # Load basemap
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f5f0eb")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#999")
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="#bbb")
        ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="#aac4d4", alpha=0.6)
        transform = ccrs.PlateCarree()
    except ImportError:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_facecolor("#f5f0eb")
        transform = None

    # Plot points
    ax.scatter(
        gdf_clip.geometry.x, gdf_clip.geometry.y,
        s=4, color="#2b8cbe", alpha=0.4, linewidths=0,
        label="ALA sightings",
        **({} if transform is None else {"transform": transform}),
    )

    # Mitchell bbox
    mx0, my0, mx1, my1 = MITCHELL_BBOX
    rect_kwargs = dict(linewidth=2, edgecolor="black", facecolor="none", linestyle="--",
                       **({} if transform is None else {"transform": transform}))
    ax.add_patch(Rectangle((mx0, my0), mx1 - mx0, my1 - my0, **rect_kwargs))
    ax.text(mx0, my1 + 0.15, "Mitchell catchment", fontsize=8, color="black",
            **({} if transform is None else {"transform": transform}))

    # Candidate bboxes
    legend_handles = [
        plt.scatter([], [], s=20, color="#2b8cbe", alpha=0.6, label="ALA sightings"),
        mpatches.Patch(facecolor="none", edgecolor="black", linestyle="--", linewidth=2,
                       label="Mitchell catchment"),
    ]
    for cand in CANDIDATES:
        cx0, cy0, cx1, cy1 = cand["bbox"]
        n = count_in_bbox(gdf, cand["bbox"])
        rect = Rectangle(
            (cx0, cy0), cx1 - cx0, cy1 - cy0,
            linewidth=2, edgecolor=cand["color"], facecolor=cand["color"], alpha=0.12,
            **({} if transform is None else {"transform": transform}),
        )
        ax.add_patch(rect)
        ax.add_patch(Rectangle(
            (cx0, cy0), cx1 - cx0, cy1 - cy0,
            linewidth=2, edgecolor=cand["color"], facecolor="none",
            **({} if transform is None else {"transform": transform}),
        ))
        ax.text(cx0 + 0.1, cy1 - 0.5, f"{cand['label']}\nn={n:,}",
                fontsize=8, color=cand["color"], va="top",
                **({} if transform is None else {"transform": transform}))
        legend_handles.append(
            mpatches.Patch(facecolor=cand["color"], alpha=0.4,
                           edgecolor=cand["color"], label=f"{cand['label']} (n={n:,})")
        )
        logger.info("%-40s  %d sightings", cand["label"], n)

    ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
    ax.set_title(
        "Parkinsonia aculeata — candidate training regions\n"
        "NT and western QLD sightings (ALA)",
        fontsize=13,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", OUT_PATH)
    plt.close(fig)


if __name__ == "__main__":
    main()
