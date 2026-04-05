"""Spot test: flowering index quicklooks over the Longreach Parkinsonia infestation.

Fetches Sentinel-2 scenes for:
  - Apr–May 2025  (pre-flowering baseline)
  - Aug–Sep 2025  (expected flowering peak)

For each scene produces a side-by-side PNG: true colour | flowering index (water-masked).
Output: input-img/spot_longreach/

Centre: -22.7667, 145.4248  (known infestation, orange-box site)
Box:    5 × 5 km

Usage:
    python analysis/spot_test_longreach.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CENTRE_LON = 145.4248
CENTRE_LAT = -22.7667
BOX_M      = 5000          # half-side in metres → 5 × 5 km box
UTM_CRS    = "EPSG:32755"  # WGS 84 / UTM zone 55S

STAC_ENDPOINT  = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION  = "sentinel-2-l2a"
CLOUD_MAX      = 30
RESOLUTION     = 10        # metres per pixel

# Band asset names at earth-search (NOT B05/B08 style)
BANDS = {
    "B05": "rededge1",
    "B07": "rededge3",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
    "red": "red",
    "green": "green",
    "blue": "blue",
}

NIR_DN_WATER_THRESHOLD = 1500   # raw DN; ~0.05 reflectance

PERIODS = [
    ("2025-04-01", "2025-05-31", "Apr-May 2025 (baseline)"),
    ("2025-08-01", "2025-09-30", "Aug-Sep 2025 (flowering)"),
]

OUT_DIR = PROJECT_ROOT / "input-img" / "spot_longreach"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utm_bbox(lon: float, lat: float, half_m: float, utm_crs: str) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in UTM metres centred on lon/lat."""
    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    return cx - half_m, cy - half_m, cx + half_m, cy + half_m


def wgs84_bbox(xmin, ymin, xmax, ymax, utm_crs: str) -> list[float]:
    """Convert UTM bbox to WGS84 [lon_min, lat_min, lon_max, lat_max]."""
    from pyproj import Transformer
    to_geo = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = to_geo.transform(xmin, ymin)
    lon1, lat1 = to_geo.transform(xmax, ymax)
    return [min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1)]


def fetch_scene(item, bounds_utm: tuple, epsg: int, resolution: int) -> dict[str, np.ndarray]:
    """Load one STAC item into a dict of raw-DN numpy arrays (rescale=False)."""
    import stackstac

    assets = list(BANDS.values())
    da = stackstac.stack(
        [item],
        assets=assets,
        resolution=resolution,
        bounds=bounds_utm,
        epsg=epsg,
        rescale=False,
        chunksize=2048,
    )
    arr = da.squeeze("time").compute()

    result = {}
    for b_name, asset_name in BANDS.items():
        if asset_name in arr.coords["band"].values:
            idx = list(arr.coords["band"].values).index(asset_name)
            result[b_name] = arr.values[idx].astype(np.float32)
    return result


def flowering_index_arr(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Vectorised flowering index from raw DN arrays (no pre-scaling needed — ratios cancel)."""
    EPS = 1e-9
    b05 = bands["B05"].astype(np.float64)
    b07 = bands["B07"].astype(np.float64)
    b08 = bands["B08"].astype(np.float64)
    b11 = bands["B11"].astype(np.float64)
    re_slope = (b07 - b05) / (b07 + b05 + EPS)
    nir_swir  = (b08 - b11) / (b08 + b11 + EPS)
    fi = (re_slope + nir_swir) / 2.0
    return np.clip(fi, -1.0, 1.0).astype(np.float32)


def water_mask(nir_dn: np.ndarray) -> np.ndarray:
    """Boolean mask: True where pixel is likely water (NIR DN < threshold)."""
    return nir_dn < NIR_DN_WATER_THRESHOLD


def make_rgb(bands: dict[str, np.ndarray], percentile: float = 2.0) -> np.ndarray:
    """Produce a uint8 (H, W, 3) RGB array from raw DN, contrast-stretched."""
    rgb = np.stack([bands["red"], bands["green"], bands["blue"]], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb)
    for i in range(3):
        ch = rgb[:, :, i]
        lo = np.nanpercentile(ch, percentile)
        hi = np.nanpercentile(ch, 100 - percentile)
        out[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)


def fi_colormap(fi: np.ndarray, water: np.ndarray, vmin=-0.2, vmax=0.5) -> np.ndarray:
    """Map FI values to uint8 (H, W, 3) RGB using RdYlGn; blue for water."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("RdYlGn")
    rgba = cmap(norm(fi))
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[water] = [20, 20, 100]   # blue for water-masked pixels
    return rgb


def re_ratio_arr(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Red-edge chlorophyll ratio B07/B05.

    Measures chlorophyll concentration independently of canopy structure.
    In dry season, senescent grasses collapse toward B07/B05 ≈ 1; Parkinsonia
    (retaining active chlorophyll) stays elevated. Ratio range ~[0.5, 3+].
    """
    EPS = 1e-9
    b05 = bands["B05"].astype(np.float64)
    b07 = bands["B07"].astype(np.float64)
    return (b07 / (b05 + EPS)).astype(np.float32)


def swir_water_arr(bands: dict[str, np.ndarray]) -> np.ndarray:
    """SWIR canopy-water index (B08 - B11) / (B08 + B11).

    A normalised difference that isolates canopy liquid-water absorption at
    1.6 µm (B11). Parkinsonia's deep roots sustain higher canopy water
    year-round; surrounding dry grass/bare soil have low water content and
    so fall toward -1. Values near +0.5 indicate moist, green canopy.
    """
    EPS = 1e-9
    b08 = bands["B08"].astype(np.float64)
    b11 = bands["B11"].astype(np.float64)
    ndwi = (b08 - b11) / (b08 + b11 + EPS)
    return np.clip(ndwi, -1.0, 1.0).astype(np.float32)


def ratio_colormap(
    arr: np.ndarray,
    water: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str = "YlOrRd",
) -> np.ndarray:
    """Generic single-band → uint8 RGB colourmap with water mask."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(arr))
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[water] = [20, 20, 100]
    return rgb


def save_sidebyside(rgb: np.ndarray, fi_rgb: np.ndarray, path: Path, title: str) -> None:
    """Save a 1:1 pixel side-by-side PNG (TC left, FI right) with a thin divider."""
    from PIL import Image, ImageDraw, ImageFont

    H, W = rgb.shape[:2]
    GAP   = 4
    BAR   = 24   # title bar height

    canvas = np.zeros((H + BAR, W * 2 + GAP, 3), dtype=np.uint8)
    canvas[BAR:, :W]        = rgb
    canvas[BAR:, W+GAP:]    = fi_rgb
    canvas[:BAR]            = [30, 30, 30]   # dark title bar

    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    draw.text((4, 4), f"True Colour  |  Flowering Index (RdYlGn, water=blue)   {title}", fill=(220, 220, 220), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    print(f"  Saved: {path.name}  ({W*2+GAP}×{H+BAR} px)")


def save_foursquare(
    rgb: np.ndarray,
    fi_rgb: np.ndarray,
    re_rgb: np.ndarray,
    swir_rgb: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Save a 2×2 grid PNG: TC | FI (top row) / RE ratio | SWIR water (bottom row)."""
    from PIL import Image, ImageDraw, ImageFont

    H, W = rgb.shape[:2]
    GAP = 4
    BAR = 24

    canvas = np.zeros((H * 2 + GAP + BAR, W * 2 + GAP, 3), dtype=np.uint8)
    canvas[:BAR] = [30, 30, 30]
    # top row
    canvas[BAR:BAR+H, :W]       = rgb
    canvas[BAR:BAR+H, W+GAP:]   = fi_rgb
    # bottom row
    canvas[BAR+H+GAP:, :W]      = re_rgb
    canvas[BAR+H+GAP:, W+GAP:]  = swir_rgb

    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    labels = [
        (4,        BAR + 4,      "True Colour"),
        (W + GAP + 4, BAR + 4,   "Flowering Index (RdYlGn)"),
        (4,        BAR + H + GAP + 4, "Red-edge ratio B07/B05 (YlOrRd)"),
        (W + GAP + 4, BAR + H + GAP + 4, "SWIR water (B08-B11)/(B08+B11) (RdYlGn)"),
    ]
    draw.text((4, 6), title, fill=(220, 220, 220), font=font)
    small_font = font
    for x, y, lbl in labels:
        draw.text((x, y), lbl, fill=(240, 240, 100), font=small_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    print(f"  Saved: {path.name}  ({W*2+GAP}×{H*2+GAP+BAR} px)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import pystac_client

    bounds_utm = utm_bbox(CENTRE_LON, CENTRE_LAT, BOX_M / 2, UTM_CRS)
    bbox_wgs84 = wgs84_bbox(*bounds_utm, UTM_CRS)
    epsg_int   = int(UTM_CRS.split(":")[1])

    print(f"Centre: {CENTRE_LAT}, {CENTRE_LON}")
    print(f"UTM bounds: {[round(x, 1) for x in bounds_utm]}")
    print(f"WGS84 bbox: {[round(x, 5) for x in bbox_wgs84]}")
    print(f"Output dir: {OUT_DIR}\n")

    catalog = pystac_client.Client.open(STAC_ENDPOINT)

    for start, end, label in PERIODS:
        print(f"\n{'='*60}")
        print(f"Period: {label}  ({start} → {end})")
        search = catalog.search(
            collections=[S2_COLLECTION],
            bbox=bbox_wgs84,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": CLOUD_MAX}},
        )
        items = list(search.items())
        print(f"  {len(items)} scenes found (cloud < {CLOUD_MAX}%)")

        if not items:
            print("  No scenes — skipping period.")
            continue

        # Sort by date, pick up to 4 scenes spread across the period
        items.sort(key=lambda i: i.datetime)
        step  = max(1, len(items) // 4)
        picks = items[::step][:4]
        print(f"  Processing {len(picks)} scenes: {[i.datetime.strftime('%Y-%m-%d') for i in picks]}")

        for item in picks:
            date_str = item.datetime.strftime("%Y-%m-%d")
            print(f"\n  [{date_str}] fetching bands ...", flush=True)
            try:
                bands = fetch_scene(item, bounds_utm, epsg_int, RESOLUTION)
            except Exception as e:
                print(f"  ERROR fetching {date_str}: {e}")
                continue

            missing = [b for b in ("B05","B07","B08","B11","red","green","blue") if b not in bands]
            if missing:
                print(f"  Missing bands {missing} — skipping")
                continue

            water   = water_mask(bands["B08"])
            fi      = flowering_index_arr(bands)
            re      = re_ratio_arr(bands)
            swir_w  = swir_water_arr(bands)

            rgb_img  = make_rgb(bands)
            fi_img   = fi_colormap(fi, water)
            re_img   = ratio_colormap(re, water, vmin=0.8, vmax=2.5, cmap_name="YlOrRd")
            swir_img = ratio_colormap(swir_w, water, vmin=-0.2, vmax=0.6, cmap_name="RdYlGn")

            n_water = int(water.sum())
            n_total = water.size
            fi_land   = fi[~water]
            re_land   = re[~water]
            swir_land = swir_w[~water]
            print(f"    Water mask: {n_water}/{n_total} px ({100*n_water/n_total:.1f}%)")
            print(f"    FI   land: mean={fi_land.mean():.3f}  p90={np.percentile(fi_land,90):.3f}  max={fi_land.max():.3f}")
            print(f"    RE   land: mean={re_land.mean():.3f}  p90={np.percentile(re_land,90):.3f}  max={re_land.max():.3f}")
            print(f"    SWIR land: mean={swir_land.mean():.3f}  p90={np.percentile(swir_land,90):.3f}  max={swir_land.max():.3f}")

            tag   = label.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
            fname = f"spot_{date_str}_{tag}.png"
            save_foursquare(
                rgb_img, fi_img, re_img, swir_img,
                OUT_DIR / fname,
                f"{date_str} — {label}",
            )

    print(f"\nDone. All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
