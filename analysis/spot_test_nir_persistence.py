"""Spot test: dry-season NIR persistence over the Longreach infestation.

Signal hypothesis: Parkinsonia (deep-rooted leguminous tree) holds higher NIR
reflectance in the dry season (Aug–Oct) than surrounding dormant grass/bare soil.

Fetches all available S2 scenes Aug–Oct 2025, renders a three-panel PNG per scene:
  - Left:   True colour (contrast-stretched)
  - Centre: NIR reflectance (grey-to-white) with orange patch boundary
  - Right:  NIR delta = per-pixel NIR minus scene median, diverging RdBu colourmap
            (red = above median, blue = below); orange patch boundary overlaid

Output: input-img/spot_nir/

Patch bbox (LONGREACH.md): lon [145.4213, 145.4287], lat [-22.7671, -22.7597]
Scene bbox (5km):          [145.40019, -22.78903, 145.4494, -22.74436]
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
BOX_M      = 5000
UTM_CRS    = "EPSG:32755"

STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
CLOUD_MAX     = 30
RESOLUTION    = 10

# All Aug–Oct dry-season passes
STAC_START = "2025-08-01"
STAC_END   = "2025-10-31"

# NIR asset name at earth-search
BANDS = {
    "nir":   "nir",
    "red":   "red",
    "green": "green",
    "blue":  "blue",
}

# Infestation patch bbox from LONGREACH.md (lon_min, lat_min, lon_max, lat_max)
ORANGE_BOX_WGS84 = [145.4213, -22.7671, 145.4287, -22.7597]

NIR_VMIN = 500    # raw DN  (~0.05 refl)
NIR_VMAX = 3500   # raw DN  (~0.35 refl)

OUT_DIR = PROJECT_ROOT / "input-img" / "spot_nir"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utm_bbox(lon, lat, half_m, utm_crs):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    cx, cy = t.transform(lon, lat)
    return cx - half_m, cy - half_m, cx + half_m, cy + half_m


def wgs84_bbox(xmin, ymin, xmax, ymax, utm_crs):
    from pyproj import Transformer
    t = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(xmin, ymin)
    lon1, lat1 = t.transform(xmax, ymax)
    return [min(lon0,lon1), min(lat0,lat1), max(lon0,lon1), max(lat0,lat1)]


def fetch_scene(item, bounds_utm, epsg, resolution):
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
    band_vals = list(arr.coords["band"].values)
    for b_name, asset_name in BANDS.items():
        if asset_name in band_vals:
            idx = band_vals.index(asset_name)
            result[b_name] = arr.values[idx].astype(np.float32)
    return result


def make_rgb(bands, percentile=2.0):
    rgb = np.stack([bands["red"], bands["green"], bands["blue"]], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb)
    for i in range(3):
        ch = rgb[:, :, i]
        lo = np.nanpercentile(ch, percentile)
        hi = np.nanpercentile(ch, 100 - percentile)
        out[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)


def nir_grey(nir_dn, vmin=NIR_VMIN, vmax=NIR_VMAX):
    """Map NIR DN to uint8 (H, W, 3) greyscale."""
    norm = np.clip((nir_dn.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    grey = (norm * 255).astype(np.uint8)
    return np.stack([grey, grey, grey], axis=-1)


def box_to_pixels(wgs84_box, scene_wgs84_bbox, img_shape):
    """Convert a WGS84 bounding box to pixel coords within the scene image.

    scene_wgs84_bbox: [lon_min, lat_min, lon_max, lat_max]
    img_shape: (H, W)
    Returns (px0, py0, px1, py1) in pixel space (top-left origin).
    """
    H, W = img_shape
    lon_min_s, lat_min_s, lon_max_s, lat_max_s = scene_wgs84_bbox
    lon_min_b, lat_min_b, lon_max_b, lat_max_b = wgs84_box

    px0 = int((lon_min_b - lon_min_s) / (lon_max_s - lon_min_s) * W)
    px1 = int((lon_max_b - lon_min_s) / (lon_max_s - lon_min_s) * W)
    # lat is inverted (top = max lat)
    py0 = int((lat_max_s - lat_max_b) / (lat_max_s - lat_min_s) * H)
    py1 = int((lat_max_s - lat_min_b) / (lat_max_s - lat_min_s) * H)

    return (
        max(0, px0), max(0, py0),
        min(W - 1, px1), min(H - 1, py1),
    )


def draw_box(img_arr, px0, py0, px1, py1, colour=(255, 120, 0), thickness=3):
    """Draw a rectangle onto a (H, W, 3) uint8 array in-place."""
    for t in range(thickness):
        # top
        img_arr[max(0,py0-t), px0:px1] = colour
        # bottom
        img_arr[min(img_arr.shape[0]-1,py1+t), px0:px1] = colour
        # left
        img_arr[py0:py1, max(0,px0-t)] = colour
        # right
        img_arr[py0:py1, min(img_arr.shape[1]-1,px1+t)] = colour


def local_annulus(px0, py0, px1, py1, img_shape, buffer_px: int):
    """Return pixel bounds of a buffer annulus around the patch box.

    The annulus is a rectangle expanded by buffer_px on each side, clipped to
    the image, with the inner patch excluded. Used as the local reference window
    so that distant bright features (river corridors, tree lines) don't bias the
    comparison.
    """
    H, W = img_shape
    ax0 = max(0, px0 - buffer_px)
    ay0 = max(0, py0 - buffer_px)
    ax1 = min(W - 1, px1 + buffer_px)
    ay1 = min(H - 1, py1 + buffer_px)
    return ax0, ay0, ax1, ay1


def stats_in_box(nir_dn, px0, py0, px1, py1, buffer_px: int = 50):
    """Patch mean vs. local annulus mean.

    buffer_px: width of the reference annulus in pixels (default 50 px = 500 m
    at 10 m/px), chosen to sample the immediate gilgai/grass surroundings while
    excluding distant river corridors and tree lines.
    """
    H, W = nir_dn.shape
    chip = nir_dn[py0:py1, px0:px1].astype(np.float32)

    ax0, ay0, ax1, ay1 = local_annulus(px0, py0, px1, py1, (H, W), buffer_px)
    annulus = nir_dn[ay0:ay1, ax0:ax1].astype(np.float32).copy()
    # blank out the patch interior so it doesn't contribute to the reference
    inner_y0 = py0 - ay0
    inner_x0 = px0 - ax0
    inner_y1 = inner_y0 + (py1 - py0)
    inner_x1 = inner_x0 + (px1 - px0)
    annulus[inner_y0:inner_y1, inner_x0:inner_x1] = np.nan

    return {
        "box_mean":    float(np.nanmean(chip)),
        "box_p50":     float(np.nanpercentile(chip, 50)),
        "local_mean":  float(np.nanmean(annulus)),
        "local_p50":   float(np.nanpercentile(annulus, 50)),
        "annulus_px":  buffer_px,
    }


def nir_delta_img(nir_dn: np.ndarray, px0, py0, px1, py1,
                  buffer_px: int = 50, vabs: float = 600.0) -> np.ndarray:
    """NIR delta = per-pixel minus local annulus median, RdBu_r colourmap.

    Red = above local reference (relatively high NIR), blue = below.
    Using the local annulus median keeps the reference tied to the same
    landscape type (gilgai clay / dry grass) as the patch itself.
    vabs: symmetric clamp in raw DN (~600 DN ≈ 0.06 refl).
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    H, W = nir_dn.shape
    ax0, ay0, ax1, ay1 = local_annulus(px0, py0, px1, py1, (H, W), buffer_px)
    annulus = nir_dn[ay0:ay1, ax0:ax1].astype(np.float32).copy()
    inner_y0 = py0 - ay0
    inner_x0 = px0 - ax0
    annulus[inner_y0:inner_y0+(py1-py0), inner_x0:inner_x0+(px1-px0)] = np.nan
    local_ref = float(np.nanmedian(annulus))

    delta = nir_dn.astype(np.float32) - local_ref
    norm  = mcolors.Normalize(vmin=-vabs, vmax=vabs)
    cmap  = cm.get_cmap("RdBu_r")
    rgba  = cmap(norm(delta))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def save_threepanel(rgb, nir_img, delta_img, path, title):
    """TC | NIR grey | NIR delta — three panels side by side."""
    from PIL import Image, ImageDraw, ImageFont
    H, W = rgb.shape[:2]
    GAP  = 4
    BAR  = 28
    canvas = np.zeros((H + BAR, W * 3 + GAP * 2, 3), dtype=np.uint8)
    canvas[BAR:, :W]               = rgb
    canvas[BAR:, W+GAP:W*2+GAP]    = nir_img
    canvas[BAR:, W*2+GAP*2:]       = delta_img
    canvas[:BAR]                   = [30, 30, 30]
    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 6),
              f"True Colour  |  NIR (grey=low, white=high)  |  NIR delta vs scene median (red=above, blue=below)   {title}",
              fill=(220, 220, 220), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    import pystac_client

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Cap the number of scenes (evenly spaced); default = all")
    args = parser.parse_args()

    bounds_utm   = utm_bbox(CENTRE_LON, CENTRE_LAT, BOX_M / 2, UTM_CRS)
    bbox_wgs84   = wgs84_bbox(*bounds_utm, UTM_CRS)
    epsg_int     = int(UTM_CRS.split(":")[1])

    print(f"Scene bbox (WGS84): {[round(x,5) for x in bbox_wgs84]}")
    print(f"Orange box (WGS84): {ORANGE_BOX_WGS84}")

    catalog = pystac_client.Client.open(STAC_ENDPOINT)
    search  = catalog.search(
        collections=[S2_COLLECTION],
        bbox=bbox_wgs84,
        datetime=f"{STAC_START}/{STAC_END}",
        query={"eo:cloud_cover": {"lt": CLOUD_MAX}},
    )
    items = list(search.items())
    items.sort(key=lambda i: i.datetime)
    print(f"\n{len(items)} scenes found ({STAC_START} → {STAC_END}, cloud < {CLOUD_MAX}%)")

    if args.max_scenes and args.max_scenes < len(items):
        step  = max(1, len(items) // args.max_scenes)
        items = items[::step][:args.max_scenes]
        print(f"Thinned to {len(items)} scenes (--max-scenes {args.max_scenes})")

    print(f"Dates: {[i.datetime.strftime('%Y-%m-%d') for i in items]}\n")

    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        print(f"[{date_str}] fetching ...", flush=True)
        try:
            bands = fetch_scene(item, bounds_utm, epsg_int, RESOLUTION)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if "nir" not in bands:
            print("  No NIR band — skipping")
            continue

        nir = bands["nir"]
        H, W = nir.shape

        # Pixel coords of orange box within this scene
        px0, py0, px1, py1 = box_to_pixels(ORANGE_BOX_WGS84, bbox_wgs84, (H, W))

        # Stats (local annulus reference)
        s = stats_in_box(nir, px0, py0, px1, py1)
        refl_box   = s["box_mean"]   * 0.0001 - 0.1
        refl_local = s["local_mean"] * 0.0001 - 0.1
        print(f"  NIR — box mean DN={s['box_mean']:.0f} (refl≈{refl_box:.3f})  "
              f"local ({s['annulus_px']}px annulus) mean DN={s['local_mean']:.0f} (refl≈{refl_local:.3f})  "
              f"Δ={refl_box-refl_local:+.3f}")

        # Build images
        rgb_img   = make_rgb(bands)
        nir_img   = nir_grey(nir)
        delta_img = nir_delta_img(nir, px0, py0, px1, py1)

        # Draw patch box on all three panels
        draw_box(rgb_img,   px0, py0, px1, py1)
        draw_box(nir_img,   px0, py0, px1, py1)
        draw_box(delta_img, px0, py0, px1, py1)

        path = OUT_DIR / f"nir_{date_str}.png"
        save_threepanel(rgb_img, nir_img, delta_img, path, date_str)

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
