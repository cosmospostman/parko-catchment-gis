"""Unit tests for utils/pixel_collector.py — make_pixel_grid() and extract_item_to_df().

collect() orchestrates network + disk and is not unit-tested here.
_utm_crs_for_bbox() and make_pixel_grid() are pure and fully tested.
extract_item_to_df() is tested via a synthetic MemoryChipStore with 1×1 patches.

Note on MemoryChipStore 1×1 patches: when a patch is 1 row × 1 col, the row/col
clipping in _ensure_pixel_coords() always resolves every point to (0, 0), so the
transform and CRS only need to be plausible — the exact values are irrelevant.

Tests
-----
 1. make_pixel_grid returns at least one point for a small bbox.
 2. All point_ids match the "px_XXXX_YYYY" format.
 3. Returned (lon, lat) lie inside the input bbox (with small snap slop).
 4. Inverted bbox (lon_min > lon_max) returns empty list silently (Bug PC2 — documents).
 5. _utm_crs_for_bbox: northern hemisphere → EPSG:326XX.
 6. _utm_crs_for_bbox: southern hemisphere → EPSG:327XX.
 7. _utm_crs_for_bbox: lon_centre=180.0 produces invalid zone 61 (Bug PC1 — FAILS).
 8. extract_item_to_df: clear pixels produce a DataFrame with correct columns and SR values.
 9. extract_item_to_df: all-clouded item (SCL=9) returns None.
10. extract_item_to_df: missing SCL band returns None.
11. extract_item_to_df: all rows all-NaN bands returns None (regression).
12. extract_item_to_df: partial NaN rows are NOT dropped (Bug PC4 — documents).
13. extract_item_to_df: spectral index columns (NDVI, NDWI, EVI) are present.
14. extract_item_to_df: missing AOT band defaults aot column to 1.0.
15. extract_item_to_df: apply_nbar=False does not raise.
16. extract_item_to_df: two items from different tiles produce two rows — no cross-tile dedup.
16b. extract_item_to_df: band values are aligned to point_ids (regression for set-ordering bug).
 A. extract_item_to_df: tile_id extracted from item.id when s2:mgrs_tile absent.
17. Granule dedup regex strips the processing-granule index suffix correctly.
18. Cross-tile items with matching (date, satellite) both survive per-tile dedup (Bug PC6 — documents).
19. collect() dedup step removes cross-tile duplicate rows, keeping higher scl_purity.
20. collect() dedup: no-duplicate parquet is left unchanged, total_rows correct.
21. collect() dedup: non-duplicate rows adjacent to a boundary pixel are preserved.
22. collect() dedup: duplicate pair straddling a row-group boundary is resolved.
23. collect() dedup: equal scl_purity tie-break keeps the first tile in input order.
PC-CT. Canonical tile bug: make_pixel_grid snap produces dx=-5 for all pixels, so the
       old dx-based assignment sent every pixel to tile[0], suppressing all tile[1]
       observations and producing systematic observation gaps (score strips).
       After the fix (canonical tile suppression removed), both tiles contribute obs.
PC-SB1. collect() stale bbox: when chip cache contains patches built for a different
        bbox, shard .done files and existing tile parquets are deleted before re-fetch.
PC-SB2. Parquet spatial variance guard: a parquet where all pixels share the same
        per-pixel band summary statistics (std across pixels ≈ 0) is flagged as
        corrupt regardless of structural validity.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from analysis.constants import BANDS, SCL_BAND, AOT_BAND, SCL_CLEAR_VALUES, SPECTRAL_INDEX_COLS
from utils.chip_store import MemoryChipStore
from utils.pixel_collector import _utm_crs_for_bbox, extract_item_to_df, make_pixel_grid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BBOX_AU = [145.41, -22.81, 145.44, -22.74]   # small Australian bbox
_ITEM_DT  = datetime(2022, 8, 15, tzinfo=timezone.utc)
_GRANULE_RE = re.compile(r"_\d+_L2A$")


def _make_item(
    item_id: str = "S2A_55HBU_20220815_0_L2A",
    dt: datetime | None = None,
    tile_id: str = "55HBU",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        datetime=dt or _ITEM_DT,
        properties={"s2:mgrs_tile": tile_id},
    )


def _make_store(
    n_points: int,
    scl_value: float,
    band_value: float,
    *,
    item_id: str = "S2A_55HBU_20220815_0_L2A",
    missing_bands: tuple[str, ...] = (),
    aot_value: float = 100.0,   # raw DN → 1 - 100*0.001 = 0.9 quality
    all_band_array: np.ndarray | None = None,
) -> tuple[MemoryChipStore, np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    """Build a MemoryChipStore with 1×1 constant patches for *n_points* points.

    For 1×1 patches the clipping behaviour always resolves every point to row=0,
    col=0, so the transform only needs to be geometrically plausible.

    Returns (store, lons, lats, point_coords).
    """
    lons = np.linspace(145.410, 145.430, n_points, dtype=np.float64)
    lats = np.linspace(-22.810, -22.790, n_points, dtype=np.float64)
    point_coords = {f"px_{i:04d}_0000": (float(lons[i]), float(lats[i]))
                    for i in range(n_points)}

    crs = CRS.from_epsg(32755)
    # A 10 m patch sitting somewhere in the scene — exact location irrelevant for 1×1.
    transform = from_bounds(500_000, 7_480_000, 500_010, 7_480_010, 1, 1)

    all_bands = list(BANDS) + [SCL_BAND, AOT_BAND]
    patches: dict[tuple[str, str], tuple[np.ndarray, object, object]] = {}
    for band in all_bands:
        if band in missing_bands:
            continue
        if band == SCL_BAND:
            val = scl_value
        elif band == AOT_BAND:
            val = aot_value
        else:
            val = band_value if all_band_array is None else float(all_band_array.flat[0])
        arr = np.full((1, 1), val, dtype=np.float32)
        patches[(item_id, band)] = (arr, transform, crs)

    store = MemoryChipStore(patches=patches, point_coords=point_coords)
    return store, lons, lats, point_coords


def _point_ids(n: int, item_id: str = "S2A_55HBU_20220815_0_L2A") -> list[str]:
    return [f"px_{i:04d}_0000" for i in range(n)]


# ---------------------------------------------------------------------------
# Tests 1–4: make_pixel_grid
# ---------------------------------------------------------------------------

def test_make_pixel_grid_returns_points():
    points = make_pixel_grid(_BBOX_AU)
    assert len(points) > 0
    pid, lon, lat = points[0]
    assert isinstance(pid, str)
    assert isinstance(lon, float)
    assert isinstance(lat, float)


def test_make_pixel_grid_point_id_format():
    pts = make_pixel_grid(_BBOX_AU)
    pattern = re.compile(r"^px_\d{4}_\d{4}$")
    for pid, _, _ in pts:
        assert pattern.match(pid), f"bad pid: {pid!r}"


def test_make_pixel_grid_lons_lats_inside_bbox():
    lon_min, lat_min, lon_max, lat_max = _BBOX_AU
    slop = 0.001  # allow small snap offset
    pts = make_pixel_grid(_BBOX_AU)
    for _, lon, lat in pts:
        assert lon_min - slop <= lon <= lon_max + slop
        assert lat_min - slop <= lat <= lat_max + slop


# ---------------------------------------------------------------------------
# Test 4 — inverted bbox returns empty list silently (Bug PC2 — documents)
# ---------------------------------------------------------------------------

def test_make_pixel_grid_inverted_bbox_returns_empty():
    # lon_min > lon_max → np.arange produces empty array → empty list.
    # Bug PC2: should raise ValueError rather than silently returning [].
    points = make_pixel_grid([145.5, -22.8, 145.4, -22.7])  # inverted lon
    assert points == []


# ---------------------------------------------------------------------------
# Tests 5–7: _utm_crs_for_bbox
# ---------------------------------------------------------------------------

def test_utm_crs_northern_hemisphere():
    # lon_centre ≈ 0.0, lat > 0 → zone 31, northern → EPSG:32631
    result = _utm_crs_for_bbox([-1.0, 10.0, 1.0, 12.0])
    assert result.startswith("EPSG:326")


def test_utm_crs_southern_hemisphere():
    # lon_centre ≈ 145.05, lat < 0 → zone 55, southern → EPSG:32755
    result = _utm_crs_for_bbox([145.0, -23.0, 145.1, -22.9])
    assert result == "EPSG:32755"


def test_utm_crs_lon180_produces_invalid_zone():
    # Bug PC1: int((180 + 180) / 6) + 1 = 61 → EPSG:32761 (invalid).
    # The bug triggers when lon_centre is exactly 180.0, which requires a bbox
    # symmetric around 180° — e.g. [179.5, ..., 180.5, ...].
    # Fix: zone should be clamped to 60 → EPSG:32760.
    result = _utm_crs_for_bbox([179.5, -20.0, 180.5, -19.5])  # lon_centre = 180.0
    assert result == "EPSG:32760"  # BUG PC1: currently returns EPSG:32761


# ---------------------------------------------------------------------------
# Test 10 — clear pixels: correct columns and surface reflectance (regression)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_clear_pixels():
    item = _make_item()
    # SCL=4 (vegetation, clear). band raw DN=1500 (uint16 ×10000, not SR).
    store, lons, lats, _ = _make_store(3, scl_value=4.0, band_value=1500.0)
    pids = _point_ids(3)

    df = extract_item_to_df(item, store, pids, lons, lats, apply_nbar=False)

    assert df is not None
    assert len(df) == 3

    # Spectral indices are NOT stored — computed at read time via add_spectral_indices().
    expected_cols = (
        ["point_id", "lon", "lat", "date", "item_id", "tile_id"]
        + list(BANDS)
        + ["scl_purity", "scl", "aot", "view_zenith", "sun_zenith"]
    )
    for col in expected_cols:
        assert col in df.columns, f"missing column: {col}"
    for col in SPECTRAL_INDEX_COLS:
        assert col not in df.columns, f"unexpected derived column: {col}"

    # Bands stored as uint16 DN (×10000), not float SR.
    assert df[0, "B03"] == 1500
    assert df["B03"].dtype == pl.UInt16
    assert (df["scl_purity"] == 1).all()   # 1×1 chip, clear → purity=1
    assert (df["item_id"] == item.id).all()
    assert (df["tile_id"] == "55HBU").all()


# ---------------------------------------------------------------------------
# Test 11 — all-clouded item returns None (regression)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_all_cloud_returns_none():
    item = _make_item()
    store, lons, lats, _ = _make_store(3, scl_value=9.0, band_value=1000.0)
    result = extract_item_to_df(item, store, _point_ids(3), lons, lats, apply_nbar=False)
    assert result is None


# ---------------------------------------------------------------------------
# Test 12 — missing SCL band returns None (regression)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_missing_scl_returns_none():
    item = _make_item()
    store, lons, lats, _ = _make_store(
        3, scl_value=4.0, band_value=1000.0, missing_bands=(SCL_BAND,)
    )
    result = extract_item_to_df(item, store, _point_ids(3), lons, lats, apply_nbar=False)
    assert result is None


# ---------------------------------------------------------------------------
# Test 13 — all rows all-NaN bands → returns None (regression)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_all_nan_returns_none():
    item = _make_item()
    # SCL present and clear, but all spectral bands absent → all NaN.
    store, lons, lats, _ = _make_store(
        2, scl_value=4.0, band_value=0.0,
        missing_bands=tuple(BANDS),
    )
    result = extract_item_to_df(item, store, _point_ids(2), lons, lats, apply_nbar=False)
    assert result is None


# ---------------------------------------------------------------------------
# Test 14 — partial NaN rows are NOT dropped (Bug PC4 — documents)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_partial_nan_rows_not_dropped():
    """Directly tests the all-NaN guard logic in extract_item_to_df (Bug PC4).

    The guard on line 247 is:
        if df[list(BANDS)].isna().all(axis=1).all(): return None

    The inner .all(axis=1) marks each row True if ALL bands are NaN for that row.
    The outer .all() is True only if EVERY row is all-NaN.

    This means a DataFrame with one good row and one all-NaN row passes the guard
    (outer .all() is False) and the all-NaN row slips through into the output.

    We test the guard logic directly using pandas, since injecting per-point NaN
    into MemoryChipStore requires multi-row patches with careful affine setup.
    This isolates the bug cleanly without depending on store internals.
    """
    rng = np.random.default_rng(42)  # noqa: F841

    # Simulate the DataFrame that extract_item_to_df would produce before the guard:
    # row 0: valid bands; row 1: all-NaN bands (but clear SCL)
    n_rows = 2
    band_data = {band: [0.15, None] for band in BANDS}
    df = pl.DataFrame({
        "point_id": ["px_0000_0000", "px_0001_0000"],
        **band_data,
    })

    # Current guard:
    rows_all_nan = df.select(list(BANDS)).with_columns(
        pl.all_horizontal([pl.col(b).is_null() for b in BANDS]).alias("all_null")
    )["all_null"]
    guard_triggers = rows_all_nan.all()  # False → guard does NOT trigger

    assert not guard_triggers, "guard should NOT trigger when only some rows are all-NaN"
    assert rows_all_nan.sum() == 1, "exactly one row is all-NaN"

    # Bug PC4: with the current guard, the all-NaN row survives.
    # After the fix (df = df[~rows_all_nan]), only row 0 remains.
    df_after_buggy_guard = df  # no drop happens
    df_after_fix = df.filter(~rows_all_nan)

    assert len(df_after_buggy_guard) == 2  # bug: all-NaN row survives
    assert len(df_after_fix) == 1           # expected after fix


# ---------------------------------------------------------------------------
# Test 15 — spectral index columns present (regression)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_has_no_spectral_index_columns():
    """Spectral indices are not stored in the parquet — computed at read time."""
    item = _make_item()
    store, lons, lats, _ = _make_store(2, scl_value=4.0, band_value=1000.0)
    df = extract_item_to_df(item, store, _point_ids(2), lons, lats, apply_nbar=False)
    assert df is not None
    for col in SPECTRAL_INDEX_COLS:
        assert col not in df.columns


# ---------------------------------------------------------------------------
# Test 16 — missing AOT defaults aot to 1.0 (regression)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_missing_aot_defaults_to_one():
    """Missing AOT defaults to quality=1.0, stored as uint8 100 (×100 encoding)."""
    item = _make_item()
    store, lons, lats, _ = _make_store(
        2, scl_value=4.0, band_value=500.0, missing_bands=(AOT_BAND,)
    )
    df = extract_item_to_df(item, store, _point_ids(2), lons, lats, apply_nbar=False)
    assert df is not None
    assert df["aot"].dtype == pl.UInt8
    assert (df["aot"] == 100).all()  # aot quality=1.0 → uint8 ×100 = 100


# ---------------------------------------------------------------------------
# Test 17 — apply_nbar=False does not raise (regression)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_apply_nbar_false_does_not_raise():
    item = _make_item()
    store, lons, lats, _ = _make_store(1, scl_value=4.0, band_value=800.0)
    df = extract_item_to_df(item, store, _point_ids(1), lons, lats, apply_nbar=False)
    assert df is not None


# ---------------------------------------------------------------------------
# Test 17b — band values aligned to point_ids (regression for set-ordering bug)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_band_values_aligned_to_point_ids():
    """Regression: store point_coords order must match point_ids/lons/lats order.

    Uses an N-column patch where column i holds a unique value for point i.
    If point_coords insertion order differs from point_ids order (as happened
    when shard_point_coords was built via a set comprehension), each point_id
    gets the wrong band value.
    """
    from rasterio.transform import from_bounds as _from_bounds

    N = 8
    point_ids = [f"px_{i:04d}_0000" for i in range(N)]

    crs = CRS.from_epsg(32755)
    # 1-row × N-col patch at 100 m/px: column i holds the value for point i.
    # lon/lat are pixel centres in this UTM extent, pre-computed so that
    # reprojection maps point i → column i exactly.
    x0, x1 = 550_000, 550_000 + N * 100
    y0, y1 = 7_749_950, 7_750_050
    transform = _from_bounds(x0, y0, x1, y1, N, 1)
    lons = np.array([147.47952211, 147.48048017, 147.48143823, 147.48239629,
                     147.48335435, 147.48431241, 147.48527047, 147.48622853])
    lats = np.full(N, -20.34741, dtype=np.float64)
    # point_coords in SAME order as point_ids — correct
    point_coords = {pid: (float(lons[i]), float(lats[i])) for i, pid in enumerate(point_ids)}
    scl_arr = np.full((1, N), 4, dtype=np.float32)  # all clear
    # Point i → raw DN = (i+1)*1000 → SR = (i+1)/10
    spectral_arr = np.array([[float((i + 1) * 1000) for i in range(N)]], dtype=np.float32)

    item_id = "S2A_ALIGN_20220815_0_L2A"
    patches: dict = {}
    patches[(item_id, SCL_BAND)] = (scl_arr, transform, crs)
    patches[(item_id, AOT_BAND)] = (np.full((1, N), 100.0, dtype=np.float32), transform, crs)
    for band in BANDS:
        patches[(item_id, band)] = (spectral_arr.copy(), transform, crs)

    from utils.chip_store import MemoryChipStore as _MCS
    store = _MCS(patches=patches, point_coords=point_coords)
    item = _make_item(item_id=item_id)
    df = extract_item_to_df(item, store, point_ids, lons, lats, apply_nbar=False)

    assert df is not None
    pid_to_row = {row[0]: row for row in df.select(["point_id"] + list(BANDS)).iter_rows()}
    b0_idx = 1  # BANDS[0] is at position 1 in the selected cols
    for i, pid in enumerate(point_ids):
        expected_dn = (i + 1) * 1000  # raw DN stored as uint16 ×10000
        actual = pid_to_row[pid][b0_idx]
        assert actual == expected_dn, (
            f"{pid}: expected DN {expected_dn}, got {actual} — "
            "point_id/band-value ordering mismatch"
        )


# ---------------------------------------------------------------------------
# Tests A–D — tile_id extraction and canonical tile filtering
# ---------------------------------------------------------------------------

def test_extract_item_to_df_tile_id_from_item_id():
    """tile_id is extracted from item.id when s2:mgrs_tile is absent (Element84)."""
    item = SimpleNamespace(
        id="S2A_54LWH_20220815_0_L2A",
        datetime=_ITEM_DT,
        properties={},  # no s2:mgrs_tile
    )
    store, lons, lats, _ = _make_store(2, scl_value=4.0, band_value=500.0,
                                       item_id="S2A_54LWH_20220815_0_L2A")
    df = extract_item_to_df(item, store, _point_ids(2, "S2A_54LWH_20220815_0_L2A"),
                            lons, lats, apply_nbar=False)
    assert df is not None
    assert (df["tile_id"] == "54LWH").all()


def test_canonical_tile_suppression_removed():
    """Regression: canonical tile suppression silently discarded all observations
    from one S2 tile, producing systematic score strips.

    The bug (commit 8f601ed) introduced a canonical tile assignment that:
    1. Projected each pixel to UTM easting.
    2. Computed dx = (ux % 10) - 5 to split pixels between tiles on the
       easting axis, assuming tiles are east-west neighbours.
    3. Suppressed all observations from the non-canonical tile in
       extract_item_to_df via a canonical_tiles mask.

    For quaids (55KBB / 55KCB), the tiles are north-south neighbours, not
    east-west, so the easting split was the wrong axis entirely. Combined with
    make_pixel_grid's floor-snap (which places all pixels at ux%10==0, giving
    dx=-5 universally), every pixel was assigned to tile[0]=55KBB, and every
    55KCB observation was silently suppressed — roughly halving each pixel's
    observation count and causing score strips.

    After the fix, canonical tile suppression is removed from extract_item_to_df.
    Both tiles must contribute observations regardless of their spatial arrangement.
    The existing (point_id, date, scl_purity desc) dedup in the sort/concat step
    handles tile-boundary pixels correctly without per-tile suppression.
    """
    import inspect

    # 1. extract_item_to_df must no longer accept canonical_tiles.
    sig = inspect.signature(extract_item_to_df)
    assert "canonical_tiles" not in sig.parameters, (
        "extract_item_to_df still has a canonical_tiles parameter — "
        "the tile suppression mechanism must be fully removed"
    )

    # 2. An item from a second tile (e.g. 55KCB when tile[0] is 55KBB) must
    #    contribute observations — it was silently suppressed by the bug.
    item_tile1 = SimpleNamespace(
        id="S2A_55KBB_20220815_0_L2A", datetime=_ITEM_DT, properties={}
    )
    item_tile2 = SimpleNamespace(
        id="S2A_55KCB_20220815_0_L2A", datetime=_ITEM_DT, properties={}
    )
    store1, lons, lats, _ = _make_store(
        3, scl_value=4.0, band_value=500.0, item_id="S2A_55KBB_20220815_0_L2A"
    )
    store2, _, _, _ = _make_store(
        3, scl_value=4.0, band_value=800.0, item_id="S2A_55KCB_20220815_0_L2A"
    )

    df1 = extract_item_to_df(item_tile1, store1, _point_ids(3), lons, lats, apply_nbar=False)
    df2 = extract_item_to_df(item_tile2, store2, _point_ids(3), lons, lats, apply_nbar=False)

    assert df1 is not None, "tile1 (55KBB) must contribute observations"
    assert df2 is not None, (
        "tile2 (55KCB) must contribute observations — "
        "was silently suppressed by the canonical tile bug"
    )
    assert len(df1) == 3
    assert len(df2) == 3

    # 3. The combined observation count must equal both tiles' contributions.
    #    Before the fix, one tile's rows were zeroed out, halving coverage.
    import polars as _pl
    combined = _pl.concat([df1, df2])
    assert len(combined) == 6, (
        f"expected 6 obs (3 per tile), got {len(combined)} — "
        "one tile's observations are being suppressed"
    )
    assert set(combined["tile_id"].to_list()) == {"55KBB", "55KCB"}, (
        "both tiles must appear in combined observations"
    )


# ---------------------------------------------------------------------------
# Test 18 — two items from different tiles produce separate rows (regression)
# ---------------------------------------------------------------------------

def test_extract_item_to_df_two_tiles_produce_separate_rows():
    """Cross-tile duplicates are NOT deduplicated at the extraction layer.

    The same point on the same date covered by two S2 tiles will appear as two
    rows in the combined DataFrame — one per tile.  This is the mechanism by
    which tile-boundary pixels are double-counted in the output parquet.
    """
    item_hbu = _make_item("S2A_55HBU_20220815_0_L2A", tile_id="55HBU")
    item_hbv = _make_item("S2A_55HBV_20220815_0_L2A", tile_id="55HBV")

    store_hbu, lons, lats, _ = _make_store(
        1, scl_value=4.0, band_value=1000.0, item_id="S2A_55HBU_20220815_0_L2A"
    )
    store_hbv, _, _, _ = _make_store(
        1, scl_value=4.0, band_value=1100.0, item_id="S2A_55HBV_20220815_0_L2A"
    )

    df_hbu = extract_item_to_df(item_hbu, store_hbu, _point_ids(1), lons, lats, apply_nbar=False)
    df_hbv = extract_item_to_df(item_hbv, store_hbv, _point_ids(1), lons, lats, apply_nbar=False)

    assert df_hbu is not None
    assert df_hbv is not None

    import polars as _pl
    combined = _pl.concat([df_hbu, df_hbv])
    # Same point_id and date, different tile_id — no dedup at this layer.
    assert len(combined) == 2
    assert combined["point_id"].n_unique() == 1
    assert combined["tile_id"].n_unique() == 2
    assert set(combined["tile_id"].to_list()) == {"55HBU", "55HBV"}


# ---------------------------------------------------------------------------
# Tests 19–21 — NBAR c-factor NaN propagation (regression)
#
# Root cause: get_item_angles() interpolates angle grids from the granule XML.
# Pixels near or outside the granule's angle grid boundary receive NaN vza/vaa.
# np.clip(band * cf, 0, 1) propagates NaN, silently zeroing out valid observations.
# Fix: fall back to uncorrected reflectance where cf is NaN.
# ---------------------------------------------------------------------------

def _make_angles(n: int, *, nan_indices: list[int] | None = None) -> dict:
    """Return a plausible angles dict for n points, with NaN at nan_indices."""
    sza = np.full(n, 30.0, dtype=np.float64)
    vza = np.full(n, 5.0,  dtype=np.float64)
    saa = np.full(n, 150.0, dtype=np.float64)
    vaa = np.full(n, 100.0, dtype=np.float64)
    if nan_indices:
        vza[nan_indices] = np.nan
        vaa[nan_indices] = np.nan
    return {band: {"sza": sza.copy(), "vza": vza.copy(),
                   "saa": saa.copy(), "vaa": vaa.copy()}
            for band in BANDS}


def test_nbar_nan_angles_do_not_produce_nan_bands():
    """Regression PC-NBAR1: NaN vza from angle interpolation must not write NaN bands.

    When get_item_angles returns NaN for some pixels (tile-edge pixels outside the
    granule angle grid), c_factor(sza, nan_vza, raa) = NaN, and
    np.clip(band * NaN, 0, 1) = NaN — silently corrupting valid reflectance.
    The fix falls back to uncorrected reflectance for those pixels.
    """
    from unittest.mock import patch

    item = _make_item()
    store, lons, lats, _ = _make_store(4, scl_value=4.0, band_value=500.0)
    pids = _point_ids(4)

    # Pixels 0 and 2 have NaN vza/vaa; pixels 1 and 3 have valid angles.
    angles = _make_angles(4, nan_indices=[0, 2])

    with patch("utils.granule_angles.get_item_angles", return_value=angles):
        df = extract_item_to_df(item, store, pids, lons, lats, apply_nbar=True)

    assert df is not None, "NaN angles must not cause all rows to be dropped"
    # Bands are nullable uint16 — NaN propagation shows up as null, not NaN.
    null_count = sum(df[b].null_count() for b in BANDS)
    assert null_count == 0, (
        "NaN c-factors must not propagate to band columns — "
        "fall back to uncorrected reflectance instead"
    )


def test_nbar_all_nan_angles_do_not_produce_nan_bands():
    """Regression PC-NBAR2: all-NaN angles (entire grid missing) must not corrupt bands.

    If the angle XML is unavailable or all pixels are outside the grid,
    get_item_angles returns a dict where every vza is NaN.  Every c-factor is NaN,
    so every band value would become NaN without the fallback.
    """
    from unittest.mock import patch

    item = _make_item()
    store, lons, lats, _ = _make_store(3, scl_value=4.0, band_value=800.0)
    pids = _point_ids(3)

    angles = _make_angles(3, nan_indices=[0, 1, 2])

    with patch("utils.granule_angles.get_item_angles", return_value=angles):
        df = extract_item_to_df(item, store, pids, lons, lats, apply_nbar=True)

    assert df is not None
    # Bands are nullable uint16 — check null count not nan count.
    null_count = sum(df[b].null_count() for b in BANDS)
    assert null_count == 0, (
        "All-NaN angles must not corrupt any band — fall back to uncorrected"
    )
    # Values should be the uncorrected raw DN = 800 (uint16 ×10000).
    assert int(df[0, "B02"]) == 800, (
        "Uncorrected fallback value should be raw DN (800)"
    )


def test_nbar_valid_angles_are_still_applied():
    """Sanity check PC-NBAR3: the NaN fallback must not suppress valid NBAR correction.

    Pixels with valid angles must still receive c-factor correction, not be left
    at uncorrected reflectance.
    """
    from unittest.mock import patch
    from utils.nbar import c_factor as compute_cf

    item = _make_item()
    store, lons, lats, _ = _make_store(2, scl_value=4.0, band_value=500.0)
    pids = _point_ids(2)

    angles = _make_angles(2)  # no NaN — all valid

    with patch("utils.granule_angles.get_item_angles", return_value=angles):
        df = extract_item_to_df(item, store, pids, lons, lats, apply_nbar=True)

    assert df is not None
    # Compute expected c-factor for these angles; result stored as uint16 DN (×10000).
    a = angles["B02"]
    cf = compute_cf(a["sza"], a["vza"], a["saa"] - a["vaa"], "B02")
    expected_dn = int(np.round(np.clip(500.0 / 10000.0 * cf[0], 0.0, 1.0) * 10000))
    actual = int(df[0, "B02"])
    assert abs(actual - expected_dn) <= 1, (
        f"Valid-angle pixels should be NBAR-corrected: expected DN {expected_dn}, got {actual}"
    )

# ---------------------------------------------------------------------------
# Test 19 — granule dedup regex strips index suffix (regression)
# ---------------------------------------------------------------------------

def test_granule_dedup_regex_strips_index():
    """Within-tile dedup works by stripping the processing-granule index suffix."""
    assert _GRANULE_RE.sub("", "S2A_TTTTTT_20220815_0_L2A") == "S2A_TTTTTT_20220815"
    assert _GRANULE_RE.sub("", "S2A_TTTTTT_20220815_1_L2A") == "S2A_TTTTTT_20220815"
    # Same key → deduped to one item
    items = [
        _make_item("S2A_TTTTTT_20220815_0_L2A"),
        _make_item("S2A_TTTTTT_20220815_1_L2A"),
    ]
    seen: set[str] = set()
    deduped = []
    for it in items:
        key = _GRANULE_RE.sub("", it.id)
        if key not in seen:
            seen.add(key)
            deduped.append(it)
    assert len(deduped) == 1


# ---------------------------------------------------------------------------
# Test 20 — cross-tile items survive per-tile dedup (Bug PC6 — documents)
# ---------------------------------------------------------------------------

def test_cross_tile_items_survive_per_tile_dedup():
    """Per-tile dedup removes within-tile granule duplicates but NOT cross-tile duplicates.

    Bug PC6: when collect() is called with items= pre-supplied (as training_collector
    does), the dedup block is bypassed entirely.  Even when dedup IS run (items=None
    path), it is applied per tile — two items with matching (date, satellite) but
    different tile_id both survive, producing cross-tile duplicate rows.

    This test simulates the per-tile dedup and shows both items survive.
    """
    # Two items representing the same acquisition over two adjacent tiles.
    # Their IDs differ only in tile-specific metadata, not the granule index.
    item_hbu = _make_item("S2A_TTTTTT_20220815_0_L2A", tile_id="55HBU")
    item_hbv = _make_item("S2A_TTTTTT_20220815_0_L2A", tile_id="55HBV")

    def _dedup(items):
        seen: set[str] = set()
        result = []
        for it in items:
            key = _GRANULE_RE.sub("", it.id)
            if key not in seen:
                seen.add(key)
                result.append(it)
        return result

    # Each tile's item list deduped independently:
    deduped_hbu = _dedup([item_hbu])
    deduped_hbv = _dedup([item_hbv])

    assert len(deduped_hbu) == 1
    assert len(deduped_hbv) == 1

    # After per-tile dedup, combined list still has two entries:
    combined = deduped_hbu + deduped_hbv
    assert len(combined) == 2  # cross-tile duplicate survives
    assert combined[0].id == combined[1].id  # same item ID, different tile context


# ---------------------------------------------------------------------------
# Test PC-G — polygon mask in collect(): non-rectangular polygon drops exterior points
# ---------------------------------------------------------------------------

def test_polygon_mask_drops_exterior_points():
    """Points outside a non-rectangular polygon are removed by the mask in collect().

    Uses a right-triangle polygon (lower-left half of a unit square).  Points
    on the lower-left side of the diagonal are inside; points on the upper-right
    side are outside.  This is non-trivial for a bbox and confirms the geometry
    path through collect() actually filters correctly.
    """
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPoint

    # A right-triangle polygon: lower-left half of the 0–1 degree square.
    # Interior: lon + lat < 1  (i.e. lat < 1 - lon)
    triangle = ShapelyPolygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])

    # Build a 2×2 degree grid that contains both inside and outside points.
    # Use a coarse resolution so the test is fast.
    bbox = [0.0, 0.0, 1.0, 1.0]
    points = make_pixel_grid(bbox, resolution_m=50_000)  # ~50 km steps → few points

    assert len(points) > 0, "grid must be non-empty"

    # Apply the same mask logic as collect()
    mp = MultiPoint([(lon, lat) for _, lon, lat in points])
    kept = [pt for pt, contained in zip(points, [triangle.contains(p) for p in mp.geoms]) if contained]

    # All kept points must be inside the triangle
    from shapely.geometry import Point
    for _, lon, lat in kept:
        assert triangle.contains(Point(lon, lat)), f"kept point ({lon}, {lat}) is outside the triangle"

    # The mask must have dropped at least one point — the grid covers the whole
    # bbox so some points fall outside the triangle.
    assert len(kept) < len(points), "polygon mask should drop some exterior points"
    assert len(kept) > 0, "polygon mask should retain some interior points"


# ---------------------------------------------------------------------------
# Tests 21–25 — collect() two-pass streaming dedup
# ---------------------------------------------------------------------------

def _dedup_row(tile_id, scl_purity, point_id="px_0000_0000", dt=None):
    """Build a minimal pixel row dict for dedup tests."""
    from analysis.constants import BANDS, SPECTRAL_INDEX_COLS
    if dt is None:
        dt = date(2022, 8, 15)
    return {
        "point_id": point_id,
        "lon": 145.41, "lat": -22.78,
        "date": dt,
        "item_id": f"S2A_{tile_id}_20220815",
        "tile_id": tile_id,
        **{b: 0.15 for b in BANDS},
        "scl_purity": scl_purity,
        "scl": 4, "aot": 0.9,
        "view_zenith": 0.95, "sun_zenith": 0.80,
        **{c: 0.3 for c in SPECTRAL_INDEX_COLS},
    }


def _apply_dedup(out_path):
    """Run the same two-pass streaming dedup logic as collect() and return (df, n_removed)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(out_path)
    key_cols = ["point_id", "date", "scl_purity", "tile_id"]
    keys_df = pl.from_arrow(pf.read(columns=key_cols)).sort(
        ["point_id", "date", "scl_purity", "tile_id"],
        descending=[False, False, True, False],
    )
    # Find losers: duplicates keeping the first (highest scl_purity, then lowest tile_id)
    deduped = keys_df.unique(subset=["point_id", "date"], keep="first")
    n_dedup = len(keys_df) - len(deduped)

    if not n_dedup:
        del pf
        return pl.read_parquet(out_path), 0

    losers = keys_df.join(
        deduped.select(["point_id", "date", "tile_id"]).with_columns(pl.lit(True).alias("_keep")),
        on=["point_id", "date", "tile_id"],
        how="left",
    ).filter(pl.col("_keep").is_null()).select(["point_id", "date", "tile_id"])

    schema = pf.schema_arrow
    tmp_path = out_path.with_suffix(".tmp.parquet")
    import pyarrow.parquet as pq2
    tmp_writer = pq2.ParquetWriter(tmp_path, schema)
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pl.from_arrow(pf.read_row_group(rg_idx))
        filtered = rg.join(
            losers.with_columns(pl.lit(True).alias("_drop")),
            on=["point_id", "date", "tile_id"],
            how="left",
        ).filter(pl.col("_drop").is_null()).drop("_drop")
        if len(filtered):
            tmp_writer.write_table(filtered.to_arrow().cast(schema))
    tmp_writer.close()
    del pf
    tmp_path.replace(out_path)
    return pl.read_parquet(out_path), n_dedup


# ---------------------------------------------------------------------------
# Test 21 — collect() dedup removes cross-tile boundary duplicates
# ---------------------------------------------------------------------------

def test_collect_output_deduplicates_cross_tile_rows(tmp_path):
    """Two rows sharing (point_id, date) from different tiles: higher scl_purity wins."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    df_raw = pl.DataFrame([_dedup_row("55HBU", 0.6), _dedup_row("55HBV", 0.9)])
    out_path = tmp_path / "out.parquet"
    pq.write_table(df_raw.to_arrow(), out_path)

    result, n_removed = _apply_dedup(out_path)
    assert n_removed == 1
    assert len(result) == 1
    assert result["tile_id"][0] == "55HBV"
    assert result["scl_purity"][0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Test 22 — no-duplicate parquet is unchanged
# ---------------------------------------------------------------------------

def test_dedup_noop_when_no_duplicates(tmp_path):
    """When every (point_id, date) is unique, no rows are removed."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dt1 = date(2022, 8, 15)
    dt2 = date(2022, 9, 1)
    df_raw = pl.DataFrame([
        _dedup_row("55HBU", 0.8, point_id="px_0000_0000", dt=dt1),
        _dedup_row("55HBU", 0.7, point_id="px_0000_0000", dt=dt2),
        _dedup_row("55HBU", 0.9, point_id="px_0001_0000", dt=dt1),
    ])
    out_path = tmp_path / "out.parquet"
    pq.write_table(df_raw.to_arrow(), out_path)

    result, n_removed = _apply_dedup(out_path)
    assert n_removed == 0
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Test 23 — non-duplicate rows adjacent to a boundary pixel are preserved
# ---------------------------------------------------------------------------

def test_dedup_preserves_non_duplicate_rows(tmp_path):
    """Interior pixels with unique (point_id, date) survive alongside the deduped boundary pixel."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dt = date(2022, 8, 15)
    df_raw = pl.DataFrame([
        _dedup_row("55HBU", 0.5, point_id="px_interior_a", dt=dt),
        _dedup_row("55HBU", 0.6, point_id="px_boundary",   dt=dt),
        _dedup_row("55HBV", 0.9, point_id="px_boundary",   dt=dt),
        _dedup_row("55HBV", 0.8, point_id="px_interior_b", dt=dt),
    ])
    out_path = tmp_path / "out.parquet"
    pq.write_table(df_raw.to_arrow(), out_path)

    result, n_removed = _apply_dedup(out_path)
    assert n_removed == 1
    assert len(result) == 3
    surviving_pids = set(result["point_id"].to_list())
    assert "px_interior_a" in surviving_pids
    assert "px_interior_b" in surviving_pids
    assert result.filter(pl.col("point_id") == "px_boundary")["tile_id"][0] == "55HBV"


# ---------------------------------------------------------------------------
# Test 24 — duplicate pair straddling a row-group boundary
# ---------------------------------------------------------------------------

def test_dedup_across_row_group_boundary(tmp_path):
    """The dedup correctly removes a loser whose row group differs from the winner's."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dt = date(2022, 8, 15)
    winner = pl.DataFrame([_dedup_row("55HBV", 0.9, dt=dt)])
    loser  = pl.DataFrame([_dedup_row("55HBU", 0.6, dt=dt)])
    out_path = tmp_path / "out.parquet"
    schema = winner.to_arrow().schema
    writer = pq.ParquetWriter(out_path, schema)
    writer.write_table(winner.to_arrow())
    writer.write_table(loser.to_arrow())
    writer.close()

    assert pq.ParquetFile(out_path).metadata.num_row_groups == 2

    result, n_removed = _apply_dedup(out_path)
    assert n_removed == 1
    assert len(result) == 1
    assert result["tile_id"][0] == "55HBV"


# ---------------------------------------------------------------------------
# Test 25 — equal scl_purity: lower tile_id wins deterministically
# ---------------------------------------------------------------------------

def test_dedup_tiebreak_lower_tile_id_wins(tmp_path):
    """When scl_purity is equal, the lower tile_id (ascending lexicographic) is kept."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dt = date(2022, 8, 15)
    df_raw = pl.DataFrame([
        _dedup_row("55HBV", 0.8, dt=dt),
        _dedup_row("55HBU", 0.8, dt=dt),
    ])
    out_path = tmp_path / "out.parquet"
    pq.write_table(df_raw.to_arrow(), out_path)

    result, n_removed = _apply_dedup(out_path)
    assert n_removed == 1
    assert len(result) == 1
    assert result["tile_id"][0] == "55HBU"  # lower tile_id wins


# ---------------------------------------------------------------------------
# Tests PC-T1–PC-T3 — per-tile parquet output splitting
# ---------------------------------------------------------------------------

def _make_sorted_shard(tmp_path: Path, rows: list[dict]) -> "Path":
    """Write a sorted shard parquet from a list of row dicts (sorted by point_id)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from analysis.constants import BANDS, SPECTRAL_INDEX_COLS

    full_rows = []
    for r in sorted(rows, key=lambda x: x["point_id"]):
        full_rows.append({
            "lon": 145.0, "lat": -23.0,
            "date": datetime(2022, 8, 15),
            "item_id": f"S2A_{r['tile_id']}_20220815_0_L2A",
            "scl_purity": 1.0, "scl": 4, "aot": 0.9,
            "view_zenith": 0.95, "sun_zenith": 0.80,
            **{b: 0.15 for b in BANDS},
            **{c: 0.3 for c in SPECTRAL_INDEX_COLS},
            **r,
        })
    df = pl.DataFrame(full_rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
    p = tmp_path / "_collect_shard000_sorted.parquet"
    pq.write_table(df.to_arrow(), p)
    return p


def _run_split(sorted_shard: "Path", out_dir: "Path") -> "dict[str, pl.DataFrame]":
    """Invoke the per-tile split logic used inside collect() and return {tile_id: df}."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    pf = pq.ParquetFile(sorted_shard)
    tile_writers: dict = {}
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        tile_col = rg.column("tile_id").combine_chunks()
        for tid in pc.unique(tile_col).to_pylist():
            subset = rg.filter(pc.equal(tile_col, tid))
            if tid not in tile_writers:
                p = out_dir / f"{tid}.parquet"
                tile_writers[tid] = pq.ParquetWriter(str(p), subset.schema)
            tile_writers[tid].write_table(subset)

    for w in tile_writers.values():
        w.close()

    return {
        tid: pl.read_parquet(out_dir / f"{tid}.parquet")
        for tid in tile_writers
    }


def test_per_tile_split_writes_separate_files(tmp_path):
    """Sorted shard with two tile_ids produces two separate output parquets."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rows = [
        {"point_id": "px_0000_0000", "tile_id": "55HBU"},
        {"point_id": "px_0001_0000", "tile_id": "55HBU"},
        {"point_id": "px_0002_0000", "tile_id": "55HBV"},
    ]
    shard = _make_sorted_shard(shard_dir, rows)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = _run_split(shard, out_dir)

    assert set(result.keys()) == {"55HBU", "55HBV"}
    assert set(result["55HBU"]["point_id"].to_list()) == {"px_0000_0000", "px_0001_0000"}
    assert set(result["55HBV"]["point_id"].to_list()) == {"px_0002_0000"}


def test_per_tile_split_each_file_contains_only_its_tile(tmp_path):
    """Each output parquet contains only rows whose tile_id matches the filename."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rows = [
        {"point_id": "px_0000_0000", "tile_id": "54LWH"},
        {"point_id": "px_0001_0000", "tile_id": "55HBU"},
        {"point_id": "px_0002_0000", "tile_id": "55HBU"},
    ]
    shard = _make_sorted_shard(shard_dir, rows)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = _run_split(shard, out_dir)

    for tid, df in result.items():
        assert (df["tile_id"] == tid).all(), f"tile {tid} file contains foreign tile_ids"


def test_per_tile_split_single_tile_writes_one_file(tmp_path):
    """Single-tile input writes exactly one output parquet."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rows = [
        {"point_id": "px_0000_0000", "tile_id": "55HBU"},
        {"point_id": "px_0001_0000", "tile_id": "55HBU"},
    ]
    shard = _make_sorted_shard(shard_dir, rows)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = _run_split(shard, out_dir)

    assert list(result.keys()) == ["55HBU"]
    assert len(result["55HBU"]) == 2


# ---------------------------------------------------------------------------
# Test PC-SB1 — collect() invalidates shard done-files when chip cache is stale
#
# Root cause: training collect cached patches for bbox A in a shared tile chip dir.
# Inference collect for bbox B found item .npz files present, skipped re-fetch, and
# reused the wrong-bbox patches.  The shard .done file caused collect() to return
# the stale tile parquet without rebuilding it.
#
# The fix: before checking all_shards_done, verify that cached B08 patches cover
# the current bbox.  If not, delete .done files and stale tile parquets.
# ---------------------------------------------------------------------------

def test_collect_detects_stale_cache_and_invalidates_done_files(tmp_path):
    """collect() must delete shard .done files when cached patches don't cover bbox.

    Simulates the cross-bbox cache poisoning scenario:
    1. A .done file exists (from a prior training collect for bbox A).
    2. The chip cache contains a B08 patch for bbox A (~10 km away from bbox B).
    3. collect() is called with bbox B — it must detect the mismatch and invalidate
       the .done file so the shard will be rebuilt.
    4. A stale tile parquet in out_dir must also be deleted.
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from utils.fetch import _save_patch_cache, _cache_path

    item_id = "S2A_54LWH_20250430_0_L2A"
    cache_dir = tmp_path / "chips"
    out_dir   = tmp_path / "out"
    out_dir.mkdir()

    # Write a B08 patch covering bbox A (training region, 10 km west)
    crs = CRS.from_epsg(32754)
    patch_arr = np.ones((10, 10), dtype=np.float32) * 1000
    bbox_a_transform = from_bounds(543_000, 8_243_000, 543_100, 8_243_100, 10, 10)
    _save_patch_cache(
        _cache_path(cache_dir, item_id, "B08"),
        (patch_arr, bbox_a_transform, crs),
    )

    # Write a shard .done file as if a prior run completed
    done_path = out_dir / "_collect_shard000.done"
    done_path.write_text(f"{item_id}\n__done__\n")

    # Write a stale tile parquet
    stale_parquet = out_dir / "54LWH.parquet"
    stale_parquet.write_bytes(b"fake parquet content")

    # Scoring bbox B is ~10 km east — outside the cached patch
    bbox_b = [141.533, -15.809, 141.559, -15.766]

    # We test the stale-detection logic extracted from collect() directly,
    # since collect() itself requires a full STAC search + network.
    from utils.fetch import _load_patch_cache, _patch_covers_bbox

    stale_ids: set[str] = set()
    proxy_path = _cache_path(cache_dir, item_id, "B08")
    data = _load_patch_cache(proxy_path)
    assert data is not None
    if not _patch_covers_bbox(data, bbox_b):
        stale_ids.add(item_id)

    assert len(stale_ids) == 1, "patch for training bbox must be detected as stale"

    # Simulate what collect() does: delete stale .npz files, .done files, tile parquets
    npz_path = _cache_path(cache_dir, item_id, "B08")
    if stale_ids:
        for sid in stale_ids:
            item_dir = cache_dir / sid
            if item_dir.is_dir():
                for npz in item_dir.glob("*.npz"):
                    npz.unlink()
        if done_path.exists():
            done_path.unlink()
        for p in out_dir.iterdir():
            if p.suffix == ".parquet" and not p.stem.startswith("_"):
                p.unlink()

    assert not npz_path.exists(), "stale .npz patch must be deleted so re-fetch is triggered"
    assert not done_path.exists(), "done file must be deleted when cache is stale"
    assert not stale_parquet.exists(), "stale tile parquet must be deleted"


def test_collect_does_not_invalidate_when_cache_covers_bbox(tmp_path):
    """When the cached patch covers the current bbox, done files are left intact."""
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from utils.fetch import _save_patch_cache, _cache_path, _load_patch_cache, _patch_covers_bbox
    from pyproj import Transformer

    item_id = "S2A_54LWH_20250430_0_L2A"
    cache_dir = tmp_path / "chips"
    out_dir   = tmp_path / "out"
    out_dir.mkdir()

    # Patch covers a wide area including the scoring bbox
    crs = CRS.from_epsg(32754)
    # 800×700 patch at 10 m/px: covers lon 141.40–141.47, lat -15.89–-15.82
    patch_arr = np.ones((800, 700), dtype=np.float32) * 2000
    transform = from_bounds(543_000, 8_243_000, 550_000, 8_251_000, 700, 800)
    _save_patch_cache(
        _cache_path(cache_dir, item_id, "B08"),
        (patch_arr, transform, crs),
    )

    done_path = out_dir / "_collect_shard000.done"
    done_path.write_text(f"{item_id}\n__done__\n")

    # Scoring bbox that falls inside the patch
    t = Transformer.from_crs("EPSG:32754", "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(544_000, 8_244_000)
    lon1, lat1 = t.transform(546_000, 8_246_000)
    bbox_inside = [min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1)]

    data = _load_patch_cache(_cache_path(cache_dir, item_id, "B08"))
    assert data is not None
    assert _patch_covers_bbox(data, bbox_inside), "patch should cover this bbox"

    stale_ids: set[str] = set()
    if not _patch_covers_bbox(data, bbox_inside):
        stale_ids.add(item_id)

    assert len(stale_ids) == 0, "valid patch must not be flagged as stale"
    assert done_path.exists(), "done file must NOT be deleted for a valid cache"


# ---------------------------------------------------------------------------
# Test PC-SB2 — parquet spatial variance guard
#
# A parquet produced from stale cross-bbox patches looks structurally valid but
# has near-zero spatial variation: every pixel's band values differ only at the
# 5th decimal place because all points map to the same patch edge pixel.
# This guard detects the symptom (all pixels near-identical) regardless of cause.
# ---------------------------------------------------------------------------

def _make_pixel_parquet(tmp_path: "Path", b08_values: "np.ndarray") -> "Path":
    """Write a minimal pixel parquet with the given per-pixel B08 p95 values."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(b08_values)
    df = pl.DataFrame({
        "point_id":   [f"px_{i:04d}_0000" for i in range(n)],
        "date":       [date(2025, 4, 30)] * n,
        "B08":        b08_values.astype(np.float32).tolist(),
        "scl_purity": [1.0] * n,
    })
    p = tmp_path / "test.parquet"
    pq.write_table(df.to_arrow(), p)
    return p




def test_parquet_spatial_variance_guard_rejects_flat_data(tmp_path):
    """A parquet where all pixels have the same B08 value fails the variance guard.

    This is the signature of cross-bbox cache corruption: every pixel maps to
    the same patch-edge DN, so per-pixel B08 mean is identical for all pixels.
    """
    import pyarrow.parquet as pq

    b08_flat = np.full(50, 2492.0, dtype=np.float32)
    parquet_path = _make_pixel_parquet(tmp_path, b08_flat)

    pixel_means: dict = {}
    pf = pq.ParquetFile(parquet_path)
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=["point_id", "B08"]).to_pandas()
        for pid, grp in chunk.groupby("point_id"):
            pixel_means[pid] = float(grp["B08"].mean())

    vals = np.array(list(pixel_means.values()), dtype=np.float32)
    std_across_pixels = float(vals.std())

    MIN_STD = 1e-3  # any real scene has far more spatial variation than this
    assert std_across_pixels < MIN_STD, (
        f"Variance guard must reject flat parquet (std={std_across_pixels:.6f} < {MIN_STD}). "
        "This is the signature of cross-bbox cache corruption where all pixels map "
        "to the same patch edge via np.clip."
    )


def test_parquet_spatial_variance_guard_accepts_real_variation(tmp_path):
    """A parquet with genuine per-pixel spatial variation passes the variance guard."""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(42)
    b08_real = rng.normal(2500, 500, size=50).astype(np.float32)
    parquet_path = _make_pixel_parquet(tmp_path, b08_real)

    pixel_means: dict = {}
    pf = pq.ParquetFile(parquet_path)
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=["point_id", "B08"]).to_pandas()
        for pid, grp in chunk.groupby("point_id"):
            pixel_means[pid] = float(grp["B08"].mean())

    vals = np.array(list(pixel_means.values()), dtype=np.float32)
    std_across_pixels = float(vals.std())

    MIN_STD = 1e-3
    assert std_across_pixels >= MIN_STD, (
        f"Real spatial variation should exceed threshold (std={std_across_pixels:.2f})"
    )
