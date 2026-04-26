"""utils/chip_store.py — ChipStore protocol and implementations.

ChipStore is the interface used by extraction and composite code to retrieve
pixel values without caring about how or where the data is stored.

Implementations
---------------
MemoryChipStore : in-memory store populated from fetch_patches(). Used by
                  the pixel_collector pipeline for dense point grids.
DiskChipStore   : reads from inputs/ populated by a Stage 0 fetch run.
                  Layout: {inputs_dir}/{item_id}/{band}_{point_id}.tif
"""

from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning


class ChipStore(Protocol):
    """Protocol satisfied by any chip storage backend."""

    def get(self, item_id: str, band: str, point_id: str) -> np.ndarray:
        """Return the chip array for (item_id, band, point_id).

        Returns a 2-D numpy array of shape (rows, cols).
        Raises FileNotFoundError if the chip does not exist.
        """
        ...


class MemoryChipStore:
    """In-memory ChipStore populated from patch data returned by fetch_patches().

    Satisfies the ChipStore Protocol. Each .get() call reprojects the point's
    (lon, lat) into the patch CRS, computes the pixel (row, col) in the patch
    array, and returns a 1×1 ndarray containing that pixel's value.

    This is the efficient counterpart to DiskChipStore for dense point grids
    within a small bbox — no disk I/O, no per-point chip files.

    Parameters
    ----------
    patches:
        Mapping (item_id, band) → (2D float32 array, Affine transform, rasterio CRS).
        Produced by fetch_patches(). Cloud-filtered items and missing bands are
        absent from this dict; .get() raises FileNotFoundError for absent keys.
    point_coords:
        Mapping point_id → (lon, lat) in EPSG:4326.
    """

    def __init__(
        self,
        patches: dict[tuple[str, str], tuple[np.ndarray, object, object]],
        point_coords: dict[str, tuple[float, float]],
    ) -> None:
        self._patches = patches
        self._point_coords = point_coords
        # point_id → position index for O(1) single-point lookup
        self._point_index: dict[str, int] = {pid: i for i, pid in enumerate(point_coords.keys())}
        # Precomputed lon/lat arrays in point_coords order for vectorised projection
        _pids = list(point_coords.keys())
        self._lons = np.array([point_coords[pid][0] for pid in _pids])
        self._lats = np.array([point_coords[pid][1] for pid in _pids])
        # Cache one Transformer per CRS string to avoid repeated construction.
        self._crs_transformer: dict[str, Transformer] = {}
        # Pixel coords keyed by (crs_key, transform_tuple, h, w) — shared across all
        # items that share the same CRS and affine transform (i.e. the same UTM tile
        # and resolution).  This avoids reprojecting N points once per (item, band);
        # instead each unique (crs, transform) pair is projected exactly once.
        self._proj_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        # (item_id, band) → proj_cache_key — so get() / get_all_points() can look up
        self._item_band_proj_key: dict[tuple[str, str], tuple] = {}
        self._lock = threading.Lock()

    def _ensure_pixel_coords(self, item_id: str, band: str) -> None:
        """Compute and cache (row, col) for all points, keyed by (crs, transform).

        All S2 items over the same UTM tile share the same CRS and affine
        transform, so the reprojection only happens once per unique transform
        rather than once per (item_id, band) pair.
        """
        ib_key = (item_id, band)
        with self._lock:
            if ib_key in self._item_band_proj_key:
                return
            patch, transform, crs = self._patches[ib_key]
            h, w = patch.shape
            crs_key = crs.to_string() if hasattr(crs, "to_string") else str(crs)
            proj_key = (crs_key, tuple(transform), h, w)
            already_cached = proj_key in self._proj_cache
            t = self._crs_transformer.get(crs_key)

        if not already_cached:
            # Compute transformer outside the lock if needed
            if t is None:
                t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                with self._lock:
                    self._crs_transformer.setdefault(crs_key, t)
                    t = self._crs_transformer[crs_key]

            # Vectorised projection — CPU work done outside the lock
            xs, ys = t.transform(self._lons, self._lats)
            cols_f, rows_f = ~transform * (xs, ys)
            rows = np.clip(np.floor(rows_f).astype(np.intp), 0, h - 1)
            cols = np.clip(np.floor(cols_f).astype(np.intp), 0, w - 1)

            with self._lock:
                self._proj_cache.setdefault(proj_key, (rows, cols))

        with self._lock:
            self._item_band_proj_key.setdefault(ib_key, proj_key)

    def get(self, item_id: str, band: str, point_id: str) -> np.ndarray:
        """Return a 1×1 array containing the pixel value at (item_id, band, point_id).

        Raises
        ------
        FileNotFoundError
            If the (item_id, band) patch is absent — cloud-filtered item or
            band not present in the STAC item. extract_observations() catches
            FileNotFoundError for optional bands and uses a neutral default.
        """
        key = (item_id, band)
        if key not in self._patches:
            raise FileNotFoundError(
                f"Patch not found: item_id={item_id!r}, band={band!r}"
            )
        self._ensure_pixel_coords(item_id, band)
        idx = self._point_index[point_id]
        with self._lock:
            proj_key = self._item_band_proj_key[key]
            rows, cols = self._proj_cache[proj_key]
        patch, _, _ = self._patches[key]
        r, c = int(rows[idx]), int(cols[idx])
        return patch[r : r + 1, c : c + 1]

    def get_all_points(self, item_id: str, band: str) -> np.ndarray | None:
        """Return a 1-D float32 array of pixel values for all points, in point_coords order.

        Returns None if the (item_id, band) patch is absent (cloud-filtered or
        missing band). Values are raw patch values — callers apply scaling.
        """
        key = (item_id, band)
        if key not in self._patches:
            return None
        self._ensure_pixel_coords(item_id, band)
        with self._lock:
            proj_key = self._item_band_proj_key[key]
            rows, cols = self._proj_cache[proj_key]
        patch, _, _ = self._patches[key]
        return patch[rows, cols]

    def release_item(self, item_id: str) -> None:
        """Evict item-band lookup entries for item_id to keep memory flat.

        The shared proj_cache entries (keyed by crs+transform) are retained —
        they are small (two int arrays per unique UTM tile) and will be reused
        by subsequent items on the same tile.
        """
        with self._lock:
            to_delete = [k for k in self._item_band_proj_key if k[0] == item_id]
            for k in to_delete:
                del self._item_band_proj_key[k]


class CachedNpzChipStore:
    """ChipStore backed by the .npz patch cache written by fetch_patches().

    Loads each item's band arrays on demand when first accessed, then evicts
    them on release_item() — keeping at most one item's worth of patches in RAM
    at any time instead of the full collection.

    Parameters
    ----------
    cache_dir : Path
        Directory written by fetch_patches() with layout
        ``{cache_dir}/{item_id}/{band}.npz``.
    point_coords : dict[str, tuple[float, float]]
        Mapping point_id → (lon, lat) in EPSG:4326.
    bands : list[str]
        Canonical band names to load (e.g. FETCH_BANDS).
    """

    def __init__(
        self,
        cache_dir: Path,
        point_coords: dict[str, tuple[float, float]],
        bands: list[str],
    ) -> None:
        self._cache_dir = cache_dir
        self._bands = bands
        self._point_coords = point_coords
        _pids = list(point_coords.keys())
        self._lons = np.array([point_coords[pid][0] for pid in _pids])
        self._lats = np.array([point_coords[pid][1] for pid in _pids])
        # Live patches for the current item: band → (arr, transform, crs)
        self._live: dict[str, tuple[np.ndarray, object, object]] = {}
        self._live_item: str | None = None
        # Reuse Transformer instances across items that share a CRS
        self._transformers: dict[str, Transformer] = {}
        # Pixel coords cache keyed by (crs_key, transform_tuple, h, w)
        self._proj_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _load_item(self, item_id: str) -> None:
        if self._live_item == item_id:
            return
        # Evict previous item
        self._live.clear()
        self._live_item = item_id
        from utils.fetch import _cache_path, _load_patch_cache
        for band in self._bands:
            path = _cache_path(self._cache_dir, item_id, band)
            data = _load_patch_cache(path)
            if data is not None:
                self._live[band] = data

    def _pixel_coords(self, band: str) -> tuple[np.ndarray, np.ndarray] | None:
        if band not in self._live:
            return None
        arr, transform, crs = self._live[band]
        h, w = arr.shape
        crs_key = crs.to_string() if hasattr(crs, "to_string") else str(crs)
        proj_key = (crs_key, tuple(transform), h, w)
        if proj_key not in self._proj_cache:
            t = self._transformers.get(crs_key)
            if t is None:
                t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                self._transformers[crs_key] = t
            xs, ys = t.transform(self._lons, self._lats)
            cols_f, rows_f = ~transform * (xs, ys)
            rows = np.clip(np.floor(rows_f).astype(np.intp), 0, h - 1)
            cols = np.clip(np.floor(cols_f).astype(np.intp), 0, w - 1)
            self._proj_cache[proj_key] = (rows, cols)
        return self._proj_cache[proj_key]

    def get_all_points(self, item_id: str, band: str) -> np.ndarray | None:
        """Return a 1-D float32 array of pixel values for all points, or None if missing."""
        self._load_item(item_id)
        coords = self._pixel_coords(band)
        if coords is None:
            return None
        rows, cols = coords
        arr, _, _ = self._live[band]
        return arr[rows, cols]

    def release_item(self, item_id: str) -> None:
        """Evict the live item from memory."""
        if self._live_item == item_id:
            self._live.clear()
            self._live_item = None


class DiskChipStore:
    """Reads chips from the inputs/ directory populated by a Stage 0 fetch run.

    Expected layout::

        {inputs_dir}/
          {item_id}/
            {band}_{point_id}.tif

    Parameters
    ----------
    inputs_dir:
        Root directory containing staged chips. Defaults to ``inputs/``
        relative to the working directory.
    """

    def __init__(self, inputs_dir: Path | str = Path("inputs/")) -> None:
        self.inputs_dir = Path(inputs_dir)

    def _chip_path(self, item_id: str, band: str, point_id: str) -> Path:
        return self.inputs_dir / item_id / f"{band}_{point_id}.tif"

    def get(self, item_id: str, band: str, point_id: str) -> np.ndarray:
        """Read and return the chip array.

        Returns a 2-D numpy array squeezed from the single-band GeoTIFF.

        Raises
        ------
        FileNotFoundError
            If the chip file does not exist, with the full expected path
            included in the message so callers can diagnose missing Stage 0
            runs without inspecting the directory manually.
        """
        path = self._chip_path(item_id, band, point_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Chip not found: {path}\n"
                f"  item_id={item_id!r}, band={band!r}, point_id={point_id!r}\n"
                "  Has Stage 0 fetch been run for this configuration?"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as src:
                return src.read(1)
