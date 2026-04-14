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
        # Pixel coords are computed lazily per (item_id, band) on first access.
        # Pre-computing all (item_id, band, point_id) triples upfront would
        # require O(patches × points) memory — ~200M entries for a 40k-pixel bbox.
        # (item_id, band) → (rows: int array, cols: int array) in point_coords order
        self._pixel_coords: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        # point_id → position index for O(1) single-point lookup
        self._point_index: dict[str, int] = {pid: i for i, pid in enumerate(point_coords.keys())}
        # Precomputed lon/lat arrays in point_coords order for vectorised projection
        _pids = list(point_coords.keys())
        self._lons = np.array([point_coords[pid][0] for pid in _pids])
        self._lats = np.array([point_coords[pid][1] for pid in _pids])
        # Cache one Transformer per CRS string to avoid repeated construction.
        self._crs_transformer: dict[str, Transformer] = {}
        self._lock = threading.Lock()

    def _ensure_pixel_coords(self, item_id: str, band: str) -> None:
        """Compute and cache (row, col) for all points for this (item_id, band)."""
        key = (item_id, band)
        with self._lock:
            if key in self._pixel_coords:
                return
            patch, transform, crs = self._patches[key]
            h, w = patch.shape
            crs_key = crs.to_string() if hasattr(crs, "to_string") else str(crs)
            t = self._crs_transformer.get(crs_key)

        # Compute transformer outside the lock if needed (expensive construction)
        if t is None:
            t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            with self._lock:
                self._crs_transformer.setdefault(crs_key, t)
                t = self._crs_transformer[crs_key]

        # Vectorised projection — CPU work done outside the lock
        xs, ys = t.transform(self._lons, self._lats)
        cols_f, rows_f = ~transform * (xs, ys)
        rows = np.clip(rows_f.astype(np.intp), 0, h - 1)
        cols = np.clip(cols_f.astype(np.intp), 0, w - 1)

        with self._lock:
            # Another thread may have computed this while we were working — that's fine
            self._pixel_coords.setdefault(key, (rows, cols))

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
            rows, cols = self._pixel_coords[key]
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
            rows, cols = self._pixel_coords[key]
        patch, _, _ = self._patches[key]
        return patch[rows, cols]

    def release_item(self, item_id: str) -> None:
        """Evict all cached pixel-coord dicts for item_id.

        Call this after processing each item in a streaming loop to keep
        memory flat — without it, coord dicts accumulate for every (item, band)
        pair processed so far.
        """
        with self._lock:
            to_delete = [k for k in self._pixel_coords if k[0] == item_id]
            for k in to_delete:
                del self._pixel_coords[k]


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
                arr = src.read(1)  # single-band chip → 2-D array
        return arr
