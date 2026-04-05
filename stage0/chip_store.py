"""ChipStore abstraction: decouples extraction code from chip storage strategy.

Downstream stages (extraction, composite) call chip_store.get() and never
read files directly. This keeps them identical whether data was staged to
disk by Stage 0 or must be fetched on demand.

Implementations
---------------
DiskChipStore      : reads from inputs/ populated by Stage 0 fetch.  Default.
StreamingChipStore : issues COG range requests on demand, no disk required.
                     Deferred — interface defined here, implementation pending
                     until scale requires it (e.g. national-scale inference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import warnings

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


class DiskChipStore:
    """Reads chips from the inputs/ directory populated by Stage 0 fetch.

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


# ---------------------------------------------------------------------------
# MemoryChipStore — in-memory store populated from fetch_patches()
# ---------------------------------------------------------------------------

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
        # Pre-compute (row, col) for every (item_id, band, point_id) combination.
        # Bands within an item can have different resolutions (e.g. AOT at 60m
        # vs spectral bands at 10m), so each (item_id, band) patch has its own
        # transform and shape and needs its own coordinate projection.
        self._pixel_coords: dict[tuple[str, str, str], tuple[int, int]] = {}

        crs_transformer: dict[str, Transformer] = {}
        for (item_id, band), (arr, transform, crs) in patches.items():
            crs_key = crs.to_string() if hasattr(crs, "to_string") else str(crs)
            if crs_key not in crs_transformer:
                crs_transformer[crs_key] = Transformer.from_crs(
                    "EPSG:4326", crs, always_xy=True
                )
            t = crs_transformer[crs_key]
            inv_transform = ~transform
            h, w = arr.shape
            for point_id, (lon, lat) in point_coords.items():
                x_utm, y_utm = t.transform(lon, lat)
                col_f, row_f = inv_transform * (x_utm, y_utm)
                row = max(0, min(int(row_f), h - 1))
                col = max(0, min(int(col_f), w - 1))
                self._pixel_coords[(item_id, band, point_id)] = (row, col)

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
        patch, _, _ = self._patches[key]
        row, col = self._pixel_coords[(item_id, band, point_id)]
        return patch[row : row + 1, col : col + 1]


# ---------------------------------------------------------------------------
# StreamingChipStore — interface stub (not yet implemented)
#
# This class will issue on-demand COG range requests rather than reading from
# disk. It satisfies the ChipStore Protocol so extraction and composite code
# needs no changes when switching backends.
#
# Implementation is deferred until scale requires it — e.g. national-scale
# inference where staging the full chip set would exceed EBS budget.
#
# class StreamingChipStore:
#     def __init__(self, stac_client, bands, window_px=5, max_concurrent=32):
#         ...
#     def get(self, item_id: str, band: str, point_id: str) -> np.ndarray:
#         ...  # async COG range request, returned synchronously via asyncio.run()
# ---------------------------------------------------------------------------
