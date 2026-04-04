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
