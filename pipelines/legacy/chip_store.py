"""pipelines/legacy/chip_store.py — disk-based chip storage for the legacy pipeline.

See utils/chip_store.py for the ChipStore protocol and MemoryChipStore.
"""

from __future__ import annotations

from pathlib import Path

import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from utils.chip_store import ChipStore  # re-export for callers that import from here

__all__ = ["ChipStore", "DiskChipStore"]


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
