"""Stage 5: point extraction from staged chips.

Reads chips via a ChipStore and returns one Observation per usable
(item, point) pair. An acquisition is usable when its SCL chip exists
and contains at least one clear pixel (SCL value in SCL_CLEAR_VALUES).

Quality components computed here
---------------------------------
scl_purity  : fraction of clear pixels in the SCL chip window.
aot         : 1.0 - center-pixel AOT value (higher = cleaner air).
view_zenith : 1.0 - (VZA_deg / 90), clamped to [0, 1].
sun_zenith  : 1.0 - (SZA_deg / 90), clamped to [0, 1].
greenness_z : deferred to Session 6 — always 1.0 here.

Missing optional chips (AOT, VZA, SZA) default to quality = 1.0 (neutral).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable

import numpy as np

from analysis.constants import (
    AOT_BAND,
    BANDS,
    SCL_BAND,
    SCL_CLEAR_VALUES,
    SZA_BAND,
    VZA_BAND,
)
from analysis.timeseries.observation import Observation, ObservationQuality
from utils.chip_store import ChipStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scl_has_clear_pixels(arr: np.ndarray) -> bool:
    """Return True if any pixel in arr is a clear SCL value."""
    return bool(np.any(np.isin(arr.astype(np.int32), list(SCL_CLEAR_VALUES))))


def _read_center(
    store: ChipStore,
    item_id: str,
    band: str,
    point_id: str,
    center_px: int,
    transform: Callable[[float], float],
    default: float,
) -> float:
    """Read center pixel of a chip and apply transform; return default on miss."""
    try:
        chip = store.get(item_id, band, point_id)
        raw = float(chip[center_px, center_px])
        return transform(raw)
    except FileNotFoundError:
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_item_observations(
    item,
    points: list[tuple[str, float, float]],
    store: ChipStore,
    bands: list[str],
    center_px: int,
) -> list[Observation]:
    """Extract all observations for a single item. Called in a worker thread."""
    item_id: str = item.id
    item_date: datetime = item.datetime.replace(tzinfo=None)
    tile_id: str = item.properties.get("s2:mgrs_tile", "")
    result: list[Observation] = []

    for point_id, _lon, _lat in points:

        # --- Gate 1: SCL chip must exist and contain clear pixels -----------
        try:
            scl_chip = store.get(item_id, SCL_BAND, point_id)
        except FileNotFoundError:
            continue

        if not _scl_has_clear_pixels(scl_chip):
            continue

        # --- Quality: scl_purity --------------------------------------------
        clear_mask = np.isin(scl_chip.astype(np.int32), list(SCL_CLEAR_VALUES))
        scl_purity = float(np.mean(clear_mask))

        # --- Quality: optional chips (default to 1.0 when absent) ----------
        aot_quality = _read_center(
            store, item_id, AOT_BAND, point_id, center_px,
            transform=lambda v: 1.0 - min(v * 0.001, 1.0),
            default=1.0,
        )
        vza_quality = _read_center(
            store, item_id, VZA_BAND, point_id, center_px,
            transform=lambda v: max(0.0, 1.0 - v / 90.0),
            default=1.0,
        )
        sza_quality = _read_center(
            store, item_id, SZA_BAND, point_id, center_px,
            transform=lambda v: max(0.0, 1.0 - v / 90.0),
            default=1.0,
        )

        quality = ObservationQuality(
            scl_purity=scl_purity,
            aot=aot_quality,
            view_zenith=vza_quality,
            sun_zenith=sza_quality,
            greenness_z=1.0,
        )

        # --- Extract center pixel for each spectral band --------------------
        band_values: dict[str, float] = {}
        for band in bands:
            try:
                chip = store.get(item_id, band, point_id)
                band_values[band] = float(chip[center_px, center_px]) / 10000.0
            except FileNotFoundError:
                logger.debug(
                    "Band chip missing: item=%s band=%s point=%s — skipping band",
                    item_id, band, point_id,
                )

        if not band_values:
            logger.warning(
                "No band chips for %s / %s — skipping observation",
                item_id, point_id,
            )
            continue

        result.append(Observation(
            point_id=point_id,
            date=item_date,
            bands=band_values,
            quality=quality,
            meta={"item_id": item_id, "tile_id": tile_id},
        ))

    return result


def extract_observations(
    items: list,
    points: list[tuple[str, float, float]],
    store: ChipStore,
    bands: list[str] = BANDS,
    center_px: int = 2,
    max_workers: int = 8,
) -> list[Observation]:
    """Extract one Observation per usable (item, point) pair from staged chips.

    Parameters
    ----------
    items:
        pystac.Item objects (duck-typed). Must have .id, .datetime, .properties.
    points:
        List of (point_id, lon, lat) tuples. Same shape as fetch_chips() input.
    store:
        Any ChipStore-protocol object. Typically DiskChipStore pointing at inputs/.
    bands:
        Spectral bands to extract. SCL, AOT, VZA, SZA are handled separately.
        Defaults to the 10-band BANDS list from constants.
    center_px:
        Index into the chip array for point extraction. Default 2 = center of 5x5.
    max_workers:
        Number of threads for parallel item processing. Each thread processes
        one item's worth of chip reads independently.

    Returns
    -------
    list[Observation]
        One Observation per (item, point) pair that passes the SCL usability gate.
        Ordered by item then by point (same iteration order as inputs).
    """
    # Submit one task per item; preserve item order in the output.
    futures_ordered = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in items:
            f = executor.submit(
                _extract_item_observations, item, points, store, bands, center_px
            )
            futures_ordered.append(f)

    observations: list[Observation] = []
    for f in futures_ordered:
        observations.extend(f.result())
    return observations
