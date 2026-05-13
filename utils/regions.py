"""utils/regions.py — Load and select training regions from YAML.

Usage
-----
    from utils.regions import load_regions, select_regions

    # Load all regions
    regions = load_regions()

    # Select specific regions by ID for an experiment
    selected = select_regions(["lake_mueller_presence", "lake_mueller_absence"])
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGIONS_YAML = PROJECT_ROOT / "data" / "locations" / "training.yaml"


@dataclass(frozen=True)
class TrainingRegion:
    id: str
    name: str
    label: str          # "presence" | "absence"
    bbox: list[float]   # [lon_min, lat_min, lon_max, lat_max]
    years: list[int]    # calendar years to fetch and train on (explicit, non-empty)
    tags: list[str]
    notes: str | None

    @property
    def is_presence(self) -> bool:
        return self.label == "presence"

    @property
    def bbox_tuple(self) -> tuple[float, float, float, float]:
        return tuple(self.bbox)  # type: ignore[return-value]


def _parse_years(entry: dict) -> list[int]:
    """Return the years list for a region entry.

    Accepts the new ``years: [...]`` form.  If the entry has the old singular
    ``year: YYYY`` field instead, expands it to ``[year-5, ..., year]`` and
    emits a deprecation warning so YAML authors know to migrate.
    """
    if "years" in entry:
        years = list(entry["years"])
        if not years:
            raise ValueError(
                f"Region {entry.get('id', '?')!r}: 'years' must be a non-empty list"
            )
        return years

    if "year" in entry and entry["year"] is not None:
        year = int(entry["year"])
        expanded = list(range(year - 5, year + 1))
        warnings.warn(
            f"Region {entry.get('id', '?')!r}: 'year: {year}' is deprecated — "
            f"replace with 'years: {expanded}'",
            DeprecationWarning,
            stacklevel=4,
        )
        return expanded

    raise ValueError(
        f"Region {entry.get('id', '?')!r}: must have a 'years' field"
    )


def load_regions(yaml_path: Path = _REGIONS_YAML) -> list[TrainingRegion]:
    """Load all training regions from the YAML file."""
    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)

    regions = []
    for entry in data.get("regions", []):
        regions.append(TrainingRegion(
            id=entry["id"],
            name=entry["name"],
            label=entry["label"],
            bbox=entry["bbox"],
            years=_parse_years(entry),
            tags=entry.get("tags", []),
            notes=entry.get("notes"),
        ))
    return regions


def select_regions(
    region_ids: list[str],
    yaml_path: Path = _REGIONS_YAML,
) -> list[TrainingRegion]:
    """Return TrainingRegions matching the given IDs, preserving order.

    Raises KeyError if any ID is not found in the YAML.
    """
    all_regions = {r.id: r for r in load_regions(yaml_path)}
    missing = [rid for rid in region_ids if rid not in all_regions]
    if missing:
        raise KeyError(f"Unknown training region IDs: {missing}")
    return [all_regions[rid] for rid in region_ids]
