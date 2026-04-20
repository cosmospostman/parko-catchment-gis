"""training/regions.py — Load and select training regions from YAML.

Usage
-----
    from training.regions import load_regions, select_regions

    # Load all regions
    regions = load_regions()

    # Select specific regions by ID for an experiment
    selected = select_regions(["lake_mueller_presence", "lake_mueller_absence"])
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGIONS_YAML = PROJECT_ROOT / "data" / "locations" / "training.yaml"


@dataclass(frozen=True)
class TrainingRegion:
    id: str
    name: str
    label: str          # "presence" | "absence"
    bbox: list[float]   # [lon_min, lat_min, lon_max, lat_max]
    year: int | None    # if set, pin observations to [year-5, year]
    tags: list[str]
    notes: str | None

    @property
    def is_presence(self) -> bool:
        return self.label == "presence"

    @property
    def bbox_tuple(self) -> tuple[float, float, float, float]:
        return tuple(self.bbox)  # type: ignore[return-value]


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
            year=entry.get("year"),
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
