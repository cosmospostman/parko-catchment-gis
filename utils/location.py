"""utils/location.py — Location abstraction for Parkinsonia analysis sites.

Locations are loaded from individual YAML files in data/locations/<id>.yaml.
Each file defines the primary bounding box, optional sub-bboxes for presence/
absence regions, and site metadata.

Usage
-----
from utils.location import get, all_locations

loc = get("longreach")
print(loc.summary())
print(loc.bbox_cli)          # "145.415267,-22.781368,145.434751,-22.745436"
print(loc.pixel_count)       # ~80000
print(loc.area_km2)          # ~8.0

# Fetch Sentinel-2 pixel observations for this location:
loc.fetch(start="2020-01-01", end="2025-12-31")
# Writes to data/pixels/longreach/longreach.parquet
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

_LOCATIONS_DIR = Path(__file__).parent.parent / "data" / "locations"
_PROJECT_ROOT = Path(__file__).parent.parent


def tile_chips_path(tile_id: str) -> Path:
    """Shared chip cache directory for an S2 tile: data/pixels/{tile_id}/{tile_id}.chips/"""
    return _PROJECT_ROOT / "data" / "pixels" / tile_id / f"{tile_id}.chips"


@dataclass(frozen=True)
class SubBbox:
    label: str
    role: str   # "presence" | "absence" | "survey"
    bbox: list[float]  # [lon_min, lat_min, lon_max, lat_max]

    @property
    def as_list(self) -> list[float]:
        return self.bbox

    @property
    def as_dict(self) -> dict:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return dict(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    bbox: list[float]              # [lon_min, lat_min, lon_max, lat_max]
    dry_months: list[int]
    centroid: Optional[tuple[float, float]]  # (lat, lon) or None
    notes: Optional[str]
    sub_bboxes: dict[str, SubBbox] = field(default_factory=dict)
    signal_params: dict = field(default_factory=dict)  # raw signals: dict from YAML
    score_bbox: Optional[list[float]] = None            # if set, restrict scoring to this bbox

    # ------------------------------------------------------------------
    # Bbox accessors
    # ------------------------------------------------------------------

    @property
    def bbox_list(self) -> list[float]:
        """[lon_min, lat_min, lon_max, lat_max] — for fetch_patches() / collect()"""
        return self.bbox

    @property
    def bbox_dict(self) -> dict:
        """dict(lon_min=...) — for WMS tile fetch and plotting scripts"""
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return dict(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)

    @property
    def bbox_cli(self) -> str:
        """'lon_min,lat_min,lon_max,lat_max' — for --bbox CLI argument"""
        return ",".join(str(v) for v in self.bbox)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def _lon_m(self) -> float:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        lat_centre = (lat_min + lat_max) / 2
        return (lon_max - lon_min) * 111_320 * math.cos(math.radians(lat_centre))

    @property
    def _lat_m(self) -> float:
        _, lat_min, _, lat_max = self.bbox
        return (lat_max - lat_min) * 111_320

    @property
    def width_km(self) -> float:
        return self._lon_m / 1000

    @property
    def height_km(self) -> float:
        return self._lat_m / 1000

    @property
    def area_km2(self) -> float:
        return self._lon_m * self._lat_m / 1e6

    @property
    def pixel_count(self) -> int:
        """Approximate S2 pixel count at 10 m resolution."""
        return int(self._lon_m / 10) * int(self._lat_m / 10)

    def estimated_parquet_mb(
        self,
        obs_per_pixel: int = 387,
        bytes_per_row: int = 120,
    ) -> float:
        """Rough parquet size estimate in MB.

        Defaults match empirical constants derived from Longreach:
        ~387 clear S2 observations per pixel over 2020–2025, ~120 bytes/row.
        """
        return self.pixel_count * obs_per_pixel * bytes_per_row / 1e6

    def summary(self) -> str:
        lines = [
            f"Location : {self.name}  (id={self.id})",
            f"  Bbox   : {self.bbox}",
            f"  Size   : {self.width_km:.1f} km × {self.height_km:.1f} km",
            f"  Area   : {self.area_km2:.1f} km²",
            f"  Pixels : ~{self.pixel_count:,} at 10 m",
            f"  ~Parquet: {self.estimated_parquet_mb():.0f} MB",
        ]
        if self.centroid:
            lines.append(f"  Centroid: {self.centroid[0]}, {self.centroid[1]}")
        if self.sub_bboxes:
            lines.append(f"  Sub-bboxes: {list(self.sub_bboxes.keys())}")
        if self.notes:
            lines.append(f"  Notes  : {self.notes.strip()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def parquet_path(self) -> Path:
        """Canonical pixel observation parquet: data/pixels/<id>/<id>.parquet"""
        return _PROJECT_ROOT / "data" / "pixels" / self.id / f"{self.id}.parquet"

    def coords_cache_path(self, tile_id: str | None = None) -> Path:
        """Sidecar parquet caching unique (point_id, lon, lat) for this location."""
        suffix = f".{tile_id}" if tile_id else ""
        return _PROJECT_ROOT / "data" / "pixels" / self.id / f"{self.id}.coords{suffix}.parquet"

    def chips_path(self) -> Path:
        """Canonical fetch chip cache: data/pixels/<id>/<id>.chips/"""
        return _PROJECT_ROOT / "data" / "pixels" / self.id / f"{self.id}.chips"

    def calibration_path(self) -> Path | None:
        """Return the tile harmonisation correction table path if it exists, else None."""
        p = _PROJECT_ROOT / "data" / "calibration" / f"{self.id}.parquet"
        return p if p.exists() else None

    def cache_dir(self) -> Path:
        """Chip cache dir for this location.

        Single-tile locations return the shared tile-level chips path so that
        training and inference pipelines naturally share the same cache tree.
        Multi-tile locations fall back to the per-location chips path.
        """
        from utils.s2_tiles import bbox_to_tile_ids  # noqa: PLC0415
        tile_ids = bbox_to_tile_ids(tuple(self.bbox))  # type: ignore[arg-type]
        if len(tile_ids) == 1:
            return tile_chips_path(tile_ids[0])
        return self.chips_path()

    # ------------------------------------------------------------------
    # Fetch integration
    # ------------------------------------------------------------------

    def fetch(
        self,
        out_path: Optional[Path] = None,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        cloud_max: int = 30,
        cache_dir: Optional[Path] = None,
        stride: int = 1,
        apply_nbar: bool = True,
    ) -> Path:
        """Fetch Sentinel-2 pixel observations for this location.

        Delegates to scripts.collect_pixel_observations.collect().
        Output is written to data/pixels/<id>/<id>.parquet by default.

        Returns the output parquet path.
        """
        from utils.pixel_collector import collect  # noqa: PLC0415

        _out = out_path or self.parquet_path()
        _end = end or date.today().isoformat()
        _cache = cache_dir or self.cache_dir()

        _out.parent.mkdir(parents=True, exist_ok=True)

        collect(
            bbox_wgs84=self.bbox,
            start=start,
            end=_end,
            out_path=_out,
            cloud_max=cloud_max,
            cache_dir=_cache,
            stride=stride,
            apply_nbar=apply_nbar,
        )
        return _out


# ---------------------------------------------------------------------------
# Registry — lazy-loaded from data/locations/*.yaml
# ---------------------------------------------------------------------------

_registry: dict[str, Location] | None = None


def _load_registry(locations_dir: Path = _LOCATIONS_DIR) -> dict[str, Location]:
    result: dict[str, Location] = {}
    for yaml_path in sorted(locations_dir.glob("*.yaml")):
        loc_id = yaml_path.stem
        with yaml_path.open() as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict) or "name" not in data:
            continue

        sub_bboxes: dict[str, SubBbox] = {}
        for sub_key, sub_data in (data.get("sub_bboxes") or {}).items():
            sub_bboxes[sub_key] = SubBbox(
                label=sub_data["label"],
                role=sub_data["role"],
                bbox=sub_data["bbox"],
            )

        centroid = data.get("centroid")
        result[loc_id] = Location(
            id=loc_id,
            name=data["name"],
            bbox=data["bbox"],
            dry_months=data.get("dry_months", [6, 7, 8, 9, 10]),
            centroid=tuple(centroid) if centroid else None,
            notes=data.get("notes"),
            sub_bboxes=sub_bboxes,
            signal_params=data.get("signals") or {},
            score_bbox=data.get("score_bbox"),
        )
    return result


def get(loc_id: str) -> Location:
    """Return a Location by its id slug. Raises KeyError if not found."""
    global _registry
    if _registry is None:
        _registry = _load_registry()
    return _registry[loc_id]


def all_locations() -> list[Location]:
    """Return all locations sorted by id."""
    global _registry
    if _registry is None:
        _registry = _load_registry()
    return list(_registry.values())


# ---------------------------------------------------------------------------
# Training-region pixel estimates
# ---------------------------------------------------------------------------

def _bbox_pixel_count(bbox: list[float], resolution_m: float = 10.0) -> int:
    """Estimate S2 pixel count for a [lon_min, lat_min, lon_max, lat_max] bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_centre = (lat_min + lat_max) / 2
    lon_m = (lon_max - lon_min) * 111_320 * math.cos(math.radians(lat_centre))
    lat_m = (lat_max - lat_min) * 111_320
    return int(lon_m / resolution_m) * int(lat_m / resolution_m)


def training_pixel_summary(resolution_m: float = 10.0) -> None:
    """Print estimated pixel counts for all training regions, grouped by label."""
    from training.regions import load_regions  # noqa: PLC0415

    regions = load_regions()
    totals: dict[str, int] = {}

    print(f"{'ID':<40} {'LABEL':<10} {'PIXELS':>8}")
    print("-" * 62)
    for r in regions:
        n = _bbox_pixel_count(r.bbox, resolution_m)
        totals[r.label] = totals.get(r.label, 0) + n
        print(f"{r.id:<40} {r.label:<10} {n:>8,}")

    print("-" * 62)
    for label, total in sorted(totals.items()):
        print(f"{'Total ' + label:<40} {'':10} {total:>8,}")


if __name__ == "__main__":
    training_pixel_summary()
