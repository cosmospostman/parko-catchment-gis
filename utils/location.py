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
loc.fetch(years=[2020, 2021, 2022, 2023, 2024, 2025])
# Writes to data/pixels/longreach/<year>/<tile_id>.parquet
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
    polygon_file: Optional[Path] = None                 # GeoJSON file; pixels outside polygon are dropped

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

    @property
    def geometry(self):
        """Shapely geometry loaded from polygon_file, or None for bbox-only locations."""
        if self.polygon_file is None:
            return None
        import json
        from shapely.geometry import shape
        with self.polygon_file.open() as fh:
            gj = json.load(fh)
        if gj["type"] == "FeatureCollection":
            gj = gj["features"][0]["geometry"]
        elif gj["type"] == "Feature":
            gj = gj["geometry"]
        return shape(gj)

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
        geom = self.geometry
        if geom is not None:
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            area_m2, _ = geod.geometry_area_perimeter(geom)
            return abs(area_m2) / 1e6
        return self._lon_m * self._lat_m / 1e6

    @property
    def pixel_count(self) -> int:
        """Approximate S2 pixel count at 10 m resolution."""
        return round(self._lon_m / 10) * round(self._lat_m / 10)

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

    def parquet_year_dir(self, year: int) -> Path:
        """Directory holding per-tile parquets for one year: data/pixels/<id>/<year>/"""
        return _PROJECT_ROOT / "data" / "pixels" / self.id / str(year)

    def parquet_path(self, year: int, tile_id: str) -> Path:
        """Canonical pixel observation parquet: data/pixels/<id>/<year>/<tile_id>.parquet"""
        return self.parquet_year_dir(year) / f"{tile_id}.parquet"

    def parquet_tile_paths(self) -> dict[int, list[Path]]:
        """Return {year: [tile_parquet_paths]} for all fetched data."""
        base = _PROJECT_ROOT / "data" / "pixels" / self.id
        result: dict[int, list[Path]] = {}
        if not base.is_dir():
            return result
        for child in base.iterdir():
            if not (child.is_dir() and child.name.isdigit()):
                continue
            year = int(child.name)
            paths = sorted(
                p for p in child.iterdir()
                if p.suffix == ".parquet" and not p.stem.startswith("_collect_")
                and not p.stem.endswith("-by-pixel")
                and not p.stem.endswith(".coords")
                and "coords" not in p.stem
            )
            if paths:
                result[year] = paths
        return result

    def parquet_years(self) -> list[int]:
        """Return sorted list of years that have at least one tile parquet on disk."""
        return sorted(self.parquet_tile_paths().keys())

    def coords_cache_path(self, year: int, tile_id: str | None = None) -> Path:
        """Sidecar parquet caching unique (point_id, lon, lat) for this location and year."""
        suffix = f".{tile_id}" if tile_id else ""
        return _PROJECT_ROOT / "data" / "pixels" / self.id / str(year) / f"{self.id}.coords{suffix}.parquet"

    def chips_path(self) -> Path:
        """Canonical fetch chip cache: data/pixels/<id>/<id>.chips/ (shared across years)"""
        return _PROJECT_ROOT / "data" / "pixels" / self.id / f"{self.id}.chips"

    def calibration_path(self) -> Path | None:
        """Return the tile harmonisation correction table path if it exists, else None."""
        p = _PROJECT_ROOT / "data" / "calibration" / f"{self.id}.parquet"
        return p if p.exists() else None

    def tile_ids(self) -> list[str]:
        """Return S2 tile IDs that will actually produce observations for this location.

        When a polygon_file is defined, uses geometry_to_tile_ids() to exclude tiles
        whose overlap with the bbox contains no pixels canonically assigned to them.
        Falls back to bbox_to_tile_ids() for bbox-only locations.
        """
        from utils.s2_tiles import bbox_to_tile_ids, geometry_to_tile_ids  # noqa: PLC0415
        geom = self.geometry
        if geom is not None:
            return geometry_to_tile_ids(geom, tuple(self.bbox))  # type: ignore[arg-type]
        return bbox_to_tile_ids(tuple(self.bbox))  # type: ignore[arg-type]

    def cache_dir(self) -> Path:
        """Chip cache dir for this location.

        Single-tile locations return the shared tile-level chips path so that
        training and inference pipelines naturally share the same cache tree.
        Multi-tile locations fall back to the per-location chips path.
        """
        tile_ids = self.tile_ids()
        if len(tile_ids) == 1:
            return tile_chips_path(tile_ids[0])
        return self.chips_path()

    # ------------------------------------------------------------------
    # Fetch integration
    # ------------------------------------------------------------------

    def fetch(
        self,
        years: list[int],
        cloud_max: int = 30,
        cache_dir: Optional[Path] = None,
        apply_nbar: bool = True,
    ) -> list[Path]:
        """Fetch Sentinel-2 and Sentinel-1 pixel observations for this location.

        Writes one parquet per S2 tile per year to data/pixels/<id>/<year>/<tile_id>.parquet.
        Each parquet contains S2 rows (source="S2") interleaved with S1 rows (source="S1",
        vh, vv columns populated, S2 band columns null).

        Returns the list of written parquet paths.
        """
        from utils.pixel_collector import collect  # noqa: PLC0415
        from utils.s1_collector import collect_s1, _DEFAULT_CACHE_DIR as _S1_CACHE_DIR  # noqa: PLC0415

        _cache = cache_dir or self.cache_dir()

        _cal_out: Path | None = None
        if len(self.tile_ids()) > 1:
            _cal_out = _PROJECT_ROOT / "data" / "calibration" / f"{self.id}.parquet"
            _cal_out.parent.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for year in sorted(years):
            _out_dir = self.parquet_year_dir(year)
            _out_dir.mkdir(parents=True, exist_ok=True)
            tile_paths = collect(
                bbox_wgs84=self.bbox,
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                out_dir=_out_dir,
                cloud_max=cloud_max,
                cache_dir=_cache,
                apply_nbar=apply_nbar,
                calibration_out=_cal_out,
                geometry=self.geometry,
            )

            # collect() returns [] when all shards were already done AND the
            # sorted shard files have been consumed into tile parquets (deleted
            # after the concat step).  Fall back to existing tile parquets so
            # the S1 append step below still runs.
            if not tile_paths:
                tile_paths = [
                    p for p in sorted(_out_dir.glob("*.parquet"))
                    if not p.name.startswith("_") and ".coords." not in p.name
                ]

            # Append S1 rows to each tile parquet in-place
            for tile_path in tile_paths:
                _append_s1_to_tile_parquet(
                    tile_path=tile_path,
                    bbox_wgs84=self.bbox,
                    year=year,
                    collect_s1_fn=collect_s1,
                    s1_cache_dir=_S1_CACHE_DIR,
                )

            written.extend(tile_paths)
        return written


# ---------------------------------------------------------------------------
# S1 append helper — extracted for testability
# ---------------------------------------------------------------------------

def _append_s1_to_tile_parquet(
    tile_path: Path,
    bbox_wgs84: list[float],
    year: int,
    collect_s1_fn,
    s1_cache_dir: Path | None = None,
) -> None:
    """Append S1 rows to an existing S2-only tile parquet, in-place and atomically.

    Idempotent: skips the file if it already contains at least one S1 row.

    The parquet is pixel-sorted, so distinct pixels are spread across all row
    groups.  This function scans *all* row groups to collect the full pixel grid
    before calling collect_s1_fn — the original inline code only scanned the
    first row group, producing S1 coverage for a single pixel strip.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.training_collector import _extend_schema, _conform_table, _s1_df_to_arrow  # noqa: PLC0415

    pf = pq.ParquetFile(tile_path)

    # Already has S1 rows — skip (idempotent re-fetch).
    # Check for actual S1 rows, not just schema presence: schema columns may
    # exist from a prior incomplete fetch that returned empty data.
    if "source" in pf.schema_arrow.names and "vh" in pf.schema_arrow.names:
        has_s1 = any(
            "S1" in pf.read_row_group(rg, columns=["source"]).column("source").to_pylist()
            for rg in range(pf.metadata.num_row_groups)
        )
        if has_s1:
            return

    # Scan ALL row groups for pixel coords.  A pixel-sorted parquet places each
    # pixel's observations in a contiguous block, so the first row group only
    # covers the first ~N pixels — not the full grid.
    combined_schema = _extend_schema(pf.read_row_group(0).schema)
    seen: dict[str, tuple[float, float]] = {}
    n_rg = pf.metadata.num_row_groups
    for rg_idx in range(n_rg):
        coord_tbl = pf.read_row_group(rg_idx, columns=["point_id", "lon", "lat"])
        for pid, lon, lat in zip(
            coord_tbl.column("point_id").to_pylist(),
            coord_tbl.column("lon").to_pylist(),
            coord_tbl.column("lat").to_pylist(),
        ):
            if pid not in seen:
                seen[pid] = (lon, lat)

    points_for_s1: list[tuple[str, float, float]] = [
        (p, lo, la) for p, (lo, la) in seen.items()
    ]

    df_s1 = collect_s1_fn(
        bbox_wgs84=bbox_wgs84,
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        points=points_for_s1,
        cache_dir=s1_cache_dir,
    )

    # Rewrite atomically — stream S2 row groups to avoid loading the full file
    # into RAM, then append sorted S1 rows as a final row group.
    tmp_path = tile_path.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(tmp_path, combined_schema)
    for rg_idx in range(n_rg):
        tbl = pf.read_row_group(rg_idx)
        tbl = _conform_table(tbl, combined_schema)
        source_col = pa.array(["S2"] * len(tbl), type=pa.string())
        tbl = tbl.set_column(tbl.schema.get_field_index("source"), "source", source_col)
        writer.write_table(tbl)
    if df_s1 is not None and not df_s1.empty:
        df_s1 = df_s1.sort_values(["point_id", "date"]).reset_index(drop=True)
        writer.write_table(_s1_df_to_arrow(df_s1, combined_schema))
    writer.close()
    tmp_path.replace(tile_path)


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
        if centroid is not None:
            if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
                raise ValueError(
                    f"{yaml_path}: 'centroid' must be a [lat, lon] list, got {centroid!r}"
                )
            centroid = tuple(centroid)
        else:
            centroid = None

        polygon_file: Path | None = None
        raw_pf = data.get("polygon_file")
        if raw_pf is not None:
            polygon_file = _PROJECT_ROOT / raw_pf

        bbox = data.get("bbox")
        if bbox is None:
            if polygon_file is None:
                raise ValueError(f"{yaml_path}: must provide 'bbox' or 'polygon_file'")
            import json
            from shapely.geometry import shape
            with polygon_file.open() as fh:
                gj = json.load(fh)
            if gj["type"] == "FeatureCollection":
                gj = gj["features"][0]["geometry"]
            elif gj["type"] == "Feature":
                gj = gj["geometry"]
            geom = shape(gj)
            minx, miny, maxx, maxy = geom.bounds
            bbox = [minx, miny, maxx, maxy]

        result[loc_id] = Location(
            id=loc_id,
            name=data["name"],
            bbox=bbox,
            dry_months=data.get("dry_months", [6, 7, 8, 9, 10]),
            centroid=centroid,
            notes=data.get("notes"),
            sub_bboxes=sub_bboxes,
            signal_params=data.get("signals") or {},
            score_bbox=data.get("score_bbox"),
            polygon_file=polygon_file,
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
    return round(lon_m / resolution_m) * round(lat_m / resolution_m)


def training_pixel_summary(resolution_m: float = 10.0) -> None:
    """Print estimated pixel counts for all training regions, grouped by label."""
    from utils.regions import load_regions  # noqa: PLC0415

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
