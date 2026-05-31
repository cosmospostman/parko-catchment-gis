from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def point_ids_in_geometry(paths: "list[Path]", geometry) -> "set[str]":
    """Return the set of point_ids whose pixels fall inside *geometry*.

    Reads the point_id column from each parquet file and looks up each pixel's
    (lon, lat) via the point_id→coordinate mapping stored in the parquet.  Only
    works when the parquet contains 'lon' and 'lat' columns (written by the fetch
    pipeline).

    Useful for creating a masked ChunkPixelSource without re-fetching:
        pids = point_ids_in_geometry(chunk_paths, my_polygon)
        src  = ChunkPixelSource(chunk_paths, point_ids=pids)
    """
    from shapely import contains_xy as _contains_xy
    import pyarrow.compute as _pac

    keep: set[str] = set()
    for p in paths:
        pf = pq.ParquetFile(p)
        available = set(pf.schema_arrow.names)
        if "point_id" not in available or "lon" not in available or "lat" not in available:
            continue
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["point_id", "lon", "lat"])
            # Deduplicate — many rows per point_id; one lon/lat is enough.
            # Use a simple set to track which pids we've already tested.
            pids_col = tbl.column("point_id").to_pylist()
            lons_col = tbl.column("lon").to_pylist()
            lats_col = tbl.column("lat").to_pylist()
            seen: dict[str, tuple[float, float]] = {}
            for pid, lon, lat in zip(pids_col, lons_col, lats_col):
                if pid not in seen and pid not in keep:
                    seen[pid] = (lon, lat)
            if not seen:
                continue
            pid_list = list(seen.keys())
            lons = [seen[p][0] for p in pid_list]
            lats = [seen[p][1] for p in pid_list]
            import numpy as _np
            mask = _contains_xy(geometry, _np.asarray(lons), _np.asarray(lats))
            for pid, in_geom in zip(pid_list, mask):
                if in_geom:
                    keep.add(pid)
    return keep


def _count_distinct_pixels(paths: list[Path]) -> int:
    """Count distinct point_ids across parquet files.

    Reads from a .pixel_count sidecar written by merge_scenes when available
    (O(1), metadata-only).  Falls back to DuckDB count(distinct point_id) for
    files that predate the sidecar or were written by other paths.
    Strips are spatially disjoint so per-file counts sum correctly.
    """
    total = 0
    duckdb_paths: list[Path] = []
    for p in paths:
        sidecar = p.with_suffix(".pixel_count")
        if sidecar.exists():
            total += int(sidecar.read_text().strip())
        else:
            duckdb_paths.append(p)
    if duckdb_paths:
        import duckdb
        for p in duckdb_paths:
            total += duckdb.sql(f"SELECT count(distinct point_id) FROM read_parquet('{p}')").fetchone()[0]
    return total


class PixelSource(ABC):
    @property
    @abstractmethod
    def num_row_groups(self) -> int: ...

    @abstractmethod
    def read_row_group(self, i: int, columns: list[str]) -> pa.Table: ...

    @property
    @abstractmethod
    def schema(self) -> pa.Schema: ...

    def num_pixels(self) -> int:
        raise NotImplementedError("num_pixels() requires a path-backed PixelSource")


class ParquetPixelSource(PixelSource):
    """Wraps a single pq.ParquetFile."""
    def __init__(self, path: Path):
        self._path = path
        self._pf = pq.ParquetFile(path)

    @property
    def num_row_groups(self) -> int:
        return self._pf.metadata.num_row_groups

    def read_row_group(self, i: int, columns: list[str]) -> pa.Table:
        return self._pf.read_row_group(i, columns=columns)

    @property
    def schema(self) -> pa.Schema:
        return self._pf.schema_arrow

    def num_pixels(self) -> int:
        if not hasattr(self, "_num_pixels_cache"):
            self._num_pixels_cache = _count_distinct_pixels([self._path])
        return self._num_pixels_cache


class ChunkPixelSource(PixelSource):
    """Presents N chunk parquets as a single concatenated row-group stream.

    Paths must be supplied in row-major order: (row=0,col=0), (row=0,col=1), ...,
    (row=1,col=0), ...  This preserves northing order within each row-band and
    processes row-bands from north to south.

    num_row_groups = sum of all chunk row-group counts.
    read_row_group(i) maps global index → (chunk_file, local_rg).

    Parameters
    ----------
    paths:
        Chunk parquet paths in row-major order.
    point_ids:
        Optional set of point_id strings to keep.  When supplied, rows whose
        point_id is not in the set are dropped at read time.  Use this to mask
        a full-tile pixel store to a location geometry without re-fetching.
    """
    def __init__(self, paths: list[Path], point_ids: "set[str] | None" = None):
        self._paths = paths
        self._pfs = [pq.ParquetFile(p) for p in paths]
        self._offsets: list[int] = []  # cumulative row-group offsets
        self._point_ids = point_ids
        self._pid_contains = (np.frompyfunc(point_ids.__contains__, 1, 1)
                              if point_ids is not None else None)
        total = 0
        for pf in self._pfs:
            self._offsets.append(total)
            total += pf.metadata.num_row_groups
        self._total = total

    @property
    def num_row_groups(self) -> int:
        return self._total

    def read_row_group(self, i: int, columns: list[str]) -> pa.Table:
        file_idx = 0
        for j in range(len(self._offsets) - 1, -1, -1):
            if i >= self._offsets[j]:
                file_idx = j
                break
        local_rg = i - self._offsets[file_idx]
        return self._pfs[file_idx].read_row_group(local_rg, columns=columns)

    def filter_table(self, tbl: pa.Table) -> pa.Table:
        """Apply point_id geometry filter. Call from the parser thread, not the IO thread."""
        if self._pid_contains is not None and "point_id" in tbl.schema.names:
            mask = self._pid_contains(np.asarray(tbl.column("point_id"))).astype(bool)
            return tbl.filter(pa.array(mask))
        return tbl

    @property
    def schema(self) -> pa.Schema:
        return self._pfs[0].schema_arrow

    def num_pixels(self) -> int:
        if not hasattr(self, "_num_pixels_cache"):
            if self._point_ids is not None:
                self._num_pixels_cache = len(self._point_ids)
            else:
                self._num_pixels_cache = _count_distinct_pixels(self._paths)
        return self._num_pixels_cache
