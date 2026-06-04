"""utils/pixel_reader.py — Random-access pixel lookups from chunk parquets.

Chunk parquets live at:
    {root}/{year}/{tile_id}/{tile_id}_rNN_cNN.parquet

Each chunk is pixel-sorted: rows are grouped by point_id, with all temporal
observations for a pixel contiguous.  Row groups are sorted by northing, so
adjacent RGs cover overlapping lat bands (~1–2 pixel rows of overlap).

Usage::

    idx = ChunkIndex(Path("/mnt/external/chunkstore"), year=2025, tile_id="54LWH")
    tbl = idx.query_point(lon=141.735, lat=-15.743)   # pa.Table | None
    tbl = idx.query_bbox(141.70, -15.76, 141.80, -15.70)  # pa.Table
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_CHUNK_RE = re.compile(r"_r(\d{2})_c(\d{2})$")


@dataclass
class _RGMeta:
    index: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass
class _ChunkMeta:
    path: Path
    pf: pq.ParquetFile
    row_groups: list[_RGMeta]
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


class ChunkIndex:
    """In-memory spatial index over chunk parquet footers for a single tile×year.

    Construction opens every chunk file to read its footer (one seek per file),
    then caches per-RG lat/lon stats.  All subsequent spatial filtering is
    in-memory; only the matching RGs are read from disk per query.

    Parameters
    ----------
    root:
        Root directory containing ``{year}/{tile_id}/`` subdirectories.
    year:
        Calendar year of the chunk data.
    tile_id:
        MGRS tile ID, e.g. ``"54LWH"``.
    """

    def __init__(self, root: Path, year: int, tile_id: str) -> None:
        self._chunks: list[_ChunkMeta] = []
        tile_dir = root / str(year) / tile_id
        for path in sorted(tile_dir.glob(f"{tile_id}_r??_c??.parquet")):
            m = _CHUNK_RE.search(path.stem)
            if not m:
                continue
            pf = pq.ParquetFile(path)
            md = pf.metadata
            schema_names = pf.schema_arrow.names
            lon_idx = schema_names.index("lon")
            lat_idx = schema_names.index("lat")
            rg_metas: list[_RGMeta] = []
            for i in range(md.num_row_groups):
                rg = md.row_group(i)
                lon_st = rg.column(lon_idx).statistics
                lat_st = rg.column(lat_idx).statistics
                rg_metas.append(_RGMeta(
                    index=i,
                    lon_min=lon_st.min if lon_st else float("-inf"),
                    lon_max=lon_st.max if lon_st else float("inf"),
                    lat_min=lat_st.min if lat_st else float("-inf"),
                    lat_max=lat_st.max if lat_st else float("inf"),
                ))
            self._chunks.append(_ChunkMeta(
                path=path,
                pf=pf,
                row_groups=rg_metas,
                lon_min=min(r.lon_min for r in rg_metas),
                lon_max=max(r.lon_max for r in rg_metas),
                lat_min=min(r.lat_min for r in rg_metas),
                lat_max=max(r.lat_max for r in rg_metas),
            ))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_point(self, lon: float, lat: float) -> pa.Table | None:
        """Return all rows for the single pixel nearest to ``(lon, lat)``.

        Returns ``None`` if no pixel is found within any chunk covering the
        query location.
        """
        chunk = self._find_chunk(lon, lat)
        if chunk is None:
            return None

        candidate_rgs = self._candidate_rgs(chunk, lon, lon, lat, lat)
        if not candidate_rgs:
            return None

        tables = [chunk.pf.read_row_group(rg.index) for rg in candidate_rgs]
        combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

        # Find the point_id whose (lon, lat) is closest to the query.
        # Arrow has no argmin, so compute squared distances and find min via pylist.
        lon_arr = pc.cast(combined.column("lon"), pa.float64())
        lat_arr = pc.cast(combined.column("lat"), pa.float64())
        dist = pc.add(
            pc.power(pc.subtract(lon_arr, lon), 2),
            pc.power(pc.subtract(lat_arr, lat), 2),
        )
        dist_list = dist.to_pylist()
        nearest_idx = dist_list.index(min(dist_list))
        target_pid = combined.column("point_id")[nearest_idx].as_py()

        mask = pc.equal(combined.column("point_id"), target_pid)
        return combined.filter(mask)

    def query_bbox(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
    ) -> pa.Table:
        """Return all rows for every pixel whose ``(lon, lat)`` falls within the bbox.

        Returns an empty table (with the schema of the first chunk) if no pixels
        are found.
        """
        parts: list[pa.Table] = []

        for chunk in self._chunks:
            if not _envelopes_overlap(
                chunk.lon_min, chunk.lon_max, chunk.lat_min, chunk.lat_max,
                lon_min, lon_max, lat_min, lat_max,
            ):
                continue

            candidate_rgs = self._candidate_rgs(chunk, lon_min, lon_max, lat_min, lat_max)
            if not candidate_rgs:
                continue

            for rg in candidate_rgs:
                tbl = chunk.pf.read_row_group(rg.index)
                mask = (
                    pc.and_(
                        pc.and_(
                            pc.greater_equal(tbl.column("lon"), lon_min),
                            pc.less_equal(tbl.column("lon"), lon_max),
                        ),
                        pc.and_(
                            pc.greater_equal(tbl.column("lat"), lat_min),
                            pc.less_equal(tbl.column("lat"), lat_max),
                        ),
                    )
                )
                filtered = tbl.filter(mask)
                if filtered.num_rows > 0:
                    parts.append(filtered)

        if not parts:
            # Return empty table with correct schema if we have any chunk loaded.
            if self._chunks:
                return self._chunks[0].pf.schema_arrow.empty_table()
            return pa.table({})

        return pa.concat_tables(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_chunk(self, lon: float, lat: float) -> _ChunkMeta | None:
        for chunk in self._chunks:
            if (chunk.lon_min <= lon <= chunk.lon_max and
                    chunk.lat_min <= lat <= chunk.lat_max):
                return chunk
        return None

    def _candidate_rgs(
        self,
        chunk: _ChunkMeta,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> list[_RGMeta]:
        return [
            rg for rg in chunk.row_groups
            if _envelopes_overlap(
                rg.lon_min, rg.lon_max, rg.lat_min, rg.lat_max,
                lon_min, lon_max, lat_min, lat_max,
            )
        ]


def _envelopes_overlap(
    a_lon_min: float, a_lon_max: float, a_lat_min: float, a_lat_max: float,
    b_lon_min: float, b_lon_max: float, b_lat_min: float, b_lat_max: float,
) -> bool:
    return (
        a_lon_min <= b_lon_max and a_lon_max >= b_lon_min and
        a_lat_min <= b_lat_max and a_lat_max >= b_lat_min
    )
