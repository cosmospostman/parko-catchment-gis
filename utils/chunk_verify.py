"""utils/chunk_verify.py — Verify integrity of chunkstore pixel parquets.

Mirrors the spirit of ``cli/location.py training verify`` but for the per-chunk
scoring parquets under ``{CHUNKSTORE_DIR}/{year}/{tile}/{tile}_rNN_cMM.parquet``.

The headline check is the **S1 truncation defect** (see docs/S1-COVERAGE.md): a
healthy Sentinel-1 IW/GRD acquisition is a ~250 km scene that fully covers any
chunk, so for a correct chunk *most* S1 dates should cover *all* of the chunk's
pixel rows.  Truncated patches cause most dates to cover only a thin band of
rows, which the scorer's MIN_S1_OBS_PER_YEAR filter then drops — leaving the
characteristic thin scored strips.

We quantify this per chunk as the **median per-date S1 row coverage**:
``median over S1 dates of (distinct yi rows that date covers) / (chunk yi rows)``.
A correct chunk sits near 1.0; truncated chunks measured ~0.10.  We also report
``max_frac`` (the best single date) — when it is near 1.0 it proves the chunk
*could* be fully covered, so a low median is unambiguously a defect rather than
a genuine swath edge.

This module is pure (DuckDB over parquet on disk, no network) so it is unit
testable against tiny synthetic parquets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Matches the chunk suffix in a parquet stem, e.g. "55KCB_r01_c00" -> (1, 0).
_CHUNK_RE = re.compile(r"_r(\d+)_c(\d+)$")

# A chunk is flagged TRUNCATED only when BOTH hold:
#   * median per-date S1 row coverage < S1_TRUNCATION_MED_FRAC, AND
#   * the fraction of pixels that reach MIN_S1_OBS_PER_YEAR is below
#     S1_TRUNCATION_MIN_PCT_OBS.
# The second gate matters because some tiny edge chunks have a low med_frac purely
# because the chunk is only a few rows tall — yet every pixel still accumulates >=4
# obs across the year, so the scorer keeps them all and nothing is lost.  Such
# chunks are reported as INFO (truncated but harmless), not FAIL.
# Healthy chunks measure med_frac ~1.0; the harmful defect ~0.10 with ~10% obs.
S1_TRUNCATION_MED_FRAC = 0.5
S1_TRUNCATION_MIN_PCT_OBS = 90.0

# Minimum S1 observations per pixel the scorer requires (tam/core/dataset.py).
# Reported so verify output lines up with what score will actually keep.
MIN_S1_OBS_PER_YEAR = 4


@dataclass
class ChunkReport:
    """Per-chunk verification result.  ``issues`` holds tagged problem strings."""

    path: Path
    year: int | None
    tile: str
    chunk_row: int
    chunk_col: int
    n_pixels: int = 0
    yi_rows: int = 0           # distinct pixel rows (yi) in the chunk
    n_s1_dates: int = 0
    s1_med_frac: float = 0.0   # median per-date S1 row coverage (the key metric)
    s1_max_frac: float = 0.0   # best single date's row coverage
    pct_ge_min_obs: float = 0.0  # % pixels with >= MIN_S1_OBS_PER_YEAR S1 obs
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # non-failing INFO (e.g. harmless truncation)

    @property
    def ok(self) -> bool:
        return not self.issues


def parse_chunk_stem(stem: str) -> tuple[int, int] | None:
    """Return (chunk_row, chunk_col) from a parquet stem, or None if not a chunk."""
    m = _CHUNK_RE.search(stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def verify_chunk(path: Path, con=None) -> ChunkReport:
    """Verify a single chunk parquet.  ``con`` is an optional DuckDB connection."""
    import duckdb

    own_con = con is None
    if own_con:
        con = duckdb.connect()

    stem = path.stem
    rc = parse_chunk_stem(stem) or (-1, -1)
    # Tile is the stem up to the _rNN_cMM suffix (or the whole stem if absent).
    tile = _CHUNK_RE.sub("", stem)
    # Year is the parent-of-tile directory name when it is numeric.
    year: int | None = None
    if path.parent.parent.name.isdigit():
        year = int(path.parent.parent.name)

    rep = ChunkReport(
        path=path, year=year, tile=tile, chunk_row=rc[0], chunk_col=rc[1]
    )

    p = str(path).replace("'", "''")
    try:
        # One pass for the chunk-level shape; the yi row count is the
        # denominator for coverage fractions.
        n_pixels, yi_rows = con.execute(
            f"""
            SELECT COUNT(DISTINCT point_id),
                   COUNT(DISTINCT CAST(split_part(point_id,'_',3) AS INTEGER))
            FROM read_parquet('{p}')
            """
        ).fetchone()
        rep.n_pixels = int(n_pixels or 0)
        rep.yi_rows = int(yi_rows or 0)
    except Exception as e:  # unreadable / corrupt parquet
        rep.issues.append(f"UNREADABLE  {stem} — {e}")
        if own_con:
            con.close()
        return rep

    if rep.n_pixels == 0:
        rep.issues.append(f"EMPTY       {stem} — no pixels")
        if own_con:
            con.close()
        return rep

    # S1 per-date row coverage and per-pixel obs counts.
    row = con.execute(
        f"""
        WITH perdate AS (
            SELECT date,
                   COUNT(DISTINCT CAST(split_part(point_id,'_',3) AS INTEGER)) AS yi_cov
            FROM read_parquet('{p}')
            WHERE source = 'S1'
            GROUP BY date
        ),
        perpix AS (
            SELECT point_id, COUNT(*) FILTER (WHERE source = 'S1') AS n_s1
            FROM read_parquet('{p}')
            GROUP BY point_id
        )
        SELECT
            (SELECT COUNT(*) FROM perdate)                              AS n_dates,
            (SELECT median(yi_cov) FROM perdate)                        AS med_cov,
            (SELECT max(yi_cov) FROM perdate)                           AS max_cov,
            (SELECT COUNT(*) FROM perpix WHERE n_s1 >= {MIN_S1_OBS_PER_YEAR}) AS n_ge_min,
            (SELECT COUNT(*) FROM perpix)                               AS n_total
        """
    ).fetchone()
    n_dates, med_cov, max_cov, n_ge_min, n_total = row
    rep.n_s1_dates = int(n_dates or 0)

    denom = rep.yi_rows or 1
    rep.s1_med_frac = float(med_cov or 0) / denom
    rep.s1_max_frac = float(max_cov or 0) / denom
    rep.pct_ge_min_obs = 100.0 * float(n_ge_min or 0) / float(n_total or 1)

    if rep.n_s1_dates == 0:
        rep.issues.append(f"S1_MISS     {stem} — no S1 observations")
    elif rep.s1_med_frac < S1_TRUNCATION_MED_FRAC:
        _detail = (
            f"median S1 date covers only {rep.s1_med_frac:.0%} of rows "
            f"(max {rep.s1_max_frac:.0%}); {rep.pct_ge_min_obs:.1f}% of pixels reach "
            f">={MIN_S1_OBS_PER_YEAR} obs"
        )
        if rep.pct_ge_min_obs < S1_TRUNCATION_MIN_PCT_OBS:
            # Truncation actually costs scored pixels → rebuild.
            rep.issues.append(f"S1_TRUNC    {stem} — {_detail} — patches were truncated, rebuild needed")
        else:
            # Low med_frac but ~all pixels still reach the obs threshold (tiny edge
            # chunk): the scorer keeps everything, so this is harmless.
            rep.notes.append(f"S1_TRUNC_OK {stem} — {_detail} — harmless (all pixels still scored)")

    if own_con:
        con.close()
    return rep


def iter_chunk_parquets(
    root: Path,
    year: int | None = None,
    tile: str | None = None,
):
    """Yield chunk parquet paths under ``root``, optionally filtered.

    Layout: ``{root}/{year}/{tile}/{tile}_rNN_cMM.parquet``.
    """
    if not root.exists():
        return
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        if year is not None and int(year_dir.name) != year:
            continue
        for tile_dir in sorted(year_dir.iterdir()):
            if not tile_dir.is_dir():
                continue
            if tile is not None and tile_dir.name != tile:
                continue
            for path in sorted(tile_dir.glob("*.parquet")):
                if parse_chunk_stem(path.stem) is not None:
                    yield path


def verify_chunkstore(
    root: Path,
    year: int | None = None,
    tile: str | None = None,
    on_start=None,
    on_progress=None,
) -> list[ChunkReport]:
    """Verify every chunk parquet under ``root`` (with optional year/tile filter).

    ``on_start(total)`` is called once with the number of chunks to scan, and
    ``on_progress(done, report)`` after each chunk completes.  Both are optional;
    they let a caller (the CLI) drive a progress bar without coupling this module
    to any display library.
    """
    import duckdb

    paths = list(iter_chunk_parquets(root, year, tile))
    if on_start is not None:
        on_start(len(paths))

    con = duckdb.connect()
    reports: list[ChunkReport] = []
    try:
        for i, p in enumerate(paths, 1):
            rep = verify_chunk(p, con=con)
            reports.append(rep)
            if on_progress is not None:
                on_progress(i, rep)
        return reports
    finally:
        con.close()
