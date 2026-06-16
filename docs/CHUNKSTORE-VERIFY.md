# Chunkstore Verification — `cli/chunk.py verify`

Integrity checker for the per-chunk scoring parquets under
`{CHUNKSTORE_DIR}/{year}/{tile}/{tile}_rNN_cMM.parquet`.

Its headline job is to catch the **S1 truncation defect** (thin-strip scoring; see
[S1-COVERAGE.md](S1-COVERAGE.md)), but it also flags empty, unreadable, and S1-missing
chunks. Use it before trusting a tile for scoring and as a gate before/after a rebuild.

Core logic lives in [`utils/chunk_verify.py`](../utils/chunk_verify.py) (pure, no network —
unit-tested in [`tests/unit/test_chunk_verify.py`](../tests/unit/test_chunk_verify.py));
the CLI in [`cli/chunk.py`](../cli/chunk.py) is a thin wrapper.

---

## Usage

```
python cli/chunk.py verify [--year YYYY] [--tile TILE] [--root DIR] [--all] [--no-progress]
```

| Flag | Default | Meaning |
|---|---|---|
| `--year YYYY` | all years | Limit to one year directory. |
| `--tile TILE` | all tiles | Limit to one MGRS tile id (e.g. `55KCB`). |
| `--root DIR` | `$CHUNKSTORE_DIR` (from `.env`), else `/mnt/external/chunkstore` | Chunkstore root to scan. |
| `--all` | off | Print **every** chunk row, not just failures. (The issue list always shows all failures regardless.) |
| `--no-progress` | off | Disable the live progress bar. |

### Progress bar

A live progress bar (via `tqdm`) is shown on **stderr** while scanning, so it does not
pollute the results table or exit-code gating on stdout:

```
Verifying chunks:  52%|█████▏    | 372/711 [04:04<03:42, 1.52chunk/s, fail=361]
```

It reports completed/total chunks, percent, elapsed, ETA, rate, and a running `fail`
count. It is **automatically suppressed when stderr is not a TTY** (e.g. redirected to a
log file or piped), and can be forced off with `--no-progress`.

### Examples

```bash
# Whole chunkstore
python cli/chunk.py verify

# One year
python cli/chunk.py verify --year 2025

# One tile, all years
python cli/chunk.py verify --tile 55KCB

# One tile in one year, showing every chunk (not just failures)
python cli/chunk.py verify --year 2025 --tile 55KCB --all

# Explicit root (e.g. an archive mount)
python cli/chunk.py verify --root /mnt/gis-archive/chunkstore --year 2025
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | All scanned chunks OK (or no chunks matched the filter). |
| `1` | At least one chunk has an issue. |
| `2` | `--root` does not exist. |

Because a failed scan returns `1`, the command can gate a rebuild in a shell pipeline:

```bash
python cli/chunk.py verify --year 2025 --tile 55KCB || echo "rebuild needed"
```

---

## What it checks

For each chunk parquet, `verify` computes a few cheap aggregates (via DuckDB, directly
over the parquet) and raises a tagged issue when a value looks wrong.

| Tag | Trigger | Meaning |
|---|---|---|
| `UNREADABLE` | parquet fails to open / query | Corrupt or truncated file. |
| `EMPTY` | zero distinct pixels | No data in the chunk. |
| `S1_MISS` | zero S1 observations | Chunk has no Sentinel-1 rows at all. |
| `S1_TRUNC` | median per-date S1 row coverage `< 0.5` | **The truncation defect** — most S1 acquisitions only populate a thin band of pixel rows, so most pixels never reach the scorer's S1 obs threshold. Rebuild needed. |

### The key metric: per-date S1 row coverage

A healthy Sentinel-1 IW/GRD acquisition is a ~250 km scene that fully covers any chunk,
so **most S1 dates should cover all of the chunk's pixel rows**. The check is:

```
s1_med_frac = median over S1 dates of
                (distinct yi pixel-rows that date covers) / (chunk's total yi rows)
```

- **Healthy chunk:** `s1_med_frac ≈ 1.0` — nearly every date covers the whole chunk.
- **Truncated chunk:** `s1_med_frac ≈ 0.10` — most dates cover only a thin band.

`s1_max_frac` (the best single date) is reported alongside. When it is near `1.0` it
proves the chunk *could* be fully covered, so a low median is unambiguously a defect
rather than a genuine satellite-swath edge. A chunk is flagged `S1_TRUNC` when
`s1_med_frac < 0.5` (`S1_TRUNCATION_MED_FRAC` in `utils/chunk_verify.py`).

---

## Output columns

```
  YEAR  TILE   CHUNK       PIXELS  S1_DATES  MED_FRAC  MAX_FRAC  %>=4OBS  STATUS
  ------------------------------------------------------------------------------
  2025  55KCB  r01_c00  1,048,576        30      0.09      0.56      9.4  FAIL
  ...

  37 chunk(s) checked — 0 OK, 37 with issues.

    ! S1_TRUNC    55KCB_r01_c00 — median S1 date covers only 9% of rows (max 56%);
      only 9.4% of pixels reach >=4 obs — patches were truncated, rebuild needed
```

| Column | Meaning |
|---|---|
| `YEAR` / `TILE` / `CHUNK` | Location of the chunk parquet. |
| `PIXELS` | Distinct pixels (`point_id`) in the chunk. |
| `S1_DATES` | Number of distinct Sentinel-1 acquisition dates. |
| `MED_FRAC` | `s1_med_frac` — the key metric above. |
| `MAX_FRAC` | `s1_max_frac` — best single date's row coverage. |
| `%>=4OBS` | Percent of pixels with ≥ `MIN_S1_OBS_PER_YEAR` (=4) S1 observations — i.e. the fraction the scorer will actually keep. |
| `STATUS` | `OK` or `FAIL`. |

By default the table lists only failing chunks; the issue list beneath it always
enumerates every failure. Pass `--all` to print every chunk row.

---

## Remediation when a chunk fails

`S1_TRUNC` / `S1_MISS` chunks must be **rebuilt**. Delete each chunk's per-chunk
`s1_cache/*.npz` first so any cached truncated patches are not reused, then re-run the
fetch pipeline for that tile/year. The write-time guard in
[`utils/fetch.py`](../utils/fetch.py) `_read_bbox_patch` now rejects short reads, so a
rebuild will not re-introduce the defect. See [S1-COVERAGE.md](S1-COVERAGE.md) for the
full root-cause analysis.
