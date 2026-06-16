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
| `S1_INCOMPLETE` | `< 95%` of S1 dates cover `>= 95%` of rows | **The completeness defect** — S1 rows are missing on many acquisition dates. Rebuild needed. |

### The key metric: per-date completeness

A healthy Sentinel-1 IW/GRD acquisition is a ~250 km scene that fully covers any chunk,
so **every S1 date should cover ~all of the chunk's pixel rows**. The strict check is:

```
complete_date_frac = (# S1 dates covering >= 95% of the chunk's rows) / (# S1 dates)
```

- **Complete chunk:** `complete_date_frac ≈ 1.0` — essentially every date covers the whole chunk.
- **Damaged chunk:** `complete_date_frac` low — many dates are missing rows.

A chunk is flagged `S1_INCOMPLETE` when `complete_date_frac < 0.95`
(`S1_MIN_COMPLETE_DATE_FRAC`; per-date threshold `S1_DATE_COMPLETE_FRAC = 0.95`).

**Why not the median?** Earlier versions flagged on `s1_med_frac < 0.5` (median per-date
coverage). That was too lenient: a chunk can have median coverage well above 0.5 while
still missing rows on many dates — it would "pass verify but still be missing data". The
`merge_scenes` northing-band bug (fixed in `_sort_s1_shards`) dropped rows on every date
*after the first*, so the complete-date fraction is the reliable completeness signal.
`s1_med_frac` is still reported (column `MED_FRAC`) for context.

A tiny edge chunk where every pixel still reaches `MIN_S1_OBS_PER_YEAR` obs (so the scorer
keeps it all) and `s1_med_frac < 0.5` is reported as a non-failing `S1_TRUNC_OK` note
rather than a failure.

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
