# EBS local cache plan — stages 01 and 03

## Overview

Three new files alongside the existing pipeline:

1. `scripts/s2_sync_manifest.py` — STAC search → writes a manifest of S3 keys to sync
2. `scripts/s2_ebs_sync.sh` — reads manifest, runs `aws s3 cp` per file, reports size
3. A small addition to `utils/stac.py` — `rewrite_hrefs_to_local()` helper
4. A one-line change each in `01_ndvi_composite.py` and `03_flowering_index.py` — conditionally patch items if `LOCAL_S2_ROOT` is set
5. A one-line addition to `config.sh` — export `LOCAL_S2_ROOT` when the volume is mounted

No changes to `run_tiled_pipeline`, `config.py`, or any other file.

---

## Step 1 — `scripts/s2_sync_manifest.py`

Runs a STAC search covering **both** stage 01's composite window (May–Oct) and stage 03's
flowering window (Aug–Oct, which is a subset). Collects the S3 asset hrefs for `red`, `nir`,
`scl`, `green`, `rededge1`, `rededge2` (all bands needed by either stage). Writes one S3 URI
per line to `manifest.txt`.

```
usage: python scripts/s2_sync_manifest.py YEAR [--out manifest.txt]
```

Internally: calls `search_sentinel2` once with the wider May–Oct window (flowering is a
subset), deduplicates, extracts asset hrefs for the required bands, writes the file.

The manifest is a stable artifact — commit it or store it with the run outputs so you know
exactly what was synced.

---

## Step 2 — `scripts/s2_ebs_sync.sh`

```
usage: ./scripts/s2_ebs_sync.sh --manifest manifest.txt --dest /mnt/s2cache
```

- Reads the manifest line by line
- Converts each `s3://sentinel-cogs/PATH` URI to an `aws s3 cp` call, preserving the key
  path under `--dest`
- Runs copies in parallel (xargs -P 16)
- Prints a summary: files synced, total GB, elapsed time
- Idempotent — skips files already present at the destination (checks existence before
  copying)

Why `cp` per-file from manifest rather than `aws s3 sync` with a prefix? Because `sync` on a
broad prefix would pull everything in that prefix — the manifest gives surgical control over
exactly the 30 granules needed.

---

## Step 3 — `utils/stac.py` addition

```python
def rewrite_hrefs_to_local(items, local_root, bands):
    """Replace S3/HTTPS asset hrefs with local file paths where files exist."""
    import copy, urllib.parse
    from pathlib import Path
    patched = []
    for item in items:
        item = copy.deepcopy(item)
        for band, asset in item.assets.items():
            if band not in bands:
                continue
            parsed = urllib.parse.urlparse(asset.href)
            # s3://sentinel-cogs/a/b/c.tif → /mnt/s2cache/sentinel-cogs/a/b/c.tif
            local_path = Path(local_root) / parsed.netloc / parsed.path.lstrip("/")
            if local_path.exists():
                asset.href = str(local_path)
        patched.append(item)
    return patched
```

Falls back silently for any file not found locally — stackstac will fetch that one from S3
as normal. Safe to call even with a partial cache.

---

## Step 4 — Pipeline changes (minimal)

**`01_ndvi_composite.py`** — after the STAC search, before `fetch_fn`:

```python
if local_root := os.environ.get("LOCAL_S2_ROOT"):
    from utils.stac import rewrite_hrefs_to_local
    items = rewrite_hrefs_to_local(items, local_root, load_bands)
    logger.info("LOCAL_S2_ROOT set — hrefs rewritten to local paths")
```

**`03_flowering_index.py`** — identical pattern, same three lines.

Also drop `FETCH_WORKERS` default from 32 → 4 when `LOCAL_S2_ROOT` is set — local disk
doesn't benefit from 32 concurrent readers:

```python
FETCH_WORKERS = int(os.environ.get("FETCH_WORKERS", "4" if os.environ.get("LOCAL_S2_ROOT") else "32"))
```

---

## Step 5 — `config.sh` addition

```bash
# Set this when the S2 COG EBS volume is mounted at /mnt/s2cache
export LOCAL_S2_ROOT="${LOCAL_S2_ROOT:-}"
```

When the volume is not mounted the variable is empty, `rewrite_hrefs_to_local` is never
called, and the pipeline behaves exactly as today. No flag, no mode — just presence/absence
of the env var.

---

## Operational workflow (annual)

```
1. Restore EBS snapshot from prior year (or create fresh 500GB gp3)
2. Mount at /mnt/s2cache
3. python scripts/s2_sync_manifest.py 2025 --out manifest_2025.txt
4. ./scripts/s2_ebs_sync.sh --manifest manifest_2025.txt --dest /mnt/s2cache
   (~30-60 min, ~200-500GB)
5. export LOCAL_S2_ROOT=/mnt/s2cache   (or set in config.sh)
6. ./run.sh 2025
   (stages 01 and 03 read locally; stages 02, 04-07 unchanged)
7. Snapshot the EBS volume
8. Detach and delete the volume
```

For year 2, step 4 only syncs new scenes — the snapshot already has the prior year's data.
Existence checks in the sync script make this incremental automatically.

---

## Expected performance impact

| | Remote S3 | Local EBS |
|---|---|---|
| fetch_fn time | 65–770s | ~1–2s |
| bottleneck | network | compute_fn (~1s) |
| FETCH_WORKERS | 32 | 4 |
| stages 01 + 03 total | ~6–7h | ~10–15 min |

---

## What does not change

- `run.sh` — no changes
- `config.py` — no changes
- `utils/pipeline.py` — no changes
- All other analysis scripts — no changes
- Test suite — `LOCAL_S2_ROOT` not set in tests, existing mocks unaffected
