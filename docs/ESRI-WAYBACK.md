# ESRI Wayback — Historical Imagery

ESRI Wayback is a free, no-auth archive of World Imagery snapshots going back to 2014.
193 global releases as of May 2026, but most locations only have a handful of distinct epochs.

## Tile URL

```
https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{M}/{z}/{y}/{x}
```

`{M}` is the release's numeric ID (not the `WB_YYYY_RXX` string). Fetch the full release list from:

```
https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer?f=json
```

The response `.Selection` array contains `{ Name, M, ID }` for each release.

## Coverage at Kowanyama (141.572, -15.464)

Checked at zoom 15, tile (z=15, y=17808, x=29270). 10 distinct epochs across 193 releases:

| Date | Release ID |
|---|---|
| 2026-04-30 | WB_2026_R04 |
| 2025-04-24 | WB_2025_R04 |
| 2024-06-27 | WB_2024_R07 |
| 2023-06-29 | WB_2023_R06 |
| 2022-06-29 | WB_2022_R08 |
| 2021-09-22 | WB_2021_R09 |
| 2021-01-13 | WB_2021_R01 |
| 2020-01-08 | WB_2020_R01 |
| 2017-10-04 | WB_2017_R10 |
| 2016-08-31 | WB_2016_R09 |

Roughly annual cadence. All acquisitions fall in the May–October dry season window, consistent
with Queensland aerial/satellite collection schedules. Nothing before 2016 for this area.

## Finding unique epochs for a location

Most Wayback releases are identical at any given tile — only a small fraction have new imagery.
To find which releases are distinct, fetch the tile for each release and compare MD5 hashes;
skip releases whose hash matches the previous unique epoch.

```python
import urllib.request, hashlib

def unique_epochs(releases, z, y, x):
    prev_hash = None
    unique = []
    for name, m in releases:
        url = f'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{m}/{z}/{y}/{x}'
        with urllib.request.urlopen(url, timeout=10) as r:
            h = hashlib.md5(r.read()).hexdigest()
        if h != prev_hash:
            unique.append((name, m))
            prev_hash = h
    return unique
```

## Notes

- Free, no API key required.
- Tiles redirect (HTTP 301) — follow redirects.
- Resolution at Kowanyama appears consistent with high-res imagery (~0.3–1 m) based on tile byte
  sizes (13–23 KB per 256×256 tile), though this varies by source dataset and location.
- ESRI ToS applies; suitable for research/internal tooling.
