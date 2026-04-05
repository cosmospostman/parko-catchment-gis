# Longreach Expansion — Eastern Riparian Zone

## Purpose

Extend the Longreach S2 dataset eastward to capture land-cover classes absent from the
existing 748-pixel confirmed-infestation strip:

1. **Dense native riparian woodland** — closed-canopy coolibah/eucalyptus; primary
   false-positive risk class; required for Priority 5 (LONGREACH-STAGE2.md)
2. **Isolated scattered Parkinsonia on clay** — individual crowns at low density on
   orange/pale clay substrate; the realistic detection scenario vs the dense patch used
   for training
3. **Floodplain gilgai mosaic** — transitional mixed-cover pixels between the riparian
   corridor and open grassland; tests the monotone canopy-fraction response criterion

The expansion bbox dovetails directly with the eastern edge of the existing strip
(existing strip lon_max ≈ 145.4287; expansion lon_min = 145.421 — slight overlap
intentional for continuity).

---

## Bbox

| Parameter | Value |
|-----------|-------|
| `lon_min` | 145.421 |
| `lon_max` | 145.448 |
| `lat_min` | −22.771 |
| `lat_max` | −22.758 |
| Width     | ~2.77 km |
| Height    | ~1.45 km |
| Area      | ~4.0 km² |
| S2 pixels (10 m grid) | ~39,900 |
| Estimated parquet size | ~1.9 GB |

**Derived from:** Queensland Globe JPEGW world file provided 2026-04-05, clipped outward
to nearest 0.001° to give clean round coordinates.

```
lon: [145.421, 145.448]
lat: [-22.771, -22.758]
```

---

## Landscape description

Based on the Queensland Globe 20cm imagery (viewed 2026-04-05):

- **Western zone (lon 145.421–145.430):** Dense riparian woodland. Dark, heterogeneous
  closed canopy with branching channel network visible. Crown structure irregular and
  varied — consistent with coolibah/eucalyptus, not Parkinsonia. This is the key
  native-riparian contrast class needed for Priority 5.

- **Central zone (lon 145.430–145.438):** Floodplain gilgai mosaic. Pale clay, dark
  gilgai depressions, scattered shrubs. Highly mixed pixels at 10 m resolution. Useful
  for testing the monotone canopy-fraction response.

- **Eastern zone (lon 145.438–145.448):** Open clay flats with scattered individual dark
  crowns — probable isolated Parkinsonia invading from the riparian corridor into the
  clay plain. A second drainage channel visible at the eastern edge carries additional
  riparian scrub.

---

## Relationship to existing data

The existing 748-pixel strip covers:
- `lon: [145.4213, 145.4287]`, `lat: [-22.7671, -22.7597]`
- Confirmed dense Parkinsonia infestation (362 px) + grassland/riparian proxy (386 px)

The expansion bbox shares a ~600 m longitudinal overlap with the existing strip's eastern
edge and extends the coverage 2.1 km further east. The two datasets can be merged on
`(lon, lat)` for joint analysis; any overlapping point_ids should be identical.

---

## Next step

Fetch S2 time series for this bbox using the same pipeline as the existing strip
(2020–2025 archive, same band set, same SCL quality filter). The fetch script should
produce a parquet at `data/longreach_expansion_pixels.parquet` with identical schema to
`data/longreach_pixels.parquet`.
