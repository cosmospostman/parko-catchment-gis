"""scripts/build_science_points.py — generate science test point set.

Samples presence and absence points from manually-defined zones derived from
Queensland Globe imagery of a known Parkinsonia infestation near the mouth of
the Mitchell River, Gulf Plains QLD (141.59E, 14.66S).

Zones were digitised from a georeferenced Queensland Globe export
(EPSG:3857, pixel size ~8.1m):

  Presence  — core of the dark infestation patch
              141.5856–141.5960E, 14.6549–14.6659S

  Absence 1 — open floodplain / light patch north of the infestation
              141.5594–141.5960E, 14.6390–14.6479S

  Absence 2 — small light patch in the coastal strip south-west of core
              141.5660–141.5762E, 14.6603–14.6670S

Points are sampled on a regular grid at ~100m spacing (0.0009° ≈ 100m),
then jittered slightly so they don't all fall on the same Sentinel-2 pixel.

Output: tests/fixtures/science_points.csv  (point_id, lon, lat, label)

Usage:
    python scripts/build_science_points.py [--n-presence N] [--n-absence N] [--seed S]
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Zone definitions  (lon_min, lon_max, lat_min, lat_max)  WGS84
# ---------------------------------------------------------------------------

PRESENCE_ZONE = (141.58561, 141.59600, -14.66591, -14.65494)

ABSENCE_ZONES = [
    (141.55941, 141.59600, -14.64786, -14.63901),  # north merged
    (141.56600, 141.57624, -14.66697, -14.66025),  # bottom-left
]

# Grid spacing ~100m
GRID_SPACING = 0.0009  # degrees
# Max jitter: ±30m
JITTER = 0.00027

OUTPUT_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "science_points.csv"


def _grid_points(zone: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Return all grid points within the zone bounding box."""
    lon_min, lon_max, lat_min, lat_max = zone
    points = []
    lon = lon_min + GRID_SPACING / 2
    while lon < lon_max:
        lat = lat_min + GRID_SPACING / 2
        while lat < lat_max:
            points.append((lon, lat))
            lat += GRID_SPACING
        lon += GRID_SPACING
    return points


def _sample_zone(
    zone: tuple[float, float, float, float],
    n: int,
    rng: random.Random,
) -> list[tuple[float, float]]:
    """Sample up to n jittered grid points from a zone."""
    candidates = _grid_points(zone)
    chosen = rng.sample(candidates, min(n, len(candidates)))
    return [
        (lon + rng.uniform(-JITTER, JITTER), lat + rng.uniform(-JITTER, JITTER))
        for lon, lat in chosen
    ]


def build(n_presence: int, n_absence: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    # Presence
    presence_pts = _sample_zone(PRESENCE_ZONE, n_presence, rng)

    # Absence — sample proportionally across zones by area
    zone_areas = [
        (z, (z[1] - z[0]) * (z[3] - z[2]))
        for z in ABSENCE_ZONES
    ]
    total_area = sum(a for _, a in zone_areas)
    absence_pts: list[tuple[float, float]] = []
    remaining = n_absence
    for i, (zone, area) in enumerate(zone_areas):
        if i == len(zone_areas) - 1:
            n = remaining
        else:
            n = round(n_absence * area / total_area)
            remaining -= n
        absence_pts.extend(_sample_zone(zone, n, rng))

    rows = []
    for i, (lon, lat) in enumerate(presence_pts, start=1):
        rows.append({"point_id": f"pres_{i:03d}", "lon": round(lon, 6), "lat": round(lat, 6), "label": 1})
    for i, (lon, lat) in enumerate(absence_pts, start=1):
        rows.append({"point_id": f"abs_{i:03d}", "lon": round(lon, 6), "lat": round(lat, 6), "label": 0})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-presence", type=int, default=40)
    parser.add_argument("--n-absence",  type=int, default=40)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    rows = build(args.n_presence, args.n_absence, args.seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["point_id", "lon", "lat", "label"])
        writer.writeheader()
        writer.writerows(rows)

    n_pres = sum(1 for r in rows if r["label"] == 1)
    n_abs  = sum(1 for r in rows if r["label"] == 0)
    print(f"Written {len(rows)} points ({n_pres} presence, {n_abs} absence) → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
