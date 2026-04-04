# Science Validation Report
Generated: 2026-04-04T06:16:54Z
Commit: 33e6a69
Points: 40 presence, 40 absence

## Signal 1: Flowering flush
SKIP — insufficient data (presence n=0, absence n=0)

## Signal 1: Flowering flush (per year)
  2021: SKIP (presence n=0, absence n=0)
  2022: SKIP (presence n=0, absence n=0)
  2025: SKIP (presence n=0, absence n=0)

## Signal 2a: Dry-season NDVI mean
Hypothesis: presence has higher mean dry-season NDVI than absence (DOY 182–273)
  Presence median: 0.1597  (n=40)
  Absence median:  0.1458  (n=40)
  Mann-Whitney U:  p=0.0447  PASS

## Signal 2b: Dry-season NDVI inter-annual variance
Hypothesis: presence has lower inter-annual NDVI variance than absence
  Presence median var: 0.000529  (n=40)
  Absence median var:  0.000509  (n=40)
  Mann-Whitney U:      p=0.1291  FAIL  WARNING: direction reversed

## Signal 3a: Peak DOY clustering
SKIP — insufficient presence peak DOYs (n=0, need ≥6)

## Signal 3b: Inter-annual peak DOY SD
SKIP — insufficient data (presence n=0, absence n=0)

## Signal 4: Quality weighting false-peak rate
INFO — no raw false peaks detected on 40 absence points; quality filter effect cannot be measured
