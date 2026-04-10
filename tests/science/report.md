# Science Validation Report
Generated: 2026-04-06T14:29:28Z
Commit: ebe1a0e
Points: 40 presence, 40 absence

## Signal 1: Flowering flush
SKIP — insufficient data (presence n=0, absence n=0)

## Signal 1: Flowering flush (per year)
  2021: SKIP (presence n=0, absence n=0)
  2022: SKIP (presence n=0, absence n=0)
  2025: SKIP (presence n=0, absence n=0)

## Signal 2a: Dry-season NDVI mean
Hypothesis: presence has higher mean dry-season NDVI than absence (DOY 182–273)
  Presence median: 0.2748  (n=40)
  Absence median:  0.5153  (n=40)
  Mann-Whitney U:  p=1.0000  FAIL  WARNING: direction reversed

## Signal 2b: Dry-season NDVI inter-annual variance
Hypothesis: presence has lower inter-annual NDVI variance than absence
  Presence median var: 0.001836  (n=40)
  Absence median var:  0.001661  (n=40)
  Mann-Whitney U:      p=0.7919  FAIL  WARNING: direction reversed

## Signal 3a: Peak DOY clustering
SKIP — insufficient presence peak DOYs (n=0, need ≥6)

## Signal 3b: Inter-annual peak DOY SD
SKIP — insufficient data (presence n=0, absence n=0)

## Signal 4: Quality weighting false-peak rate
Hypothesis: quality-weighted detection produces fewer false peaks than raw max
  Absence points tested:   40
  Raw false peaks:         26
  Weighted false peaks:    0
  Reduction:               26
  Binomial p:              0.0000  PASS
