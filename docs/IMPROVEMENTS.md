# Improvements

## Stage 5: Sparse ALA training data in the Mitchell catchment

The ALA database contains only ~16 georeferenced *Parkinsonia aculeata* records within
the catchment bounding box. This is insufficient for reliable Random Forest training —
the model will technically run but cross-validation accuracy is unreliable and the
classifier is likely to be overfit or noisy.

Expanding the bbox to pull more records from broader Australia is not a valid fix,
because the other signals (NDVI anomaly, flowering index, distance to watercourse) are
only computed for the catchment — there would be no feature data to pair with distant
occurrence records.

### Partial mitigation implemented

`06_classifier.py` now supplements the ALA records with **ecological bootstrap
samples** — high-confidence synthetic presences and absences derived from known
Parkinsonia biology:

- **Synthetic presences**: within 500m of watercourse + NDVI anomaly in top quartile
  + flowering index above scene median + moderate NDVI median (0.15–0.65)
- **Synthetic absences**: beyond 2km from water, or bare ground, or low anomaly +
  low flowering + beyond 500m from water
- Synthetic samples are down-weighted at 0.3 vs 1.0 for real ALA records, so ground
  truth anchors the model

This brings the training set from ~16 to ~600+ samples but introduces circularity —
the features used to generate training labels are the same features the model learns
from. The model may become overconfident in the ecological prior rather than
discovering new signal.

### Remaining fixes needed

**Option 1 — Extend the GIS analysis to ecologically comparable areas**

Run stages 1–4 over additional catchments or regions with similar ecology (tropical
savanna, riparian corridors) where ALA has denser Parkinsonia sightings. Train the
Random Forest on the combined multi-region dataset, then apply it to the Mitchell
catchment. This is the most principled fix as it preserves the integrity of the
feature-label pairing.

**Option 2 — Supplement with manually collected presence records**

Incorporate field survey GPS points or manually digitised presence polygons from
land managers, NRM groups, or remote sensing interpretation. These can be merged with
the ALA records before training. A GeoPackage drop-in at
`{CACHE_DIR}/manual_occurrences.gpkg` would be a natural integration point in
`fetch_ala_occurrences.py`.
