"""tam/experiments/v5_spectral.py — Spectral bands + NDVI/NDWI/EVI indices (V5).

Extends V4. Changes:
- band_noise_std=0 (band jitter removed — not shown to help, adds noise)
- use_n_obs=True retained (obs count appended to pooled repr)
"""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v5_spectral",
    region_ids=[
        "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
        # "lake_mueller_presence_mixed",  # removed — noisy labels (Parkinsonia mixed with native canopy)
        "lake_mueller_absence",
        "barcoorah_presence", "barcoorah_absence_lake", "barcoorah_absence_woodland",
        "frenchs_presence_1", "frenchs_presence_2", "frenchs_presence_3",
        "frenchs_presence_4", "frenchs_presence_5", "frenchs_presence_6",
        "frenchs_absence_bare_soil_1", "frenchs_absence_bare_soil_2", "frenchs_absence_bare_soil_3",
        "frenchs_absence_mangrove", "frenchs_absence_ocean",
        "frenchs_absence_riparian_woodland", "frenchs_absence_riparian", "frenchs_absence_savanna",
        "frenchs_absence_water_1", "frenchs_absence_water_2", "frenchs_absence_water_3",
        "maria_downs_presence", "maria_downs_absence",
        "mitchell_river_absence",
        # "nardoo_presence",  # removed — presence-only site, no local absence
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_absence_1", "norman_road_absence_2",
        "quaids_absence_sparse_shrub_1", "quaids_absence_sparse_shrub_2",
        "quaids_absence_sparse_eucalypt_1",
        "quaids_absence_shrub_1", "quaids_absence_shrub_2", "quaids_absence_shrub_3",
        "quaids_absence_very_sparse_1",
        "quaids_absence_dense_eucalypt_1",
        "nassau_presence_1", "nassau_presence_2", "nassau_presence_3", "nassau_presence_4",
        "nassau_absence_1", "nassau_absence_2", "nassau_absence_3",
        "pormpuraaw_presence_1", "pormpuraaw_presence_2", "pormpuraaw_presence_3",
        "pormpuraaw_absence_1", "pormpuraaw_absence_2",
        "stockholm_presence_1", "stockholm_presence_2",
        "stockholm_absence_1",
        "muttaburra_absence_1", "muttaburra_absence_2",
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={},
    train_kwargs={"band_noise_std": 0.0, "val_sites": ("pormpuraaw", "stockholm")},
)
