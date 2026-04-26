"""tam/experiments/v6_spectral.py — Spectral bands + NDVI/NDWI/EVI indices (V6).

Extends V5. Changes:
- stockholm, lake_mueller, pormpuraaw all in training
- pormpuraaw held out (3 presence + 2 absence — balanced val set)
"""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v6_spectral",
    region_ids=[
        "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
        # "lake_mueller_presence_mixed",  # removed — noisy labels (Parkinsonia mixed with native canopy)
        "lake_mueller_absence",
        "barcoorah_presence", "barcoorah_presence_2", "barcoorah_presence_3",
        "barcoorah_absence_lake", "barcoorah_absence_woodland",
        "frenchs_presence_1", "frenchs_presence_2", "frenchs_presence_3",
        "frenchs_presence_4", "frenchs_presence_5", "frenchs_presence_6",
        "frenchs_absence_bare_soil_2", "frenchs_absence_bare_soil_3",
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
        "barkly_presence_1", "barkly_presence_2", "barkly_presence_3",
        "barkly_presence_4", "barkly_presence_5",
        "barkly_absence_1",
        "wongalee_presence_1", "wongalee_presence_2", "wongalee_presence_3", "wongalee_presence_4",
        "wongalee_absence_1", "wongalee_absence_2",
        "moroak_presence_1", "moroak_presence_2", "moroak_presence_3",
        "roper_presence_1", "roper_presence_2", "roper_presence_3", "roper_presence_4",
        "roper_absence_1",
        "ranken_river_presence_N_1", "ranken_river_presence_N_2", "ranken_river_presence_N_3",
        "ranken_river_presence_N_4", "ranken_river_presence_N_5", "ranken_river_presence_N_6",
        "ranken_river_presence_N_7", "ranken_river_presence_N_8", "ranken_river_presence_N_9",
        "ranken_river_presence_N_10", "ranken_river_presence_N_11", "ranken_river_presence_N_12",
        "ranken_river_absence_N_1", "ranken_river_absence_N_2", "ranken_river_absence_N_3",
        "ranken_river_absence_N_4", "ranken_river_absence_N_5", "ranken_river_absence_N_6",
        "ranken_river_absence_N_7", "ranken_river_absence_N_8", "ranken_river_absence_N_9",
        "alexandria_presence", "alexandria_absence",
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={},
    train_kwargs={
        "band_noise_std":       0.0,
        "val_sites":            ("pormpuraaw",),
        "stride_exclude_sites": ("barkly", "stockholm", "barcoorah", "wongalee", "ranken_river", "alexandria",),
    },
)
