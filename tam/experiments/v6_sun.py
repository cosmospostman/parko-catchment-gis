"""tam/experiments/v6_sun.py — Region list matching the tam-v6-0.735-sun checkpoint.

Reconstructed from train.log pixel summary. Val site: pormpuraaw.
Architecture: d_model=32, n_layers=1 (differs from current v6_spectral defaults).
"""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v6_sun",
    region_ids=[
        "barcoorah_presence", "barcoorah_presence_2", "barcoorah_presence_3",
        "barcoorah_absence_lake", "barcoorah_absence_woodland",
        "frenchs_presence_1", "frenchs_presence_2", "frenchs_presence_3",
        "frenchs_presence_4", "frenchs_presence_5", "frenchs_presence_6",
        "frenchs_absence_bare_soil_2", "frenchs_absence_bare_soil_3",
        "frenchs_absence_mangrove", "frenchs_absence_ocean",
        "frenchs_absence_riparian_woodland", "frenchs_absence_riparian", "frenchs_absence_savanna",
        "frenchs_absence_water_1", "frenchs_absence_water_2", "frenchs_absence_water_3",
        "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
        "lake_mueller_absence",
        "maria_downs_presence", "maria_downs_absence",
        "mitchell_river_absence",
        "nassau_presence_1", "nassau_presence_2", "nassau_presence_3", "nassau_presence_4",
        "nassau_absence_1", "nassau_absence_2", "nassau_absence_3",
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_absence_1", "norman_road_absence_2",
        "quaids_absence_sparse_shrub_1", "quaids_absence_sparse_shrub_2",
        "quaids_absence_sparse_eucalypt_1",
        "quaids_absence_shrub_1", "quaids_absence_shrub_2", "quaids_absence_shrub_3",
        "quaids_absence_very_sparse_1",
        "quaids_absence_dense_eucalypt_1",
        "stockholm_presence_1", "stockholm_presence_2",
        "stockholm_absence_1",
        "wongalee_presence_1", "wongalee_presence_2", "wongalee_presence_3", "wongalee_presence_4",
        "wongalee_absence_1", "wongalee_absence_2",
        "pormpuraaw_presence_1", "pormpuraaw_presence_2", "pormpuraaw_presence_3",
        "pormpuraaw_absence_1", "pormpuraaw_absence_2",
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={
        "d_model":  32,
        "n_layers": 1,
        "dropout":  0.3,
    },
    train_kwargs={
        "val_sites": ("pormpuraaw",),
    },
)
