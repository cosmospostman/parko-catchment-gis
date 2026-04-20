"""tam/experiments/v1_spectral.py — Spectral bands + NDVI/NDWI/EVI indices."""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v1_spectral",
    region_ids=[
        "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
        # "lake_mueller_presence_mixed",  # removed — noisy labels (Parkinsonia mixed with native canopy)
        "lake_mueller_absence",
        "barcoorah_presence", "barcoorah_absence_lake", "barcoorah_absence_woodland",
        "frenchs_presence_1", "frenchs_presence_2",
        "frenchs_absence_mangrove", "frenchs_absence_ocean",
        "frenchs_absence_riparian_woodland", "frenchs_absence_riparian", "frenchs_absence_savanna",
        "maria_downs_presence", "maria_downs_absence",
        "mitchell_river_absence",
        # "nardoo_presence",  # removed — presence-only site, no local absence
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_absence_1", "norman_road_absence_2",
        # "rockhampton_presence_1", "rockhampton_presence_2",  # removed — uncertain labels (swamp context)
        # stonehenge removed — patches too small (<100px each)
        # "stonehenge_presence1", "stonehenge_presence_2", "stonehenge_presence_3",
        # "stonehenge_absence1", "stonehenge_absence_2", "stonehenge_absence_3",
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={},
    train_kwargs={},
)
