"""tam/experiments/v2_spectral.py — Spectral bands + NDVI/NDWI/EVI indices (V2).

Extends V1 with additional Frenchs presence sites (3–5) and water absence sites (1–3)
to improve discrimination of open water, which V1 misclassified as Parkinsonia.
"""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v2_spectral",
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
        # "rockhampton_presence_1", "rockhampton_presence_2",  # removed — uncertain labels (swamp context)
        # stonehenge removed — patches too small (<100px each)
        # "stonehenge_presence1", "stonehenge_presence_2", "stonehenge_presence_3",
        # "stonehenge_absence1", "stonehenge_absence_2", "stonehenge_absence_3",
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={},
    train_kwargs={},
)
