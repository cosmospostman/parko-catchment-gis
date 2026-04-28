"""tam/experiments/v7_frenchs_only.py — Diagnostic: Frenchs training, Pormpuraaw holdout.

Pormpuraaw is same climate zone (monsoonal riparian, Cape York) with both presence
and absence — a meaningful out-of-site val set without cross-zone contamination.
New harder absence regions (frenchs_absence_4/5/6) added to reduce easy-negative bias.
"""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v7_frenchs_only",
    region_ids=[
        "frenchs_presence_1", "frenchs_presence_2", "frenchs_presence_3",
        "frenchs_presence_4", "frenchs_presence_5", "frenchs_presence_6",
        "frenchs_absence_bare_soil_2", "frenchs_absence_bare_soil_3",
        "frenchs_absence_mangrove", "frenchs_absence_ocean",
        "frenchs_absence_riparian_woodland", "frenchs_absence_riparian", "frenchs_absence_savanna",
        "frenchs_absence_water_1", "frenchs_absence_water_2", "frenchs_absence_water_3",
        "frenchs_absence_4", "frenchs_absence_5", "frenchs_absence_6",
        "pormpuraaw_presence_1", "pormpuraaw_presence_2", "pormpuraaw_presence_3",
        "pormpuraaw_absence_1", "pormpuraaw_absence_2",
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.5,
    },
    train_kwargs={
        "lr":            1e-5,
        "weight_decay":  0.1,
        "n_epochs":      60,
        "patience":      10,
        "spatial_stride":  1,
        "band_noise_std":  0.03,
        "obs_dropout_min": 6,
        "val_sites":       ("pormpuraaw",),
    },
)
