"""tam/experiments/v7_norman_road_only.py — Diagnostic: Norman Road only.

Tests whether the model can learn Parkinsonia signal in a Gulf savanna context
against local absence. Counterpart to v7_frenchs_only.
Val site: alexandria — semi-arid riparian, same broad Gulf/semi-arid cluster.
"""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v7_norman_road_only",
    region_ids=[
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_presence_5", "norman_road_presence_6",
        "norman_road_presence_7", "norman_road_presence_8",
        "norman_road_presence_9",
        "norman_road_absence_1", "norman_road_absence_2",
        "norman_road_absence_3", "norman_road_absence_4",
        "norman_road_absence_5", "norman_road_absence_6", "norman_road_absence_7",
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.5,
    },
    train_kwargs={
        "lr":              1e-5,
        "weight_decay":    0.1,
        "n_epochs":        60,
        "patience":        10,
        "spatial_stride":    1,
        "band_noise_std":    0.03,
        "obs_dropout_min":   6,
        "doy_density_norm":  True,
    },
)
