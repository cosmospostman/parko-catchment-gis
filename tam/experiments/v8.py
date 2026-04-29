"""tam/experiments/v8.py — Norman Road with S1 time-series + global features.

First experiment using Sentinel-1 SAR backscatter. Adds VH dB, VV dB, VH−VV,
and RVI as per-observation features (snapped to nearest S2 date ±7 days) and
4 S1 global features (s1_mean_vh_dry, s1_vh_contrast, s1_vh_std, s1_mean_rvi)
alongside existing S2 globals (9 total).

Start point: Norman Road presence + absence. Woody negative sites to be added
by the user before or after the first run.
"""

from tam.core.dataset import ALL_FEATURE_COLS, S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_nr_s1",
    region_ids=[
        # Presence
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_presence_5", "norman_road_presence_6",
        "norman_road_presence_7", "norman_road_presence_8",
        "norman_road_presence_9",
        # Absence
        "norman_road_absence_1", "norman_road_absence_2",
        "norman_road_absence_3", "norman_road_absence_4",
        "norman_road_absence_5",
        "norman_road_absence_7",
        # Woody negatives — Cloncurry woodland (~100km SW, same climate zone)
        "cloncurry_absence_1", "cloncurry_absence_2", "cloncurry_absence_3",
        "cloncurry_absence_4", "cloncurry_absence_5", "cloncurry_absence_6",
        "cloncurry_absence_7",
    ],
    feature_cols=ALL_FEATURE_COLS + S1_FEATURE_COLS,
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
        "band_noise_std":  0.03,
        "obs_dropout_min": 6,
        "doy_density_norm": True,
        "use_s1":          True,
    },
)
