"""v8_s1_only_v2 — S1 time series only, tuned hyperparams.

Same as v8_s1_only but with larger d_ff and higher lr to give the temporal
stream more capacity and faster convergence.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_only_v2",
    region_ids=[
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_presence_5", "norman_road_presence_6",
        "norman_road_presence_7", "norman_road_presence_8",
        "norman_road_presence_9",
        "norman_road_absence_1", "norman_road_absence_2",
        "norman_road_absence_3", "norman_road_absence_4",
        "norman_road_absence_5", "norman_road_absence_7",
        "cloncurry_absence_1", "cloncurry_absence_2", "cloncurry_absence_3",
        "cloncurry_absence_4", "cloncurry_absence_5", "cloncurry_absence_6",
        "cloncurry_absence_7",
    ],
    feature_cols=S1_FEATURE_COLS,
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.5,
        "d_ff":     128,
        "n_bands":  4,
        "n_global_features": 0,
    },
    train_kwargs={
        "lr":              5e-5,
        "weight_decay":    0.1,
        "n_epochs":        60,
        "patience":        10,
        "band_noise_std":  0.0,
        "obs_dropout_min": 4,
        "doy_density_norm": True,
        "use_s1":          "s1_only",
    },
)
