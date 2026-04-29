"""v8_s1_pure — Pure SAR model: S1 time series + S1 global features, no S2.

S1 unsnapped at its own acquisition dates (4 bands: VH dB, VV dB, VH-VV, RVI).
S1 globals only (n_global_features=4, which slices the first 4 columns of
GLOBAL_FEATURE_NAMES — s1_mean_vh_dry, s1_vh_contrast, s1_vh_std, s1_mean_rvi).
No S2 bands, no S2 globals, no SCL filtering.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_pure",
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
        "d_ff":     64,
        "n_bands":  4,
        "n_global_features": 4,
    },
    train_kwargs={
        "lr":              1e-5,
        "weight_decay":    0.1,
        "n_epochs":        60,
        "patience":        10,
        "band_noise_std":  0.0,
        "obs_dropout_min": 4,
        "doy_density_norm": True,
        "use_s1":          "s1_only",
        "warmup_freeze_epochs": 10,
    },
)
