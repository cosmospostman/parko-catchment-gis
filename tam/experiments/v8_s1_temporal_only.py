"""v8_s1_temporal_only — S1+S2 time series, no global features.

Same sites as v8. use_s1=True (17 bands) but n_global_features=0
so the model relies entirely on the temporal attention stream including SAR.
"""

from tam.core.dataset import ALL_FEATURE_COLS, S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_temporal_only",
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
    feature_cols=ALL_FEATURE_COLS + S1_FEATURE_COLS,  # 17 bands
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.5,
        "n_bands":  17,
        "n_global_features": 0,
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
