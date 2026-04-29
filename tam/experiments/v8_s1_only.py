"""v8_s1_only — S1 time series only, no S2 bands, no snapping.

S1 observations are used directly at their own acquisition dates (6-12 day
revisit). Bands: VH dB, VV dB, VH-VV, RVI (4 bands). No S2 bands, no SCL
cloud filter (S1 penetrates cloud). No global features.

This is a fair test of S1's temporal discriminative power independent of S2.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_only",
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
    feature_cols=S1_FEATURE_COLS,   # 4 S1 bands only
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.5,
        "n_bands":  4,
        "n_global_features": 0,
    },
    train_kwargs={
        "lr":              1e-5,
        "weight_decay":    0.1,
        "n_epochs":        60,
        "patience":        10,
        "band_noise_std":  0.0,   # don't add noise to SAR bands
        "obs_dropout_min": 4,     # S1 has fewer obs/year than S2
        "doy_density_norm": True,
        "use_s1":          "s1_only",
    },
)
