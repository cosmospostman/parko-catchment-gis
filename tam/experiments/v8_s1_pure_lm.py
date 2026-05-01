"""v8_s1_pure_lm — S1 temporal + S1 globals, Lake Mueller as transfer val site.

Same as v8_s1_pure but Lake Mueller pixels are held out entirely for validation
(val_sites split) to test zero-shot site transferability.
Train: Norman Road + Cloncurry. Val: Lake Mueller.
Warmup freeze: head trains on globals for 10 epochs before temporal stream unfreezes.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_pure_lm",
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
        "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
        "lake_mueller_absence", "lake_mueller_absence_2",
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
        "val_sites":       ("lake_mueller",),
    },
)
