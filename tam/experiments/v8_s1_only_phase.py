"""v8_s1_only_phase — S1 temporal only, NR+Cloncurry, full-year phase shift augmentation.

Same as v8_s1_only but with doy_phase_shift=True. Tests whether the temporal
discriminant is calendar-time specific or generalises as curve shape.
If val_auc remains ~0.914, the signal is shape-based and transferable.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_only_phase",
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
        "n_bands":  4,
        "n_global_features": 0,
    },
    train_kwargs={
        "lr":              1e-5,
        "weight_decay":    0.1,
        "n_epochs":        60,
        "patience":        15,
        "band_noise_std":  0.0,
        "obs_dropout_min": 4,
        "doy_density_norm": True,
        "doy_phase_shift": True,
        "use_s1":          "s1_only",
    },
)
