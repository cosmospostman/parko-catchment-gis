"""v8_s1_zscore_nr — S1 temporal, per-pixel z-score, phase shift, NR+Cloncurry only.

Baseline check for pixel_zscore=True. Direct comparison against v8_s1_only_phase
(same sites, no z-score, peaked at ~0.894). If val_auc stays similar, z-scoring
is not breaking the in-distribution signal. If it drops significantly, the model
was relying on absolute backscatter level even within NR.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_zscore_nr",
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
        "pixel_zscore":    True,
        "use_s1":          "s1_only",
    },
)
