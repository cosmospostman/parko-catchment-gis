"""v8_roper — adds Roper River regions to the full v8 training set.

Extends v8_s1_zscore_nr_etna_landsend_lm_corfield with:
  - Roper River (tropical, riparian) — 4 presence + 3 absence
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_roper",
    region_ids=[
        # Norman Road
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_presence_5", "norman_road_presence_6",
        "norman_road_presence_7", "norman_road_presence_8",
        "norman_road_presence_9",
        "norman_road_absence_1", "norman_road_absence_2",
        "norman_road_absence_3", "norman_road_absence_4",
        "norman_road_absence_5", "norman_road_absence_7",
        # Cloncurry
        "cloncurry_absence_1", "cloncurry_absence_2", "cloncurry_absence_3",
        "cloncurry_absence_4", "cloncurry_absence_5", "cloncurry_absence_6",
        "cloncurry_absence_7",
        # Etna Creek (held out at sweep level via val_sites=("etna",))
        "etna_presence_1", "etna_presence_2", "etna_presence_3", "etna_presence_4",
        "etna_absence_1", "etna_absence_2", "etna_absence_3", "etna_absence_4",
        "etna_absence_5",
        # Landsend
        "landsend_presence_1", "landsend_presence_2", "landsend_presence_3",
        "landsend_presence_4",
        "landsend_absence_1", "landsend_absence_2", "landsend_absence_3",
        # Lake Mueller
        "lake_mueller_presence", "lake_mueller_presence_2",
        "lake_mueller_presence_3", "lake_mueller_presence_4",
        "lake_mueller_absence", "lake_mueller_absence_2",
        "lake_mueller_absence_3",
        # Corfield
        "corfield_presence_1", "corfield_presence_2", "corfield_presence_3",
        "corfield_presence_4", "corfield_presence_5",
        "corfield_absence_1", "corfield_absence_2", "corfield_absence_3",
        # Roper River (tropical, riparian)
        "roper_presence_1", "roper_presence_2", "roper_presence_3",
        "roper_presence_4",
        "roper_absence_1", "roper_absence_2", "roper_absence_3",
    ],
    feature_cols=S1_FEATURE_COLS,
    model_kwargs={
        "d_model":  128,
        "n_layers": 2,
        "dropout":  0.5,
        "n_bands":  4,
        "n_global_features": 0,
    },
    train_kwargs={
        "lr":              5e-5,
        "weight_decay":    0.1,
        "n_epochs":        60,
        "patience":        15,
        "band_noise_std":  0.0,
        "obs_dropout_min": 4,
        "doy_density_norm": True,
        "doy_phase_shift": True,
        "pixel_zscore":    True,
        "use_s1":          "s1_only",
        "val_sites":       ("etna",),
    },
)
