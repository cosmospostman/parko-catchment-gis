"""v9_spectral — S2-only TAM, same site set as v8_roper.

S2-only leg of the three-way comparison (S2-only / S1-only / joint S1+S2).
Feature set: B02 B03 B04 B05 B07 B08 B8A B11 B12 + NDVI + NDWI (11 bands).
B06 and EVI excluded. Pixel z-score normalisation identical to V8.
"""

from tam.core.dataset import V9_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v9_spectral",
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
        # Roper River
        "roper_presence_1", "roper_presence_2", "roper_presence_3",
        "roper_presence_4",
        "roper_absence_1", "roper_absence_2", "roper_absence_3",
    ],
    feature_cols=V9_FEATURE_COLS,
    model_kwargs={
        "d_model":  128,
        "n_layers": 2,
        "dropout":  0.5,
        "n_bands":  len(V9_FEATURE_COLS),  # 11
        "n_global_features": 0,            # overridden at runtime when use_band_summaries=True
    },
    train_kwargs={
        "lr":                    5e-5,
        "weight_decay":          0.1,
        "n_epochs":              60,
        "patience":              15,
        "band_noise_std":        0.03,
        "obs_dropout_min":       4,
        "doy_density_norm":      True,
        "doy_phase_shift":       True,
        "pixel_zscore":          True,
        "use_s1":                False,
        "val_sites":             ("etna",),
        "feature_cols_override": tuple(V9_FEATURE_COLS),
    },
)
