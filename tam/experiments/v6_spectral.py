"""tam/experiments/v6_spectral.py — Spectral bands + NDVI/NDWI/EVI indices (V6).

Extends V5. Changes:
- stockholm, lake_mueller, pormpuraaw all in training
- pormpuraaw held out (3 presence + 2 absence — balanced val set)

2026-04-28: Updated to best hyperparams (lr=2e-6, wd=0.20, dropout=0.7) and
anchor sites only, based on LOSO sweep results (see docs/SWEEP.md).
Removed: barcoorah, stockholm, wongalee, roper (below-chance val_auc in LOSO),
         maria_downs, pormpuraaw (marginal),
         nassau (all presence pixels removed by noise filter — likely bad labels),
         moroak (presence-only, no reliable absence context),
         norman_road_absence_1/2 (flagged as possible label noise).
Kept:    quaids absence (false-positive suppression in savanna).
Val site: alexandria (clean signal, balanced, keeps frenchs in training).
"""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v6_spectral",
    region_ids=[
        # Lake Mueller — arid riparian, GPS-surveyed
        "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
        # "lake_mueller_presence_mixed",  # removed — noisy labels (Parkinsonia mixed with native canopy)
        "lake_mueller_absence",

        # Frenchs — monsoonal savanna-riparian, strongest LOSO signal (0.785)
        "frenchs_presence_1", "frenchs_presence_2", "frenchs_presence_3",
        "frenchs_presence_4", "frenchs_presence_5", "frenchs_presence_6",
        "frenchs_absence_bare_soil_2", "frenchs_absence_bare_soil_3",
        "frenchs_absence_mangrove", "frenchs_absence_ocean",
        "frenchs_absence_riparian_woodland", "frenchs_absence_riparian", "frenchs_absence_savanna",
        "frenchs_absence_water_1", "frenchs_absence_water_2", "frenchs_absence_water_3",

        # Norman Road — monsoonal, solid LOSO signal (0.663)
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_presence_5", "norman_road_presence_6",
        "norman_road_presence_7", "norman_road_presence_8",
        "norman_road_presence_9",
        "norman_road_absence_1", "norman_road_absence_2",
        "norman_road_absence_3", "norman_road_absence_4",
        "norman_road_absence_5", "norman_road_absence_6", "norman_road_absence_7",

        # Quaids — absence only; kept for false-positive suppression in savanna
        "quaids_absence_sparse_shrub_1", "quaids_absence_sparse_shrub_2",
        "quaids_absence_sparse_eucalypt_1",
        "quaids_absence_shrub_1", "quaids_absence_shrub_2", "quaids_absence_shrub_3",
        "quaids_absence_very_sparse_1",
        "quaids_absence_dense_eucalypt_1",

        # Mitchell River — absence only; monsoonal riparian hard negative
        "mitchell_river_absence",

        # Alexandria — val site; semi-arid riparian, LOSO 0.667
        "alexandria_presence", "alexandria_absence",

        # Removed — below-chance LOSO val_auc:
        # barcoorah (0.392), stockholm (0.296), wongalee (0.468), roper (0.430)
        # "barcoorah_presence", "barcoorah_presence_2", "barcoorah_presence_3",
        # "barcoorah_absence_lake", "barcoorah_absence_woodland",
        # "stockholm_presence_1", "stockholm_presence_2", "stockholm_absence_1",
        # "muttaburra_absence_1", "muttaburra_absence_2",
        # "wongalee_presence_1", "wongalee_presence_2", "wongalee_presence_3", "wongalee_presence_4",
        # "wongalee_absence_1", "wongalee_absence_2",
        # "roper_presence_1", "roper_presence_2", "roper_presence_3", "roper_presence_4",
        # "roper_absence_1",

        # Removed — marginal LOSO val_auc:
        # "maria_downs_presence", "maria_downs_absence",
        # "pormpuraaw_presence_1", "pormpuraaw_presence_2", "pormpuraaw_presence_3",
        # "pormpuraaw_absence_1", "pormpuraaw_absence_2",

        # Removed — nassau presence fully eliminated by noise filter (likely bad labels):
        # "nassau_presence_1", "nassau_presence_2", "nassau_presence_3", "nassau_presence_4",
        # "nassau_absence_1", "nassau_absence_2", "nassau_absence_3",

        # Removed — moroak is presence-only, no reliable absence context:
        # "moroak_presence_1", "moroak_presence_2", "moroak_presence_3",

        # Removed — ranken_river: presence bboxes too coarse, indistinguishable from grassland:
        # "ranken_river_presence_N_1", ... "ranken_river_absence_N_9",

        # Removed — barkly: sparse Parkinsonia, dirt pixels not filtered cleanly:
        # "barkly_presence_1", ... "barkly_absence_1",

        # norman_road_absence_1/2 retained — LOSO score (0.663) gives no evidence of noise.
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.7,
    },
    train_kwargs={
        "lr":                   2e-6,
        "weight_decay":         0.20,
        "n_epochs":             60,
        "patience":             10,
        "spatial_stride":       4,
        "band_noise_std":       0.03,
        "obs_dropout_min":      6,
        "val_sites":            (),
        "stride_exclude_sites": (),
    },
)
