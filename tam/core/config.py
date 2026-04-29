"""tam/core/config.py — TAMConfig: single source of truth for all TAM hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TAMConfig:
    # Model architecture
    d_model:    int   = 32
    n_heads:    int   = 4
    n_layers:   int   = 1
    d_ff:       int   = 64
    dropout:    float = 0.3
    use_n_obs:         bool  = True   # append normalised obs-count to pooled repr before head
    n_global_features: int   = 9     # 5 S2 globals + 4 S1 globals (0 to disable)
    use_s1:  bool | str  = True   # True=snap S1 to S2, "s1_only"=S1 sequence only, False=S2 only

    # Data (mirrors dataset.py constants — change both together)
    # 13 S2 features + 4 S1 features (s1_vh, s1_vv, s1_vh_vv, s1_rvi) when use_s1=True
    n_bands:          int   = 17
    max_seq_len:      int   = 128
    min_obs_per_year: int   = 8
    scl_purity_min:   float = 0.5
    doy_jitter:       int   = 7   # ±days of DOY shift applied per window during training
    band_noise_std:   float = 0.0 # std of per-window band offset in normalised space (training only)

    # Training
    n_epochs:        int   = 30
    batch_size:      int   = 1024
    lr:              float = 1e-4
    weight_decay:    float = 1e-3
    val_frac:        float = 0.2
    val_sites:       tuple = ()   # if non-empty, hold out these sites entirely instead of using val_frac
    patience:        int   = 5
    min_delta:       float = 1e-4
    obs_dropout_min: int   = 0   # if >0, subsample each window to Uniform(obs_dropout_min, n) during training
    warmup_freeze_epochs: int = 0  # if >0, freeze temporal stream for first N epochs (head-only warmup)
    doy_density_norm: bool = False  # if True, weight mean pool by inverse DOY observation frequency
    spatial_stride:       int   = 1   # if >1, thin training pixels spatially (every Nth pixel per region)
    stride_exclude_sites: tuple = ()  # site prefixes exempt from spatial stride (e.g. small/sparse sites)

    # Presence pixel noise filters (applied before training, using global features)
    presence_min_dry_ndvi: float = 0.10  # filter water/bare soil (dry-season median NDVI)
    presence_min_rec_p:    float = 0.20  # filter low-amplitude pixels (bare soil, water, grass)
    presence_grass_nir_cv: float = 0.20  # filter high-variability pixels (grass)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TAMConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})
