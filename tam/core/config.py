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
    use_n_obs:  bool  = True   # append normalised obs-count to pooled repr before head

    # Data (mirrors dataset.py constants — change both together)
    n_bands:          int   = 13
    max_seq_len:      int   = 128
    min_obs_per_year: int   = 8
    scl_purity_min:   float = 0.5
    doy_jitter:       int   = 7   # ±days of DOY shift applied per window during training

    # Training
    n_epochs:        int   = 30
    batch_size:      int   = 1024
    lr:              float = 1e-4
    weight_decay:    float = 1e-3
    val_frac:        float = 0.2
    patience:        int   = 10
    min_delta:       float = 1e-4
    obs_dropout_min: int   = 0   # if >0, subsample each window to Uniform(obs_dropout_min, n) during training

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TAMConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})
