# CLS Token Implementation Plan

## Context

The current TAM model uses a weighted mean pool over all non-padded time steps to produce a
single classification embedding. The attention visualisations from v9 show that all four heads
have specialised into seasonal windows (wet-season, dry-season, shoulder) but that presence and
absence pixels attend to the same windows — the heads are acting as site-level phenological
filters rather than class-discriminative feature selectors.

Adding a CLS token replaces mean pooling with a learned aggregator prepended to the sequence.
Because the loss gradient flows only through the CLS position, its attention weights are a
direct, clean readout of "which observations mattered for this classification decision." This
makes the attention visualisation interpretable at the class level (not just the site/seasonal
level) and may improve AUC by letting the model selectively down-weight cloudy or non-diagnostic
observations rather than averaging them in equally.

---

## Files to Modify

| File | What changes |
|------|-------------|
| `tam/core/model.py` | Add `cls_token` parameter, modify `__init__`, `forward`, `get_attention_weights`, `config`, `from_config` |
| `tam/viz_attention.py` | Update attention extraction to use CLS row (position 0) instead of mean over all query positions |

---

## Implementation

### 1. `tam/core/model.py` — `TAMClassifier`

**`__init__`** — add `use_cls_token: bool = False` parameter and store it:

```python
self.use_cls_token = use_cls_token
if use_cls_token:
    self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
```

No change to `head_in` — CLS output is `d_model`, same as the mean-pool output.

**`forward`** — after `band_proj + doy_encoding`, conditionally prepend CLS before the encoder
and extract position 0 instead of pooling:

```python
x = self.band_proj(bands) + _doy_encoding(doy, self.d_model)  # (B, T, d_model)

if self.use_cls_token:
    cls = self.cls_token.expand(x.shape[0], -1, -1)   # (B, 1, d_model)
    x   = torch.cat([cls, x], dim=1)                  # (B, T+1, d_model)
    # Extend padding mask: CLS position is never masked
    cls_mask = key_padding_mask.new_zeros(x.shape[0], 1)   # (B, 1) False
    safe_mask = torch.cat([cls_mask, safe_mask], dim=1)     # (B, T+1)

x = self.encoder(x, src_key_padding_mask=safe_mask)

if self.use_cls_token:
    x_pool = x[:, 0, :]   # (B, d_model) — CLS position
else:
    # existing weighted mean pool (lines 154–162, unchanged)
    ...
```

Note: `safe_mask` must be constructed before the `use_cls_token` branch extends it.
The `all_masked` guard (lines 149–150) runs on the original `key_padding_mask` before
the CLS column is prepended, so it remains correct.

**`get_attention_weights`** — same CLS prepend logic, then return only the CLS row of each
layer's attention matrix so callers get `(n_heads, T)` directly:

```python
if self.use_cls_token:
    cls = self.cls_token.expand(1, -1, -1)
    x   = torch.cat([cls, x], dim=1)
    key_padding_mask = torch.cat(
        [key_padding_mask.new_zeros(1, 1), key_padding_mask], dim=1
    )

# existing per-layer loop unchanged, but after squeezing:
attn_weights.append(w.squeeze(0))  # (n_heads, T+1, T+1) if CLS, else (n_heads, T, T)
```

Callers that extract CLS attention do `attn[layer][:, 0, 1:]` — head × CLS-query × obs-keys,
dropping the CLS-to-CLS self-weight at position 0.

**`config` / `from_config`** — add `"use_cls_token": self.use_cls_token` to the config dict
and pass it through `from_config`. This ensures checkpoints are self-describing.

---

### 2. `tam/viz_attention.py` — `attention_by_doy`

Current line 91 averages over the query dimension of `(n_heads, T, T)`:

```python
key_attn = attn_list[layer].cpu().numpy().mean(axis=1)  # (n_heads, T)
```

With CLS token, replace this with the CLS row (query position 0, key positions 1..T):

```python
if model.use_cls_token:
    # CLS row: (n_heads, T+1, T+1)[:, 0, 1:] → (n_heads, T)
    key_attn = attn_list[layer].cpu().numpy()[:, 0, 1:]
else:
    key_attn = attn_list[layer].cpu().numpy().mean(axis=1)
```

The rest of the DOY-binning loop (`lines 92–97`) is unchanged — `key_attn` shape is still
`(n_heads, T)` in both branches.

---

## Config / TAMConfig — `tam/core/config.py`

Add `use_cls_token: bool = False` to the `TAMConfig` dataclass (line 67, after the
`doy_density_norm` field is a natural place). The default `False` means all existing
checkpoints and experiments load without change. `from_dict` already filters to known fields
(line 65), so old configs that lack the key silently default to `False`.

---

## Backwards Compatibility

- Default `use_cls_token=False` — no behaviour change for existing runs.
- Existing checkpoints missing the key in `tam_config.json` will default to `False` via
  `cfg_dict.get("use_cls_token", False)` in `from_config`.
- The model parameter count increases by `d_model` scalars (128 floats = negligible).

---

## Verification

1. **Unit smoke test** — instantiate both `use_cls_token=False` and `True`, run a random
   batch through `forward` and `get_attention_weights`, assert output shapes are correct and
   no NaN.
2. **Attention shape check** — with CLS, `get_attention_weights` should return tensors of
   shape `(n_heads, T+1, T+1)`; the CLS row `[:, 0, 1:]` should sum to ≈1 per head.
3. **Training run** — add `use_cls_token: true` to a sweep config and run a short training
   (e.g. 5 epochs) to confirm loss decreases and no shape errors.
4. **Viz run** — run `viz_attention.py` against the new checkpoint; confirm the per-head
   profiles now show presence/absence divergence on the shoulder window (Mar–May) if the
   hypothesis is correct.
5. **Backwards compat** — load an existing v9 checkpoint with the updated code and confirm
   inference produces identical outputs to before.

---

## Sweep Results — sweep-v9-cls (2026-05-15)

Five variants were run against the standard v9 holdout set (barcoorah, etna, frenchs, hughenden,
landsend) using `d_model=128, n_heads=4, n_layers=2, d_ff=64, lr=5e-05, max_epochs=60,
early_stop=15`.

| Run | `use_cls_token` | `cls_warm_init` | `cls_lr_scale` | Best val AUC |
|-----|:-:|:-:|:-:|:---:|
| `mean_pool` | False | — | — | **0.7555** |
| `cls_cold` | True | False | 10× | 0.7473 |
| `cls_warm_init` | True | True | 10× | 0.7308 |
| `cls_high_lr` | True | False | 10× | 0.7290 |
| `cls_warm_high` | True | True | 10× | ~running~ |

(`cls_warm_high` was still training at epoch 21 when analysis was written; early epochs tracked
similarly to the other CLS variants.)

### Findings

**CLS token did not improve on mean pooling.** The baseline mean-pool model achieved 0.756 val
AUC and continued improving through all 60 epochs. Every CLS variant underperformed by 0.008–
0.027 AUC and early-stopped between epoch 25 and 40, indicating that the CLS token saturated
faster and generalised less well.

**Warm initialisation from `band_proj` weights was counterproductive.** The warm-init variants
(`cls_warm_init` 0.731, `cls_warm_high` tracking similarly) were worse than cold random
initialisation (`cls_cold` 0.747). Seeding the CLS vector from existing band-projection weights
apparently biases it toward an observation-level representation that conflicts with its role as a
sequence-level aggregator.

**Higher CLS learning rate (10×) did not recover performance.** All CLS variants used the same
10× LR scale; neither cold nor warm initialisation benefited from faster CLS adaptation.

### Interpretation

The attention visualisations from v9 showed all four heads specialising into seasonal windows
(wet, dry, shoulder) rather than class-discriminative patterns. The hypothesis was that a CLS
token — by routing gradients through a single position — would force the heads to become
class-discriminative. The sweep suggests that mean pooling already extracts sufficient signal
from these seasonal windows, and the CLS token's additional flexibility does not compensate for
the reduction in training signal per parameter.

A secondary possibility is that the dataset is too small for a learned aggregator to outperform a
simple average: with ~490 k training windows the CLS token may not see enough diversity to learn
a robust sequence summary.

### Conclusion

**Mean pooling remains the default aggregation strategy.** The `use_cls_token` implementation is
kept in the codebase for completeness and future experimentation (e.g. larger datasets, longer
sequences, or pre-training), but it will not be used in v9 production runs.
