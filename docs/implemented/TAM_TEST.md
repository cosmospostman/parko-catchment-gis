# TAM Test Suite Plan

## Objectives

1. Verify correctness of each TAM component in isolation
2. Catch latent bugs introduced by recent changes (circular DOY encoding, TAMConfig, DOY jitter)
3. Establish a regression baseline before further model changes

---

## File Layout

```
tests/tam/
├── __init__.py
├── conftest.py          # shared fixtures
├── test_config.py       # TAMConfig
├── test_dataset.py      # TAMDataset, TAMSample, collate_fn
├── test_model.py        # TAMClassifier, _doy_encoding
└── test_train.py        # spatial_split, train_tam (fast smoke)
```

---

## Shared Fixtures (`conftest.py`)

```python
@pytest.fixture
def band_cols() -> list[str]:
    return list(BAND_COLS)  # 10 S2 bands

@pytest.fixture
def pixel_df(band_cols) -> pd.DataFrame:
    """30 observations for two pixels across one year.
    Seeded for reproducibility. Both pixels have ≥8 clear observations."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in ["px_pres", "px_abs"]:
        dates = pd.date_range("2023-01-15", periods=30, freq="12D")
        for d in dates:
            rows.append({
                "point_id": pid, "date": str(d.date()),
                "scl_purity": 1.0, "year": 2023,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
    return pd.DataFrame(rows)

@pytest.fixture
def labels() -> pd.Series:
    return pd.Series({"px_pres": 1.0, "px_abs": 0.0})

@pytest.fixture
def pixel_coords() -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": ["px_pres", "px_abs"],
        "lon": [144.0, 144.1],
        "lat": [-23.0, -23.5],   # px_abs is further south → goes to val set
    })

@pytest.fixture
def default_cfg() -> TAMConfig:
    return TAMConfig()

@pytest.fixture
def small_model(default_cfg) -> TAMClassifier:
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32)
    return TAMClassifier.from_config(cfg)
```

---

## `test_config.py` — TAMConfig

### TC-1 Default values are self-consistent
Assert `d_model % n_heads == 0` and `d_model % 2 == 0` (required by circular DOY encoding).

**Latent bug this catches:** d_model=odd would silently produce d_model-1 sin/cos pairs and leave one dimension unused in `_doy_encoding`.

### TC-2 Round-trip serialisation
`TAMConfig.from_dict(cfg.to_dict()) == cfg`. Verifies no field is dropped or type-coerced on round-trip.

**Latent bug:** `from_dict` filters unknown keys — confirm it also doesn't silently drop *known* fields whose value happens to be falsy (0, 0.0).

### TC-3 Unknown keys in checkpoint JSON are ignored
`TAMConfig.from_dict({"d_model": 32, "unknown_future_key": 999})` should not raise.

### TC-4 Partial dict uses defaults for missing fields
`TAMConfig.from_dict({"d_model": 32})` should produce a valid config with all other fields at their defaults.

---

## `test_dataset.py` — TAMDataset, TAMSample, collate_fn

### TD-1 Basic construction and length
Dataset built from `pixel_df` + `labels` has `len == 2` (one window per pixel, one year).

### TD-2 Band normalisation — training mode computes stats
When `band_mean/band_std` are not supplied, stats are computed from the data. `ds.band_mean` has shape `(N_BANDS,)`, all finite. `ds.band_std` has no zero entries (clamped at 1e-6).

**Latent bug:** The `np.where(band_std < 1e-6, 1.0, band_std)` guard runs on the computed std array but the comparison uses the pre-clamp values — verify that the stored `self.band_std` actually contains the clamped values, not the raw ones.

### TD-3 Band normalisation — inference mode uses supplied stats
When `band_mean/band_std` are supplied, `ds.band_mean` equals the supplied values exactly (no recomputation).

### TD-4 Normalised bands have zero mean and unit variance (approximately)
For a dataset large enough to be representative, `normed.mean(axis=0) ≈ 0` and `normed.std(axis=0) ≈ 1` across all training observations. Tolerance: `atol=0.1`.

### TD-5 Padding mask correctness
For a pixel with `n < MAX_SEQ_LEN` observations: `mask[:n] == False`, `mask[n:] == True`. Shape is `(MAX_SEQ_LEN,)`.

### TD-6 Padding positions are zero in bands and DOY
`bands[n:] == 0` and `doy[n:] == 0` for all padded positions.

**Latent bug:** If normalisation is applied before zero-padding (e.g. via broadcasting), padding rows could pick up the band_mean offset rather than staying zero. This confirms the zero-fill happens after normalisation.

### TD-7 DOY values are in 1–365 for valid positions
`doy[:n]` values all satisfy `1 <= v <= 365`.

### TD-8 DOY order is monotonically non-decreasing
`np.diff(sample.doy[:n]) >= 0` for all valid positions. Observations are sorted by date in `__init__` — verify the sort is preserved through `__getitem__`.

### TD-9 SCL filter removes low-purity observations
Build a `pixel_df` where half the rows have `scl_purity=0.1`. The resulting dataset should have fewer windows or fewer observations per window than the unfiltered case.

### TD-10 Pixels with fewer than `min_obs_per_year` observations are excluded
Set `min_obs_per_year=8`. Add a pixel with only 5 observations. Assert that pixel does not appear in `ds.unique_pixels()`.

**Latent bug:** The filter uses `len(grp) < min_obs_per_year` *after* SCL filtering — a pixel that has 10 raw rows but only 4 pass SCL should be excluded. Confirm the filter operates on the post-SCL group size.

### TD-11 Labels restrict pixels in training mode
Only pixels in `labels.index` appear in `ds.unique_pixels()`. Pixels in `pixel_df` but not in `labels` are silently excluded.

### TD-12 Inference mode (`labels=None`) includes all pixels
When `labels=None`, all pixels in `pixel_df` appear in the dataset, and `sample.label == 0.0` for all items.

### TD-13 DOY jitter — train dataset varies across calls
With `doy_jitter=7`, calling `ds[0]` twice returns different DOY tensors with non-negligible probability. Sample 20 calls; assert at least 2 distinct DOY arrays.

### TD-14 DOY jitter — order preserved after jitter
Over 500 draws with `doy_jitter=7`, assert `np.diff(doy[:n]) >= 0` for every draw. This is the regression test for the wrap-vs-clamp bug.

### TD-15 DOY jitter — values stay in 1–365 after jitter
Over 500 draws, assert `1 <= doy[:n].min()` and `doy[:n].max() <= 365`.

### TD-16 DOY jitter — val dataset is deterministic
With `doy_jitter=0`, 10 consecutive calls to `ds[0]` return identical DOY tensors.

### TD-17 Multi-year pixels produce multiple windows
Add a pixel with observations across 2023 and 2024 (≥8 obs each year). Assert `len(ds) == 3` (two windows for that pixel plus the other pixel).

### TD-18 `collate_fn` produces correct batch shapes
Collate a list of 4 samples. Assert:
- `bands`: `(4, MAX_SEQ_LEN, N_BANDS)` float32
- `doy`: `(4, MAX_SEQ_LEN)` int64
- `mask`: `(4, MAX_SEQ_LEN)` bool
- `label`: `(4,)` float32
- `weight`: `(4,)` float32
- `point_id`: list of length 4
- `year`: list of length 4

### TD-19 `band_stats` property returns copies, not references
Mutating the returned arrays does not change `ds.band_mean` / `ds.band_std`.

---

## `test_model.py` — `_doy_encoding`, TAMClassifier

### TM-1 `_doy_encoding` — padding positions are zero vectors
`doy=0` → encoding is exactly zero across all d_model dimensions.

**Latent bug this guards:** The mask `(doy != 0).float()` is applied element-wise — if the shape broadcast fails silently, padding positions could receive non-zero encodings and leak positional information into the attention.

### TM-2 `_doy_encoding` — valid positions are non-zero
`doy` in 1–365 → at least one encoding dimension is non-zero.

### TM-3 `_doy_encoding` — output shape
Input `(B, T)` → output `(B, T, d_model)` for various B and T.

### TM-4 `_doy_encoding` — seasonality is smooth
`doy=1` and `doy=365` produce encodings with high cosine similarity (≥0.99). This directly tests the circular property — a learned embedding would not guarantee this.

**Latent bug this catches:** If the formula used `doy / 365` instead of `(doy-1) / 365`, day 365 would be close to but not equal to day 1. The current formula `2π·k·doy/365` means day 365 ≠ day 1 (it equals day 0 which is padding). Adjust test to match actual formula — the point is that the gap between day 364 and day 1 should be small.

### TM-5 `_doy_encoding` — values are bounded
All encoding values are in `[-1, 1]` (sin/cos range). Verifies no scaling bug.

### TM-6 Forward pass output shapes
`model(bands, doy, mask)` with batch size B=4, sequence length T=MAX_SEQ_LEN returns `(prob, logit)` both shape `(4,)`.

### TM-7 Forward pass — probabilities in (0, 1)
`prob` values all satisfy `0 < p < 1` for non-degenerate input.

### TM-8 Forward pass — no NaNs or infs
`prob` and `logit` are all finite for random valid input.

### TM-9 Padding-only batch does not crash
A batch where every position is masked (`mask` all True) should not raise or produce NaN — the mean-pool clamps the denominator at 1.

**Latent bug:** `valid.sum(dim=1).clamp(min=1)` prevents division by zero, but if `x * valid` produces NaN before the sum (e.g. from a degenerate normalisation), the clamp doesn't help. Confirms robustness of the pool.

### TM-10 All-padding vs no-padding give different outputs
A batch of real observations should produce different `prob` values from a batch of identical observations with all positions masked.

### TM-11 `get_attention_weights` — output structure
Returns a list of length `n_layers`. Each element has shape `(n_heads, T, T)`.

### TM-12 `get_attention_weights` — weights sum to 1 over keys
For each query position, attention weights over non-masked key positions sum to approximately 1. Confirms softmax is applied correctly and padding mask is respected.

**Latent bug this catches:** The manual attention extraction in `get_attention_weights` re-implements the layer forward pass. If `key_padding_mask` is not passed correctly to `self_attn`, the model may attend to padding positions (those weights should be ~0 after softmax with -inf masking). Assert `attn[:, :, n:]` (attention to padding positions) is near zero.

### TM-13 `get_attention_weights` — layer-by-layer continuity
The x fed into each subsequent layer in `get_attention_weights` equals what the full encoder would produce up to that layer. Cross-check by comparing `get_attention_weights` output logit against `model.forward()` logit on the same input — they must match exactly.

**Latent bug:** This is the most important test for `get_attention_weights`. The manual re-implementation could diverge from PyTorch's internal forward (e.g. if dropout is not disabled, or if norm_first=True is used). Running in `model.eval()` disables dropout, so the outputs must be identical.

### TM-14 `from_config` produces correct architecture
`TAMClassifier.from_config(cfg)` with a non-default config (d_model=32, n_heads=2, n_layers=3) produces a model where `model.d_model == 32`, `model.n_heads == 2`, `model.n_layers == 3`.

### TM-15 `config()` round-trips through `TAMConfig.from_dict`
`TAMConfig.from_dict(model.config())` produces a config that reconstructs an identical architecture.

### TM-16 Parameter count decreases relative to learned DOY embedding
Verify `sum(p.numel() for p in model.parameters())` is less than the equivalent model with `nn.Embedding(366, d_model)` — i.e. the circular encoding has zero learnable parameters. Regression test against the pre-refactor architecture.

---

## `test_train.py` — `spatial_split`, `train_tam`

### TT-1 `spatial_split` — both classes appear in train and val
With enough pixels per class, both 0.0 and 1.0 labels appear in both `train_labels` and `val_labels`.

### TT-2 `spatial_split` — val fraction is approximately correct
`len(val_labels) / len(labels) ≈ val_frac` within 1 pixel (due to `max(1, int(...))` rounding).

### TT-3 `spatial_split` — no overlap between train and val
`set(train_labels.index) & set(val_labels.index) == set()`.

### TT-4 `spatial_split` — val pixels are spatially contiguous (southernmost)
For each class, the val pixels have lower latitude than all train pixels of the same class. This is the core invariant of the spatial holdout.

**Latent bug:** The sort is `ascending=True` (default), so `cls.index[:n_val]` takes the southernmost. If the latitude data has ties or NaN, the sort order is undefined. Test with unique latitudes to lock in the expected behaviour.

### TT-5 `spatial_split` — single pixel per class does not crash
With `labels` containing exactly one presence and one absence pixel, the split should not raise even though one class ends up entirely in val.

### TT-6 `train_tam` smoke — loss decreases and checkpoint is saved
Run `train_tam` with `n_epochs=5`, `patience=5` on the synthetic `pixel_df`. Assert:
- `tam_model.pt` exists in `out_dir`
- `tam_config.json` exists and is valid JSON
- `tam_band_stats.npz` exists with keys `mean` and `std`
- Returned model produces finite probabilities on the training data

This does **not** assert AUC (too few samples), only that training runs without error and checkpoints are written.

### TT-7 `load_tam` reconstructs identical architecture
`load_tam(out_dir)` after `train_tam` returns a model whose `config()` dict matches the original model's `config()` dict exactly.

### TT-8 `load_tam` produces identical predictions
The model returned by `load_tam` produces the same `prob` values as the model returned by `train_tam` on the same input batch (within float32 precision).

**Latent bug:** `train_tam` saves the config mid-training (before best checkpoint is found) and again at the end. If training is interrupted, the saved config might not match the checkpoint weights. `load_tam` must reconstruct from the *final* config. This test confirms consistency.

---

## Latent Bugs Summary

| ID | Location | Description |
|----|----------|-------------|
| TD-2 | `dataset.py:99` | `band_std` clamp uses `np.where` on pre-clamp values — confirm stored std is actually clamped |
| TD-6 | `dataset.py:126` | Normalisation applied before zero-padding could corrupt padding rows |
| TD-10 | `dataset.py:108` | `min_obs_per_year` filter operates on post-SCL group size — confirm not pre-SCL |
| TM-12 | `model.py:97-103` | `get_attention_weights` may not correctly zero out attention to padding positions |
| TM-13 | `model.py:94-111` | Manual layer re-implementation in `get_attention_weights` may diverge from `encoder.forward` |
| TT-4 | `train.py:56` | `spatial_split` sort order undefined with tied or NaN latitudes |
| TT-8 | `train.py:227-229` | Config saved twice (mid-training and end) — checkpoint may not match final config if interrupted |

---

## Conventions

- Use `np.random.default_rng(42)` for all synthetic data
- All TAM tests run on CPU; no GPU assertions
- Keep `train_tam` smoke test fast: `n_epochs=5`, `batch_size=4`, `d_model=16`, `n_layers=1`
- No parquet I/O in unit tests — pass DataFrames directly
- All tests in `tests/tam/` are self-contained; no dependency on real Sentinel-2 data
