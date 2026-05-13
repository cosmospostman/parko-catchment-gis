# Woody Classifier — Stage-1 Woody Mask

A stage-1 XGBoost classifier that separates woody vegetation (trees, shrubs, mangroves) from non-woody land cover (grass, bare soil, sand, water) at 10 m pixel resolution across Australia.

Used as a pre-filter before TAM inference: pixels classified as non-woody are suppressed, eliminating the false-positive problem observed in V9-SPECTRAL on Longreach bare ground.

## Module layout

```
woody-classifier/
├── features.py      # compute_woody_features() — band summaries + S1 stats
├── train.py         # load data → fit XGBoost → save model
├── score.py         # load model → score pixel parquets → write prob parquet
└── evaluate.py      # AUC, precision/recall curve, feature importances
```

## Running training

```bash
python woody-classifier/train.py
```

Optional flags:

| Flag | Default | Effect |
|---|---|---|
| `--out <dir>` | `outputs/woody-classifier` | Where to write model and cached parquets |
| `--no-cache` | off | Ignore cached pixel summaries and recompute from scratch |

Training regions are read from `data/locations/woody-classifier.yaml`. Regions tagged `val` are held out as the validation set; all others are used for training.

**Outputs written to `outputs/woody-classifier/`:**

| File | Contents |
|---|---|
| `model.json` | XGBoost model (or `model.joblib` for sklearn fallback) |
| `feature_names.json` | Ordered feature list required at inference time |
| `train_summaries.parquet` | Cached pixel summaries (reused on subsequent runs) |
| `train_labels.parquet` | Cached labels |
| `val_summaries.parquet` | Cached val summaries |
| `val_labels.parquet` | Cached val labels |

## Running scoring

```bash
python woody-classifier/score.py <parquet_or_dir> [<parquet_or_dir> ...]
```

Examples:

```bash
# Score all parquets under a directory
python woody-classifier/score.py data/pixels/longreach-8x8km

# Write scores to a custom location
python woody-classifier/score.py data/pixels/longreach-8x8km --out outputs/woody-classifier/scores

# Use recall-optimised threshold instead of the default high-precision one
python woody-classifier/score.py data/pixels/longreach-8x8km --threshold 0.5
```

One output parquet is written per input parquet, named `<stem>_woody_probs.parquet`, with columns `point_id` and `prob_woody` (float32). Default threshold for the mask is **0.85** (high precision); 0.5 is available for recall-optimised use.

## Integration with TAM

After validation, `tam/core/score.py` reads `prob_woody` scores and skips pixels below threshold before running TAM inference.
