# Validation Score Design

## Problem

The current primary checkpoint metric (`val_auc`) is a global pooled AUC across all validation pixels. Large sites dominate, and the model can score well on average while failing entirely on one region. The goal is a metric that reflects how much the model is **lifting the lower end of performance** — rewarding generalisation to hard or unfamiliar regions, not just overall discrimination.

## Data structure

Each validation region is a single-class bbox (presence or absence). Bboxes are grouped into sites (Etna, Hughenden, Frenchs, Landsend, Barcoorah). Each site has at least one presence bbox and one absence bbox.

V9 validation set: 25 bboxes across 5 sites (Etna 9, Hughenden 7, Frenchs 4, Landsend 3, Barcoorah 2).

Because each bbox is pure-class, AUC is undefined at the bbox level — it requires pairing a presence bbox with an absence bbox.

## Proposed metric: site-weighted CVaR AUC

### Construction

1. **Enumerate pairs.** For each site, form all (presence bbox, absence bbox) pairs. Etna with 4 presence and 5 absence bboxes gives 20 pairs; Barcoorah with 1+1 gives 1 pair.

2. **Compute pair AUC.** For each pair, pool pixels from the two bboxes and compute `roc_auc_score`. Skip pairs where either bbox contributes fewer than `min_pair_pixels` pixels (default: 50) or where both classes are not represented after filtering NaNs. A flat model producing constant probabilities will yield AUC=0.5 — this is valid signal, not an error, and should remain in the tail.

3. **Weight by site.** Among sites that produced at least one valid pair, assign each pair a weight of `1 / (pairs in site)` and normalise across active sites so weights sum to 1. Sites with zero valid pairs (both bboxes wiped by cloud/NaN filtering) are excluded from the weight budget entirely — the remaining sites scale up to fill the gap. This prevents a ZeroDivisionError and keeps the metric well-defined when a small site disappears.

4. **Compute CVaR at α=0.25.** Sort pairs by AUC ascending. Accumulate weights from the bottom until 0.25 is reached. The CVaR score is the weighted mean AUC of those pairs.

This gives a single scalar that answers: *on the weakest quarter of region-pairings, weighted so no site dominates, how well does the model discriminate?*

**Note on the Barcoorah leverage effect.** With 5 equal-weight sites, Barcoorah's single pair carries a weight of 0.20 — consuming 80% of the α=0.25 tail budget if it lands at the bottom. This is intentional: the metric must care about every site, including small ones. The risk is that Barcoorah's pair is noisy or mislabelled, in which case the checkpoint metric will stall. Mitigation is operational: keep Barcoorah's labels clean and monitor `val_auc` (global, informational) alongside `val_cvar25`. If the two diverge sharply, Barcoorah is the first place to look.

### Why not worst-case?

Worst-case (minimum pair AUC) is hostage to a single noisy pair, especially from small bboxes. CVaR averages the tail, so one bad pair doesn't collapse the metric, but a site that consistently performs poorly will still drag the score down.

### Why not per-site macro AUC?

Macro AUC pools all pixels within a site before scoring, which hides within-site variation (e.g. a model that nails dense-canopy presence but fails on sparse-canopy presence within the same site). Pair-level scoring exposes that structure.

### Why site-weighting?

Without it, Etna's 20 pairs could flood the bottom quartile and the metric would effectively be worst-case-Etna. Site-weighting ensures the tail represents distributional failure across sites, not just the largest site.

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `cvar_alpha` | 0.25 | Fraction of tail to average. 0.25 = bottom quartile. |
| `min_pair_pixels` | 50 | Minimum pixels per bbox to include a pair. Drops noisy small-region pairs. |

Both should live in the training config (`TAMConfig`) so they are logged and reproducible.

## Logged metrics (per epoch)

| Name | Description | Used for checkpoint |
|------|-------------|-------------------|
| `val_cvar25` | Site-weighted CVaR AUC at α=0.25 | Yes — replaces `val_auc` as primary |
| `val_auc_macro` | Unweighted mean of per-site AUCs | No — informational |
| `val_auc` | Global pooled AUC across all val pixels | No — informational |

`val_auc` is retained as an informational log. If `val_cvar25` drops while `val_auc` climbs, the model is over-indexing on Etna at the expense of smaller sites — a useful diagnostic that would be invisible without the global metric.

## Implementation (train.py)

The change is confined to `tam/core/train.py`.

**Replace** the `val_auc` computation block (lines ~726–735) and the `val_auc_macro` block (lines ~737–744) with a new function or inline block that:

1. Extracts region IDs from `val_pids` (everything before the final `_row_col` suffix — the region ID is already encoded in the point_id).
2. Groups pids by region, then by site (via `_site_class`).
3. Enumerates presence/absence pairs per site.
4. Filters pairs below `min_pair_pixels`.
5. Computes per-pair AUC.
6. Assigns site weights, computes weighted CVaR.
7. Also computes `val_auc_macro` (per-site pooled AUC, unchanged logic) for the secondary log.

**Update** the checkpoint condition, `best_val_auc` tracking, epoch log format string, final summary log, and `tam_config.json` write to use `val_cvar25`.

**Return value** of `train_tam()` remains `(model, best_val_auc)` — semantics change to CVaR but the interface is identical; no changes needed in sweep callers.

## Extracting region ID from point_id

Point_ids have the form `<region_id>_<row>_<col>` where `<region_id>` itself contains underscores (e.g. `etna_presence_1`). The row and col are always the last two `_`-separated tokens. So:

```python
def _region_from_pid(pid: str) -> str:
    return "_".join(pid.split("_")[:-2])
```

This recovers `etna_presence_1` from `etna_presence_1_1234_5678`.

## Resolved design questions

**Should `cvar_alpha` be sweep-tunable?** No. CVaR alpha defines risk appetite, not model capacity. A sweep will drift toward α=1.0 (plain mean AUC) because average performance is easier to optimise than tail performance. Lock it at 0.25.

**Should pairs be capped per site?** No. At 25 bboxes, maximum pairs per site is 20. Computing AUC on a few pixel arrays is milliseconds. Add capping only if the bbox count grows into the thousands.

**What if a model predicts flat probabilities on a pair?** AUC=0.5. Keep it in the tail — it's valid signal, not a degenerate case to filter.
