# Master Query Attention — Exploration Notes

## Context

This document records an exploration of the L-TAE (Lightweight Temporal Attention Encoder)
architecture and whether its master query technique is worth adopting in a future TAM version.
Started June 2026.

---

## What L-TAE is

L-TAE (Garnot & Landrieu, 2020) is a transformer designed specifically for classifying
Satellite Image Time Series (SITS). It strips standard self-attention down to three key
modifications:

1. **Master Query** — replaces per-token Q vectors with a single learned parameter vector
2. **Channel grouping** — splits feature channels across heads, forcing each head to specialise
3. **Continuous DOY positional encoding** — encodes the actual day-of-year rather than token position

---

## How V10 compares

**Already doing:**
- Continuous DOY encoding — V10 uses `d_model/2` harmonic pairs, arguably richer than L-TAE's
  original formulation ([tam/core/model.py](../tam/core/model.py))
- Multimodal channel awareness — separate `doy_inv_freq` tables for S1 vs S2, `is_s1` flags

**Gap: master query / attention-as-pooling**

V10 runs full T×T self-attention across 3 layers then mean-pools (with optional DOY density
reweighting). L-TAE collapses these two steps: the attention weights over a single global query
*are* the pooling.

**Gap: channel grouping**

V10's 4 attention heads all see all 18 features (S2 + S1 mixed). L-TAE forces each head to
operate on a distinct channel subset, which produces implicit S1 vs S2 specialisation. V10
has the data structure for this (separate S1/S2 feature lists) but not the architectural
enforcement.

---

## The master query in detail

Standard self-attention: every token computes Q, K, V and attends to every other token
(T × T matrix). For classification this is more structure than needed.

Master query: replace all T per-token queries with a single learned vector `q_master`
of shape `(d_model,)`:

```
scores  = q_master · K^T    # (T,)  — one score per timestep
weights = softmax(scores)   # (T,)
output  = weights · V       # (d_model,) — weighted sum, replaces mean pool
```

With multiple heads, each head has its own `q_master`. They learn to specialise on different
discriminative patterns (e.g. dry-season VH stability vs wet-season NDVI peak height).

**Why this matters for Parkinsonia detection:**

V10's DOY-density mean pool weights by acquisition frequency, not diagnostic value. A master
query would learn to upweight the phenologically informative timesteps — dry-season VH/VV
(woody structure discriminator vs bare ground) and wet-season NDVI peak (discriminator vs
mangrove) — regardless of observation density. This is directly relevant to the Mitchell River
false positives, where the dry-season VH signal is the key discriminator but gets diluted by
the denser wet-season optical acquisitions.

**Architectural benefit:** simpler and faster — T-length dot product replaces T×T attention.
Attention weights are directly interpretable: they show which acquisitions the model found
discriminative.

---

## Generalisation risk and mitigation

**The concern:** a fixed learned query template could overfit to the semi-arid Parkinsonia
phenological signature (Norman Road, Etna, Landsend) and fail in monsoonal contexts where
the wet–dry NDVI contrast is different and surrounding vegetation is more similar.

**Why it's manageable:**

- Multi-head design means 4+ independent query templates — they don't all need to encode the
  same phenological pattern
- The generalisation problem is not introduced by master queries — V10's mean pool has the
  same vulnerability to training distribution bias
- The real fix is the same in both architectures: diverse training sites covering the full
  ecological range (the Mitchell River data being built for V11)

**The shaped risk:** master queries inherit training distribution bias more explicitly than
mean pooling. If semi-arid presence sites dominate the gradient signal, the queries will be
pulled toward the semi-arid template. Monsoonal presence data (currently thin: Frenchs 4
regions, Roper 4 regions, no Mitchell yet) is the protection.

---

## Recommendation

Do not implement now. The Mitchell River training data and V11 retraining is the right
immediate priority — that addresses the false-positive problem regardless of architecture.

If V11 still struggles with monsoonal false positives after retraining, a master-query pooling
layer replacing the current mean pool is the principled next experiment. It would be roughly
50 lines of code as an optional mode in `TAMClassifier`, and would make attention weights
directly inspectable via `get_attention_weights()`.

---

## Files to read before implementing

- [tam/core/model.py](../tam/core/model.py) — `TAMClassifier`, especially the `forward()`
  pooling block (lines 194–206) and `get_attention_weights()` (lines 320–355)
- [tam/experiments/v10.py](../tam/experiments/v10.py) — V10 hyperparameters (d_model=256,
  4 heads, 3 layers)
- [docs/MITCHELL-TRAIN.md](MITCHELL-TRAIN.md) — V11 training data plan; complete this first
