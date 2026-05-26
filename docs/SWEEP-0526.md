# Hyperparameter sweep candidates — 2026-05-26

Baseline: v10 with 18 bands (s1_vh s1_vv s1_vh_vv s1_rvi), d_ff=1024, batch_size=4096.  
**CVaR25=0.860, macro AUC=0.944**

---

## 1. Dropout

**Motivation:** dropout=0.5 may be over-regularising at d_ff=1024. The epoch-60 loss (0.180) was still slowly declining — consistent with regularisation slowing convergence rather than a capacity ceiling. Reducing dropout should let the model converge faster and potentially to a lower loss floor.

**Risk:** dropout is doing real work suppressing site-specific overfitting across a diverse 12-site training set. Reducing it too far may push train AUC toward 1.0 while val CVaR25 drops.

**Sweep:**

| run | dropout | command |
|-----|---------|---------|
| dropout_0.5 | 0.5 | *(baseline — already run)* |
| dropout_0.4 | 0.4 | `python -m tam.pipeline train --experiment v10 --dropout 0.4 --output-dir outputs/sweep-0526/dropout_0.4` |
| dropout_0.3 | 0.3 | `python -m tam.pipeline train --experiment v10 --dropout 0.3 --output-dir outputs/sweep-0526/dropout_0.3` |

---

## 2. n_layers

**Motivation:** d_ff was the bottleneck, not depth. Now that FFN width is fixed, adding a 4th layer gives the model more representational depth for the temporal mixing task. S1+S2 interleaving may particularly benefit from deeper cross-timestep attention.

**Risk:** more layers = more parameters = more regularisation needed. May need to pair with slightly higher dropout or weight_decay if val CVaR25 drops. Also ~33% longer per epoch.

**Sweep:**

| run | n_layers | command |
|-----|----------|---------|
| layers_3 | 3 | *(baseline — already run)* |
| layers_4 | 4 | `python -m tam.pipeline train --experiment v10 --n-layers 4 --output-dir outputs/sweep-0526/layers_4` |

---

## 3. Learning rate

**Motivation:** lr=5e-5 was set when d_model=128 and d_ff=64. At d_model=256, d_ff=1024 the loss landscape is different. A higher lr may converge faster; a lower lr may find a better minimum.

**Risk:** lr=1e-4 may overshoot at this model size with AdamW + weight_decay=0.1. lr=2e-5 may just be slower convergence with no ceiling gain.

**Sweep:**

| run | lr | command |
|-----|----|---------|
| lr_5e5 | 5e-5 | *(baseline — already run)* |
| lr_1e4 | 1e-4 | `python -m tam.pipeline train --experiment v10 --lr 1e-4 --output-dir outputs/sweep-0526/lr_1e4` |
| lr_2e5 | 2e-5 | `python -m tam.pipeline train --experiment v10 --lr 2e-5 --output-dir outputs/sweep-0526/lr_2e5` |

---

## 4. max_seq_len

**Motivation:** 64 timesteps may truncate useful temporal context, particularly for S1+S2 interleaved sequences where each source contributes ~32 observations. Extending to 96 or 128 gives the attention mechanism more temporal range.

**Risk:** memory scales with seq_len (attention is O(n²)). At 128, batch_size=4096 may OOM — may need to drop to 2048. Also, longer sequences mean more padding for short-history pixels, which could hurt rather than help.

**Sweep:**

| run | max_seq_len | command |
|-----|-------------|---------|
| seq_64 | 64 | *(baseline — already run)* |
| seq_96 | 96 | `python -m tam.pipeline train --experiment v10 --max-seq-len 96 --output-dir outputs/sweep-0526/seq_96` |
| seq_128 | 128 | `python -m tam.pipeline train --experiment v10 --max-seq-len 128 --batch-size 2048 --output-dir outputs/sweep-0526/seq_128` |

---

## 5. Gate-augmented training (cascade prerequisite)

**Motivation:** The cascade gate runs V10 at T=8 (farthest-point DOY sampling) to discard high-confidence negatives before the full T=128 pass. The current checkpoint was trained only on full-length sequences — at T=8 it achieves only ~40% recall on presence pixels (r=0.39 correlation with full-T scores, presence p50=0.005). Gate-augmented training adds a stochastic short-sequence view during training so the model is discriminative at both T=128 and T=8.

**Mechanism:** `p_gate=0.3` in `TAMDataset.__getitem__` — 30% of items are subsampled to `T_gate=8` observations via farthest-point DOY sampling before the loss is computed. Every pixel is still seen at full length every epoch; the short view is an additional augmentation, not a held-out split.

**Expected outcome:** recall@gate ≥ 0.99 on presence pixels at T=8, discarding 70–80% of absence pixels → ~3.8× cascade speedup → ~12 hrs/yr on A10G (vs 46 hrs without cascade).

**Risk:** the gate augmentation fires on 30% of items, reducing the effective number of full-length training examples. If val CVaR25 drops, reduce `p_gate` to 0.2 or try `T_gate=16`.

**Sweep:**

| run | p_gate | T_gate | command |
|-----|--------|--------|---------|
| gate_aug | 0.3 | 8 | `python -m tam.pipeline train --experiment v10 --output-dir outputs/sweep-0526/gate_aug` |
| gate_aug_t16 | 0.3 | 16 | `python -m tam.pipeline train --experiment v10 --p-gate 0.3 --t-gate 16 --output-dir outputs/sweep-0526/gate_aug_t16` |

After each run, evaluate cascade quality:
```
python scripts/bench_cascade.py --checkpoint outputs/sweep-0526/gate_aug --device cuda
```
Target: recall@gate ≥ 0.99 at some threshold with ≥ 70% absence discarded.

---

## Priority order

1. **Dropout** — cheapest, most likely to move the needle given current convergence behaviour
2. **n_layers** — second cheapest, tests depth now that FFN is no longer the bottleneck
3. **lr** — quick sanity check, low expected gain
4. **max_seq_len** — most expensive per run, save for last
5. **Gate augmentation** — run in parallel with seq_128; both use T=128 and are the most expensive

---

## Results

| sweep | value | CVaR25 | macro AUC | notes |
|-------|-------|--------|-----------|-------|
| baseline | d=0.5, L=3, lr=5e-5, seq=64 | 0.860 | 0.944 | — |
