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

**Note:** The experiment default is now `max_seq_len=128`. The 0.860 result was trained at `max_seq_len=64` (before this was updated). The sweep baseline run below re-runs at current defaults to establish the new reference.

**Motivation:** 64 timesteps was too tight for high-S1-cadence sites (Kowanyama reaches 106 obs/px/yr). 128 covers 100% of single-year pixels. The sweep baseline and seq_128 runs both include `p_gate=0.3` (the current experiment default).

**Risk:** longer sequences mean more padding for short-history pixels, which adds noise. At batch_size=4096 with seq=128 and p_gate, watch for OOM — may need to drop to 2048.

**Sweep:**

| run | max_seq_len | p_gate | command |
|-----|-------------|--------|---------|
| baseline | 128 | 0.3 | `python -m tam.pipeline train --experiment v10 --output-dir outputs/sweep-0526/baseline` |

---

## 5. Gate augmentation (cascade prerequisite)

**Motivation:** Gate augmentation (`p_gate=0.3, T_gate=8`) is already in the experiment default. The `baseline` run above tests this combined with seq=128. A separate T_gate=16 variant checks whether T=8 is too tight.

**Expected outcome (gate_aug_t16):** recall@gate ≥ 0.99 on presence pixels at T=16, discarding 70–80% of absence pixels.

**Risk:** `p_gate=0.3` fires on 30% of items, reducing effective full-length examples per epoch. If val CVaR25 drops vs 0.860, reduce `p_gate` to 0.2.

**Sweep:**

| run | p_gate | T_gate | command |
|-----|--------|--------|---------|
| gate_aug_t16 | 0.3 | 16 | `python -m tam.pipeline train --experiment v10 --t-gate 16 --output-dir outputs/sweep-0526/gate_aug_t16` |

After each run, evaluate cascade quality:
```
python scripts/bench_cascade.py --checkpoint outputs/sweep-0526/baseline --device cuda
```
Target: recall@gate ≥ 0.99 at some threshold with ≥ 70% absence discarded.

---

## Run order (overnight script)

All 8 runs in sequence. The `baseline` run (seq=128 + gate) goes first to establish the new reference; dropout and layers follow as the cheapest single-variable tests.

1. `baseline` — seq=128, p_gate=0.3, all other defaults (new reference point)
2. `dropout_0.4`
3. `dropout_0.3`
4. `layers_4`
5. `lr_1e4`
6. `lr_2e5`
7. `gate_aug_t16`
8. `max_seq_len` sweep not included — baseline already runs at seq=128; add seq_64 and seq_96 as follow-up if needed

---

## Results

| run | dropout | n_layers | lr | seq | p_gate | CVaR25 | macro AUC | notes |
|-----|---------|----------|----|-----|--------|--------|-----------|-------|
| prev best (seq=64, no gate) | 0.5 | 3 | 5e-5 | 64 | 0 | 0.860 | 0.944 | trained before seq/gate changes |
| baseline | 0.5 | 3 | 5e-5 | 128 | 0.3 | | | new reference |
| dropout_0.4 | 0.4 | 3 | 5e-5 | 128 | 0.3 | | | |
| dropout_0.3 | 0.3 | 3 | 5e-5 | 128 | 0.3 | | | |
| layers_4 | 0.5 | 4 | 5e-5 | 128 | 0.3 | | | |
| lr_1e4 | 0.5 | 3 | 1e-4 | 128 | 0.3 | | | |
| lr_2e5 | 0.5 | 3 | 2e-5 | 128 | 0.3 | | | |
| gate_aug_t16 | 0.5 | 3 | 5e-5 | 128 | 0.3 (T=16) | | | |
