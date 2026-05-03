# Sweep Results — v6_spectral

## 1. Hyperparameter sweep (2026-04-27)

**Script:** `sweeps/sweep.sh`  
**Output:** `outputs/sweep/summary.txt`  
**Grid:** lr × weight_decay × dropout (2 × 3 × 2 = 12 runs)

| run_id | lr | wd | dropout | best_val_auc |
|---|---|---|---|---|
| lr2e-6_wd0.20_do0.7 | 2e-6 | 0.20 | 0.7 | **0.677** |
| lr2e-6_wd0.20_do0.5 | 2e-6 | 0.20 | 0.5 | 0.669 |
| lr2e-6_wd0.12_do0.7 | 2e-6 | 0.12 | 0.7 | 0.635 |
| lr5e-6_wd0.12_do0.5 | 5e-6 | 0.12 | 0.5 | 0.639 |
| lr5e-6_wd0.05_do0.5 | 5e-6 | 0.05 | 0.5 | 0.627 |
| lr2e-6_wd0.12_do0.5 | 2e-6 | 0.12 | 0.5 | 0.572 |
| lr2e-6_wd0.05_do0.7 | 2e-6 | 0.05 | 0.7 | 0.606 |
| lr5e-6_wd0.12_do0.7 | 5e-6 | 0.12 | 0.7 | 0.637 |
| lr5e-6_wd0.20_do0.5 | 5e-6 | 0.20 | 0.5 | 0.632 |
| lr5e-6_wd0.20_do0.7 | 5e-6 | 0.20 | 0.7 | 0.556 |
| lr2e-6_wd0.05_do0.5 | 2e-6 | 0.05 | 0.5 | 0.493 |
| lr5e-6_wd0.05_do0.7 | 5e-6 | 0.05 | 0.7 | 0.491 |

**Best run:** `lr=2e-6, wd=0.20, dropout=0.7` → val_auc=0.677, loss=1.0166 at epoch 13/60.

### Key findings

- **Weight decay is the dominant factor.** At `lr=2e-6`, AUC climbs monotonically from 0.49→0.67 as `wd` goes 0.05→0.20. Strong regularisation is clearly helping.
- **Higher dropout helps at `lr=2e-6`.** Both `wd=0.12` and `wd=0.20` improve with dropout 0.7. The benefit is smaller at high wd, suggesting the two regularisers are partially redundant.
- **`lr=5e-6` is less stable.** Results are noisier and the interaction with dropout flips direction. That LR is likely too large.
- **val_auc consistently exceeds train_auc throughout training.** At the best epoch, val_auc=0.677 vs train_auc=0.518. This is a label noise signature — the val set (pormpuraaw) is cleaner than the training pool. The signal is real but the training data is noisier than the val data.
- **The ceiling is low.** Loss moved from 1.07 to 1.02 over 13 epochs before early stopping at epoch 23. The model is learning slowly and the bottleneck is data quality, not hyperparameters.

**Fixed hyperparams for subsequent experiments:** `lr=2e-6, wd=0.20, dropout=0.7`

---

## 2. Leave-one-site-out sweep (2026-04-28)

**Script:** `sweeps/sweep_loso.sh`  
**Output:** `outputs/models/sweep_loso/loso_summary.txt`  
**Hyperparams:** fixed at best from sweep above.  
**Method:** Hold out each site entirely as the validation set; train on all remaining sites.  
Sites excluded from LOSO (absence-only or commented out of experiment): `quaids`, `mitchell_river`, `muttaburra`, `barkly`, `ranken_river`.

| held_out_site | best_val_auc | train_auc | n_val_px |
|---|---|---|---|
| frenchs | **0.785** | 0.445 | 14,998 |
| alexandria | 0.667 | 0.541 | 2,752 |
| norman_road | 0.663 | 0.559 | 4,037 |
| lake_mueller | 0.657 | 0.489 | 1,460 |
| maria_downs | 0.624 | 0.493 | 11,284 |
| pormpuraaw | 0.576 | 0.580 | 6,187 |
| wongalee | 0.468 | 0.476 | 3,604 |
| roper | 0.430 | 0.556 | 1,949 |
| barcoorah | 0.392 | 0.494 | 2,433 |
| stockholm | 0.296 | 0.566 | 1,954 |
| nassau | n/a | n/a | 4,417 |
| moroak | n/a | n/a | 3,474 |

### Key findings

**Anchor sites (strong, reliable signal):**
- **Frenchs (0.785)** — standout result. val_auc=0.785 at epoch 1 before the model has learned much (train_auc=0.445), indicating Frenchs pixels are spectrally distinct from the rest of the training distribution. Largest site (15k pixels). Core anchor.
- **Alexandria (0.667), Norman Road (0.663), Lake Mueller (0.657)** — solid, above the overall sweep ceiling. Reliable, discriminative sites. Lake Mueller is the only GPS-surveyed presence site.

**Marginal sites:**
- **Maria Downs (0.624), Pormpuraaw (0.576)** — reasonable but not strong. Maria Downs is large (11k pixels); its moderate score suggests some label noise or spectral ambiguity.

**Below-chance sites (candidates for culling):**
- **Barcoorah (0.392), Stockholm (0.296), Wongalee (0.468), Roper (0.430)** — all below 0.5. The model does actively worse than random when these are held out. Three of the four (Barcoorah, Stockholm, Wongalee) are arid sites, suggesting a systematic arid-zone problem rather than individual label noise.

**Broken runs:**
- **Nassau, Moroak** — returned n/a despite having both presence and absence regions. Likely the val set collapses to a single class after SCL/noise filtering. Logs need inspection before these sites can be assessed.

### Interpretation

The val_auc >> train_auc pattern (Frenchs: 0.785 vs 0.445) suggests some held-out sites are spectrally distinct from the training pool — either genuinely clean signal or climate-zone distribution shift. Frenchs, Alexandria, Norman Road, and Lake Mueller are the load-bearing sites.

The below-chance arid sites (Barcoorah, Stockholm, Wongalee) may reflect a bimodal discrimination problem: arid Parkinsonia is spectrally distinct from monsoonal Parkinsonia, and a single mixed model may not handle both zones well. A separate arid-only model or a zone-conditioned architecture may be required.

### Open questions

- Nassau and Moroak n/a: inspect logs to determine if pixel filtering or class collapse is the cause.
- Arid bimodal hypothesis: run a training experiment on Barcoorah + Stockholm + Wongalee only to determine if arid-zone signal exists independently.
- Anchor-only experiment: train on Frenchs + Alexandria + Norman Road + Lake Mueller only as a clean-signal baseline — establishes the ceiling given current features.
- Quaids false-positive suppression: Quaids absence regions were added specifically to suppress V2 false positives. Dropping them from the anchor-only experiment may increase false positive rates in savanna-like areas at inference time.
