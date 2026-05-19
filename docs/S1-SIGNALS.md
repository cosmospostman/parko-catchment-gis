# S1 SAR Signals — Evaluation and Integration Strategy

## Motivation

V9-spectral (S2-only, d_model=256, n_layers=3) achieves CVaR25=0.824 across the full
multi-site val set. The persistent weak site is **Etna Creek** (~0.78 AUC), which has
been in the CVaR tail across every training run.

Attention visualisation (`tam/viz_attention.py`) on the V9 checkpoint shows identical
per-head attention profiles for Etna presence and absence across all four heads — the
model attends to the same temporal windows for both classes. The root cause is structural:
at Etna, Parkinsonia is often the only woody vegetation for miles. Presence and absence
pixels go through the same seasonal greenness cycle; the difference is amplitude
(how green), not timing. Temporal attention cannot discriminate them.

S1 VH backscatter responds to canopy volume structure rather than greenness. A Parkinsonia
canopy produces consistent elevated VH returns year-round — including during the dry season
when surrounding grass has senesced and collapsed. This is the signal that temporal attention
cannot see. The hypothesis is that S1 features would break Etna's ceiling without degrading
the sites where V9 already performs well.

---

## Evaluation strategy

Before committing to a full S1 model training run, evaluate S1 signals against labeled
regions using the `signals/` harness. This answers the discriminability question in hours
rather than epochs.

### Signals to implement (`signals/s1.py`)

Four signals corresponding to the S1_FEATURE_COLS used in the TAM pipeline:

| Signal | Formulation | Rank key | Hypothesis |
|---|---|---|---|
| `vh_db` | 10·log10(VH_linear) | `dry_mean` | Woody canopy volume scattering; elevated year-round vs grass |
| `vv_db` | 10·log10(VV_linear) | `dry_mean` | Secondary structural signal; less sensitive than VH to canopy |
| `vh_vv` | VH_db − VV_db | `dry_mean` | Cross-pol ratio; sensitive to canopy randomness/disorder |
| `rvi` | 4·VH_lin / (VH_lin + VV_lin) | `dry_mean` | Radar Vegetation Index; high for structurally complex canopy |

All four require a **dry-season windowed `summarise()` override** — the discrimination
window for S1 is the dry season (DOY ~121–304), when grass backscatter collapses but
Parkinsonia's woody structure maintains elevated returns. The base `summarise()` is
not sufficient; S1 signals must implement seasonal windowing.

### Implementation notes

The `Signal` base class `quality_mask()` filters to `source == "S2"`. S1 signals must
implement their own mask: `source == "S1"`, no SCL filter (S1 has no cloud mask). All
other harness machinery (`compute()`, `summarise()`, `eval.py`) works unchanged.

### Site comparisons

Priority sites for evaluation, chosen to span the main failure modes:

```python
from signals.eval import SiteSpec

sites = [
    # Etna — the primary hypothesis site; grass vs Parkinsonia
    SiteSpec("etna", [
        ("etna_presence_2",  "presence"),
        ("etna_presence_5",  "presence"),
        ("etna_presence_6",  "presence"),
        ("etna_absence_6",   "absence"),
        ("etna_absence_7",   "absence"),
        ("etna_absence_8",   "absence"),   # woody absence
    ]),
    # Landsend — semi-arid riparian; sparse presence vs grass/riverbed
    SiteSpec("landsend", [
        ("landsend_sparse_presence_1", "presence"),
        ("landsend_sparse_presence_2", "presence"),
        ("landsend_absence_grass_1",   "absence"),
        ("landsend_absence_riverbed_1","absence"),
    ]),
    # Frenchs — monsoonal; Parkinsonia in woody matrix
    SiteSpec("frenchs", [
        ("frenchs_presence_1",            "presence"),
        ("frenchs_presence_2",            "presence"),
        ("frenchs_absence_riparian",      "absence"),
        ("frenchs_absence_riparian_woodland", "absence"),
    ]),
    # Burdekin — coastal irrigated; cropland false positives
    SiteSpec("burdekin", [
        ("burdekin_presence_1",    "presence"),
        ("burdekin_absence_4",     "absence"),  # cropland
        ("burdekin_absence_5",     "absence"),  # cropland
        ("burdekin_absence_8",     "absence"),  # woodland
    ]),
]
```

### Interpretation

- **AUROC > 0.75** at Etna for VH dry-season mean → hypothesis confirmed, S1 adds signal
- **AUROC < 0.6** at Etna → S1 doesn't discriminate at this site; look elsewhere
- Check direction: lower VH_db = absence (grass), higher = presence (woody). If AUROC < 0.5,
  invert the rank key or use `1 - AUROC`
- Frenchs and Burdekin results inform whether S1 helps or hurts in those contexts before
  any model integration

### Data prerequisite

S1 pixel data must be present in the training tile parquets for evaluation to work.
The eval harness reads from region parquets via `_region_parquet_path`. Confirm S1 rows
(`source == "S1"`) are present before running:

```python
import pyarrow.parquet as pq
from utils.training_collector import _region_parquet_path

pf = pq.ParquetFile(_region_parquet_path("etna_presence_5"))
df = pf.read_row_group(0).to_pandas()
print(df["source"].value_counts())  # should show S1 and S2 rows
```

---

## Integration pathway

Depending on eval results:

### If S1 discriminates at Etna (AUROC > 0.75)

Add S1 features to V9 as a joint S1+S2 model. The pipeline already supports this via
`use_s1=True` (snap mode) or `use_s1="s1_only"`. For joint mode:

- `use_s1=True` — S1 observations are snapped to nearest S2 date within ±7 days and
  appended as four additional band columns (VH, VV, VH−VV, RVI). The sequence input
  becomes 15 bands (11 S2 + 4 S1).
- `use_band_summaries=True` remains valid — band summaries are computed from S2 rows only,
  so the global feature pathway is unaffected.
- S1 data must be fetched for all training tiles before a joint run. Frenchs, Hughenden,
  Burdekin, Maria Downs, and Rupert Creek currently have no S1 parquets — run the pixel
  collector against all region IDs and verify outputs.

Suggested first joint run (inherit V9 256/3 architecture):

```
python -m tam.pipeline train --experiment v9_spectral \
    --spatial-stride 2 --epochs 100 --patience 20 \
    --output-dir outputs/models/tam-v9_joint
```

with `use_s1=True` added to `v9_spectral.py` train_kwargs.

### If S1 does not discriminate at Etna

Etna's ceiling is likely a label or data quality issue rather than a missing feature.
Consider:
- Inspecting Etna presence regions in Google Earth for mixed/sparse canopy
- Adding more Etna woody absence to force harder discrimination
- Accepting ~0.78 as the Etna ceiling given the inherent difficulty of the site

### S1-only model (V10-SAR)

Deferred until after joint model evaluation. If the joint model outperforms V9 at Etna,
an S1-only model is useful for deployment contexts where S2 data is unavailable or
cloud-contaminated. Hyperparameters from V8 with V9 site set and training improvements
(doy_phase_shift=False, per-year VH filter, max_seq_len=64, band_noise_std=0.05).

---

## Status

- [ ] Implement `signals/s1.py` — VH, VV, VH−VV, RVI with dry-season windowed summarise()
- [ ] Verify S1 rows present in Etna/Landsend region parquets
- [ ] Run eval harness across priority sites
- [ ] Interpret results and decide on integration pathway
