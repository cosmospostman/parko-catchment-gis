# Plan: signals/scl_composition.py — SCL Class Composition Signal

## Context
Raw SCL class values are now stored per observation in new parquets (forward-only from the frenchs re-collection). This enables a time-series signal that describes how a pixel's surface composition changes across the calendar year — water, vegetation, bare soil — without collapsing to a scalar. This is a descriptive signal intended for `describe.py` rather than discriminative use in `explore.py`.

## Signal description

For each pixel, compute the fraction of observations falling into each SCL class, grouped by calendar month, averaged across all years. Produces a 12-month profile per pixel per class — the full seasonal shape of surface composition.

SCL classes of interest (from `analysis/constants.py` `SCL_CLEAR_VALUES = {4, 5, 6, 7, 11}`):
- **4** — vegetation
- **5** — bare soil
- **6** — water
- **7** — unclassified
- **11** — snow/ice (rare in northern Australia, but retained)

No minimum observations threshold — use all available cloud-free observations.

## Output schema

`compute()` returns a long-format DataFrame, one row per (pixel, month):

```
point_id | lon | lat | month | n_obs | scl_veg | scl_bare | scl_water | scl_other
str      | f32 | f32 | int8  | int32 | f32     | f32      | f32       | f32
```

- `n_obs` — number of cloud-free observations in that month across all years
- fractions sum to 1.0 per row (scl_veg + scl_bare + scl_water + scl_other = 1.0)
- `scl_other` = unclassified (7) + snow/ice (11)

## Implementation

### `signals/scl_composition.py`

```python
class SclCompositionSignal:

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)
        # scl_purity_min still applied — only use observations already flagged clear

    def compute(self, pixel_df, loc) -> pd.DataFrame:
        # load_cols: ["point_id", "lon", "lat", "date", "scl_purity", "scl"]
        # filter via load_and_filter() (scl_purity >= min) — adds year/month columns
        # group by (point_id, month), count rows where scl == class / total rows
        # return long-format DataFrame
        ...
```

**Core aggregation (Polars):**
```python
df = load_and_filter(pixel_df, p.quality.scl_purity_min,
                     load_cols=["point_id", "lon", "lat", "date", "scl_purity", "scl"])

monthly = (
    df.group_by(["point_id", "month"])
    .agg([
        pl.len().alias("n_obs"),
        (pl.col("scl") == 4).mean().alias("scl_veg"),
        (pl.col("scl") == 5).mean().alias("scl_bare"),
        (pl.col("scl") == 6).mean().alias("scl_water"),
        pl.col("scl").is_in([7, 11]).mean().alias("scl_other"),
    ])
    .sort(["point_id", "month"])
)
```

Then join coords back in (same pattern as `NirCvSignal`).

### Export

Add to `signals/__init__.py`:
```python
from signals.scl_composition import SclCompositionSignal
```
And to `__all__`.

## Plot: `plot_scl_composition` in `signals/diagnostics.py`

New function — does not fit the existing `plot_signal_map` / `plot_distributions` pattern.

```python
def plot_scl_composition(
    monthly_df: pd.DataFrame,   # output of SclCompositionSignal.compute()
    title: str,
    out_path: Path,
) -> Figure:
```

For `describe.py`, summarise across all pixels in the bbox: median fraction per (month, class) with IQR shading. Render as a stacked area chart:
- x = month (1–12, labelled Jan–Dec)
- y = fraction (0–1)
- stacked layers: water (blue), vegetation (green), bare soil (brown), other (grey)
- IQR shading per layer to show pixel-level spread within the bbox

## Wiring into `describe.py`

Add as a new step after curve-based signals:

```python
from signals.scl_composition import SclCompositionSignal
from signals.diagnostics import plot_scl_composition

scl_monthly = SclCompositionSignal().compute(raw_df_bbox, loc)
plot_scl_composition(scl_monthly, title=..., out_path=out_dir / "scl_composition.png")
scl_monthly.to_csv(out_dir / "scl_composition.csv", index=False)
```

Add a section to the report:
```markdown
## SCL Class Composition
![scl_composition](scl_composition.png)
```

## Graceful degradation

Old parquets without the `scl` column will raise on load. Catch with:
```python
if "scl" not in pixel_df.columns:
    log("  scl_composition: skipping — parquet predates raw SCL storage")
    return pd.DataFrame()
```

## Files touched
- **New**: `signals/scl_composition.py`
- **Modified**: `signals/__init__.py` — add export
- **Modified**: `signals/diagnostics.py` — add `plot_scl_composition`
- **Modified**: `cli/describe.py` — wire in as new step (once describe.py exists)

## Verification
```bash
python -c "
import pandas as pd
from utils.location import get
from signals.scl_composition import SclCompositionSignal
loc = get('frenchs')
df = pd.read_parquet(loc.parquet_path())
result = SclCompositionSignal().compute(df, loc)
print(result.head(24))
assert set(result.columns) >= {'point_id', 'month', 'scl_water', 'scl_veg', 'scl_bare'}
assert (result[['scl_veg','scl_bare','scl_water','scl_other']].sum(axis=1) - 1.0).abs().max() < 1e-4
"
```
