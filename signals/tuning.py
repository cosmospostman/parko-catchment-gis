"""signals/tuning.py — Parameter grid search for individual signals.

Usage
-----
from signals.tuning import sweep_signal
from signals import NirCvSignal

results = sweep_signal(
    NirCvSignal,
    param_grid={
        "scl_purity_min": [0.3, 0.5, 0.7],
        "min_obs_dry":    [3, 5, 8],
    },
    df=df,
    loc=loc,
)
# → DataFrame sorted by |separability| descending, columns:
#   scl_purity_min, min_obs_dry, separability, n_pixels, n_presence, n_absence

Quality keys (``scl_purity_min``, ``min_obs_per_year``, ``min_obs_dry``) are
recognised automatically and routed into QualityParams; all other keys are
passed directly to the signal's Params.

FloweringSignal is handled transparently — its presence/absence split comes
from ``population`` column rather than loc.sub_bboxes.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from signals import QualityParams
from signals.diagnostics import separability_score, _resolve_classes

_QUALITY_KEYS = {"scl_purity_min", "min_obs_per_year", "min_obs_dry"}


def sweep_signal(
    signal_cls: type,
    param_grid: dict[str, list[Any]],
    df: pd.DataFrame,
    loc: object,
) -> pd.DataFrame:
    """Grid search over a signal's parameters, ranked by separability.

    Parameters
    ----------
    signal_cls:
        One of the signal classes, e.g. ``NirCvSignal``, ``FloweringSignal``.
    param_grid:
        Dict mapping parameter names to lists of candidate values.
        Quality keys (``scl_purity_min``, ``min_obs_per_year``,
        ``min_obs_dry``) are routed into ``QualityParams``; all other keys are
        passed directly to the signal's ``Params``.
    df:
        Pre-loaded pixel DataFrame (output of ``pd.read_parquet(loc.parquet_path())``).
    loc:
        ``utils.location.Location`` instance.

    Returns
    -------
    DataFrame with one row per parameter combination, columns:
      - one column per parameter key in ``param_grid``
      - ``separability``  — signed (presence_median − absence_median) / pooled_std
      - ``n_pixels``      — total pixels that survived quality filtering
      - ``n_presence``    — pixels in the presence class
      - ``n_absence``     — pixels in the absence class

    Sorted by ``|separability|`` descending (NaN last).

    Notes
    -----
    Combinations that fail (e.g. too few pixels) produce NaN separability and
    are included at the bottom of the results so you can see where the
    parameter space breaks down.
    """
    from signals.flowering import FloweringSignal

    is_flowering = issubclass(signal_cls, FloweringSignal)

    # Determine which value_col to score (inferred from first default compute)
    _value_col = _infer_value_col(signal_cls)

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    rows = []
    for combo in combos:
        combo_dict = dict(zip(keys, combo))
        row: dict[str, Any] = dict(combo_dict)

        try:
            params = _build_params(signal_cls, combo_dict)
            sig = signal_cls(params)
            stats = sig.compute(df, loc)

            if is_flowering:
                presence_ids = stats.loc[stats["population"] == "infestation", "point_id"]
                absence_ids = stats.loc[stats["population"] == "extension", "point_id"]
            else:
                presence_ids, absence_ids = _resolve_classes(stats, loc)

            sep = separability_score(stats, _value_col, presence_ids, absence_ids)
            n_pres = len(presence_ids) if presence_ids is not None else None
            n_abs = len(absence_ids) if absence_ids is not None else None

            row["separability"] = sep
            row["n_pixels"] = len(stats)
            row["n_presence"] = n_pres
            row["n_absence"] = n_abs

        except Exception as exc:
            row["separability"] = float("nan")
            row["n_pixels"] = None
            row["n_presence"] = None
            row["n_absence"] = None
            row["_error"] = str(exc)

        rows.append(row)

    result = pd.DataFrame(rows)

    # Sort by |separability| descending, NaN last
    result["_abs_sep"] = result["separability"].abs()
    result = result.sort_values("_abs_sep", ascending=False, na_position="last")
    result = result.drop(columns=["_abs_sep"])

    # Drop internal error column if no errors occurred
    if "_error" not in result.columns or result["_error"].isna().all():
        result = result.drop(columns=["_error"], errors="ignore")

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_params(signal_cls: type, combo_dict: dict[str, Any]) -> object:
    """Construct a signal Params instance from a flat combo dict."""
    quality_kwargs = {k: v for k, v in combo_dict.items() if k in _QUALITY_KEYS}
    signal_kwargs = {k: v for k, v in combo_dict.items() if k not in _QUALITY_KEYS}

    quality = QualityParams(**quality_kwargs) if quality_kwargs else QualityParams()
    return signal_cls.Params(quality=quality, **signal_kwargs)


def _infer_value_col(signal_cls: type) -> str:
    """Return the primary value column name for a signal class."""
    # Map by class name — avoids importing each signal at module load time
    _map = {
        "NirCvSignal":          "nir_cv",
        "RecPSignal":           "rec_p",
        "RedEdgeSignal":        "re_p10",
        "SwirSignal":           "swir_p10",
        "FloweringSignal":      "fi_p90_cg",
        "NdviIntegralSignal":   "ndvi_integral",
    }
    name = signal_cls.__name__
    if name not in _map:
        raise ValueError(
            f"Cannot infer value column for {name!r}. "
            f"Known signals: {list(_map)}"
        )
    return _map[name]
