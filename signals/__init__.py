"""signals — shared pixel-sorting and preprocessing utilities.

The signal analysis classes (NirCvSignal, RecPSignal, etc.) have been removed.
Active pipeline code uses signals._shared directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityParams:
    """Quality-filter parameters (kept for any code that imports from signals)."""
    scl_purity_min: float = 0.5
    min_obs_per_year: int = 10
    min_obs_dry: int = 5


__all__ = ["QualityParams"]
