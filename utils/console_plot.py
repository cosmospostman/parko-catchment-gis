"""utils/console_plot.py — Text-based console plots for signal inspection.

Three standalone functions:

    plot_waveform(obs_list, index_fn)       — per-point spectral index time-series
    plot_distributions(presence, absence)   — presence vs. absence histogram
    plot_doy_calendar(doy_vals)             — month-bucketed DOY histogram

All print to stdout. No side effects beyond printing. No external dependencies
beyond stdlib; rich is used for colour if installed but is not required.
"""

from __future__ import annotations

import statistics
from typing import Callable


# ---------------------------------------------------------------------------
# Colour helpers (graceful fallback if rich not present)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console as _RichConsole
    _console = _RichConsole(highlight=False)

    def _print(text: str = "", markup: bool = False) -> None:
        _console.print(text, markup=markup)

except ImportError:
    def _print(text: str = "", markup: bool = False) -> None:  # type: ignore[misc]
        print(text)


# ---------------------------------------------------------------------------
# 1. plot_waveform
# ---------------------------------------------------------------------------

_FLOWERING_WINDOW = (213, 304)   # Aug–Oct (DOY)


def _doy(d: object) -> int:
    """Return day-of-year from a date or datetime."""
    d_date = d.date() if hasattr(d, "date") else d  # type: ignore[union-attr]
    return d_date.timetuple().tm_yday


def plot_waveform(
    obs_list: list,
    index_fn: Callable[[dict], float],
    *,
    width: int = 80,
    height: int = 20,
    title: str = "",
) -> None:
    """Render a per-point spectral index time-series.

    Parameters
    ----------
    obs_list:
        List of Observation objects (any order; sorted internally by date).
    index_fn:
        Function with signature (bands: dict[str, float]) -> float.
        Typically analysis.primitives.indices.flowering_index.
    width:
        Target character width of the plot area (columns for observations).
    height:
        Number of rows for the Y axis.
    title:
        Optional header line.
    """
    if not obs_list:
        _print("(no observations)")
        return

    obs_sorted = sorted(obs_list, key=lambda o: o.date)
    n = len(obs_sorted)

    values = [index_fn(o.bands) for o in obs_sorted]
    qualities = [o.quality.score() for o in obs_sorted]

    y_min = min(-1.0, min(values))
    y_max = max(1.0, max(values))
    y_range = y_max - y_min or 1.0

    # 2-D grid: grid[row][col] = char
    grid = [[" "] * n for _ in range(height)]

    def _row(v: float) -> int:
        frac = (v - y_min) / y_range
        r = height - 1 - int(frac * (height - 1))
        return max(0, min(height - 1, r))

    zero_row = _row(0.0)

    for col, (obs, val, qual) in enumerate(zip(obs_sorted, values, qualities)):
        r = _row(val)
        doy = _doy(obs.date)
        in_window = _FLOWERING_WINDOW[0] <= doy <= _FLOWERING_WINDOW[1]
        marker = "●" if qual >= 0.5 else "·"
        grid[r][col] = marker
        if in_window and grid[zero_row][col] == " ":
            grid[zero_row][col] = "▸"

    label_width = 6

    if title:
        _print(f" {title}")

    for r in range(height):
        v = y_max - r * (y_range / (height - 1))
        label = f"{v:+.2f}" if r % max(1, height // 5) == 0 else "     "
        row_chars = "".join(grid[r])
        sep = "┤" if label.strip() else "│"
        _print(f"{label:>{label_width}}{sep}{row_chars}")

    # Bottom axis: year tick marks
    tick_row = [" "] * n
    label_row = [" "] * n
    prev_year = None
    for col, obs in enumerate(obs_sorted):
        d = obs.date.date() if hasattr(obs.date, "date") else obs.date
        year = d.year
        if year != prev_year:
            tick_row[col] = "┬"
            for i, ch in enumerate(str(year)):
                if col + i < n:
                    label_row[col + i] = ch
            prev_year = year

    _print(" " * label_width + "└" + "".join(tick_row))
    _print(" " * (label_width + 1) + "".join(label_row))

    _print(
        f"\n  ● high-quality (score≥0.5)  · low-quality  "
        f"▸ Aug–Oct window (DOY {_FLOWERING_WINDOW[0]}–{_FLOWERING_WINDOW[1]})  n={n}"
    )


# ---------------------------------------------------------------------------
# 2. plot_distributions
# ---------------------------------------------------------------------------

def plot_distributions(
    presence_vals: list[float],
    absence_vals: list[float],
    *,
    width: int = 60,
    bins: int = 20,
    title: str = "",
) -> None:
    """ASCII histogram comparing two scalar distributions.

    Parameters
    ----------
    presence_vals, absence_vals:
        Lists of floats (e.g. peak_value for each group).
    width:
        Maximum bar width in characters.
    bins:
        Number of histogram bins.
    title:
        Optional header line.
    """
    all_vals = presence_vals + absence_vals
    if not all_vals:
        _print("(no data)")
        return

    v_min = min(all_vals)
    v_max = max(all_vals)
    if v_min == v_max:
        v_max = v_min + 1.0

    bin_w = (v_max - v_min) / bins
    edges = [v_min + i * bin_w for i in range(bins + 1)]

    def _hist(vals: list[float]) -> list[int]:
        counts = [0] * bins
        for v in vals:
            idx = min(int((v - v_min) / bin_w), bins - 1)
            counts[idx] += 1
        return counts

    pres_counts = _hist(presence_vals)
    abs_counts = _hist(absence_vals)
    max_count = max(max(pres_counts, default=0), max(abs_counts, default=0), 1)

    label_w = 10
    if title:
        _print(f" {title}")
    _print(f"{'':>{label_w}}  {'presence (●)':<{width}}  absence (░)")
    _print(f"{'':>{label_w}}  {'─' * width}  {'─' * width}")

    for i, (pc, ac) in enumerate(zip(pres_counts, abs_counts)):
        lo = edges[i]
        label = f"{lo:+.3f}"
        pb = round(pc / max_count * width)
        ab = round(ac / max_count * width)
        _print(
            f"{label:>{label_w}}  {'█' * pb:<{width}}  {'░' * ab:<{width}}  {pc:>3}/{ac:>3}"
        )

    pres_med = statistics.median(presence_vals) if presence_vals else float("nan")
    abs_med = statistics.median(absence_vals) if absence_vals else float("nan")
    _print(
        f"\n  presence median={pres_med:.4f} (n={len(presence_vals)})  "
        f"absence median={abs_med:.4f} (n={len(absence_vals)})"
    )


# ---------------------------------------------------------------------------
# 3. plot_doy_calendar
# ---------------------------------------------------------------------------

_MONTHS = [
    ("Jan",   1,  31),
    ("Feb",  32,  59),
    ("Mar",  60,  90),
    ("Apr",  91, 120),
    ("May", 121, 151),
    ("Jun", 152, 181),
    ("Jul", 182, 212),
    ("Aug", 213, 243),
    ("Sep", 244, 273),
    ("Oct", 274, 304),
    ("Nov", 305, 334),
    ("Dec", 335, 365),
]

_WINDOW_MONTHS = {"Aug", "Sep", "Oct"}


def plot_doy_calendar(
    doy_vals: list[int],
    *,
    title: str = "",
    bar_width: int = 40,
) -> None:
    """Month-bucketed DOY histogram.

    Parameters
    ----------
    doy_vals:
        List of day-of-year integers (1–365).
    title:
        Optional header line.
    bar_width:
        Maximum bar width in characters.
    """
    if not doy_vals:
        _print("(no DOY values)")
        return

    counts = {name: 0 for name, _, _ in _MONTHS}
    for doy in doy_vals:
        for name, lo, hi in _MONTHS:
            if lo <= doy <= hi:
                counts[name] += 1
                break

    max_count = max(counts.values(), default=1) or 1

    if title:
        _print(f" {title}")
    _print(f"  n={len(doy_vals)}\n")

    for name, lo, hi in _MONTHS:
        c = counts[name]
        bar_len = round(c / max_count * bar_width)
        bar = "█" * bar_len
        bracket = " ◀ flowering window" if name in _WINDOW_MONTHS else ""
        _print(f"  {name} ({lo:>3}–{hi:>3})  {bar:<{bar_width}}  {c:>3}{bracket}")

    _print()
