"""utils/pipeline_progress.py — Rich live-display for the fetch pipeline.

One TileYearProgress tracks a single (tile_id, year) pass through the pipeline.
TileProgress holds one TileYearProgress per year and owns the Rich Live context.

Layout (rendered once per refresh tick):

  Fetch 55HBU 2023  ████████░░░░░░░░░░░░  12/48
    r03_c02  S2 fetch   ░░░░░░░░░░░░░░░░░░░░         28s
    r03_c01  extract    ██████████░░░░░░░░░░   27/82  341s
    r03_c01  S1 fetch   ░░░░░░░░░░░░░░░░░░░░         11s
    r03_c01  merge      ░░░░░░░░░░░░░░░░░░░░          3s

  Fetch 55HBU 2024  ██░░░░░░░░░░░░░░░░░░   3/48
    r03_c03  S2 fetch   ░░░░░░░░░░░░░░░░░░░░         14s

Sub-task bars:
  - extract: real bar (done/total scenes) + count + elapsed
  - all others: empty bar (no item-level count) + elapsed only

All state mutations happen in pipeline worker threads; reads happen in the
main thread renderer.  A threading.Lock guards the mutable fields.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Per-stage activity record
# ---------------------------------------------------------------------------

@dataclass
class _StageActivity:
    """One active stage slot, carrying structured fields for the renderer."""
    chunk_id:   str          # "r03_c02"
    stage_name: str          # "S2 fetch" | "S1 fetch" | "extract" | "merge"
    done:       int  = 0     # items completed (extract only)
    total:      int  = 0     # items expected  (extract only)
    label:      str  = ""    # free-form sub-phase label (e.g. "sorting 86M rows")
    t_start:    float = field(default_factory=time.perf_counter)

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.t_start


# ---------------------------------------------------------------------------
# Per (tile, year) progress tracker
# ---------------------------------------------------------------------------

class TileYearProgress:
    """Thread-safe progress state for one (tile_id, year) pipeline run."""

    def __init__(self, tile_id: str, year: int, total_chunks: int = 0) -> None:
        self.tile_id      = tile_id
        self.year         = year
        self.total_chunks = total_chunks
        self._lock        = threading.Lock()
        self._done        = 0   # chunks that produced a parquet
        self._skipped     = 0   # chunks that had no clear pixels
        self._status      = ""  # free-form pre-pipeline status shown in sub-task area
        self._completed:  list[tuple[str, float]] = []   # (chunk_id, finish_time)
        # Keyed by slot name: "fetch_<row>_<col>_<yr>", "extract", "s1", "merge"
        self._active: dict[str, _StageActivity] = {}

    # ------------------------------------------------------------------ writes (worker threads)

    def stage_start(self, slot: str, chunk_id: str, stage_name: str, total: int = 0) -> None:
        with self._lock:
            self._active[slot] = _StageActivity(
                chunk_id=chunk_id,
                stage_name=stage_name,
                total=total,
            )

    def stage_update(self, slot: str, done: int) -> None:
        """Tick the done count for an active slot (used by extract's _as_completed loop)."""
        with self._lock:
            act = self._active.get(slot)
            if act is not None:
                act.done = done

    def stage_set_label(self, slot: str, label: str) -> None:
        """Set a free-form sub-phase label on an active slot (e.g. merge phase names)."""
        with self._lock:
            act = self._active.get(slot)
            if act is not None:
                act.label = label

    def stage_end(self, slot: str) -> None:
        with self._lock:
            self._active.pop(slot, None)

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total_chunks = total
            self._status = ""

    def chunk_done(self, chunk_id: str = "") -> None:
        with self._lock:
            self._done += 1
            if chunk_id:
                self._completed.append((chunk_id, time.time()))

    def chunk_skipped(self, chunk_id: str = "") -> None:
        with self._lock:
            self._skipped += 1
            if chunk_id:
                self._completed.append((chunk_id + " (skip)", time.time()))

    # ------------------------------------------------------------------ reads (main thread)


    def snapshot(self) -> tuple[int, int, int, list[_StageActivity], str, list[str]]:
        """Return (done, skipped, total, activities, status, completed) under lock."""
        with self._lock:
            activities = [
                _StageActivity(
                    chunk_id=a.chunk_id,
                    stage_name=a.stage_name,
                    done=a.done,
                    total=a.total,
                    label=a.label,
                    t_start=a.t_start,
                )
                for a in self._active.values()
            ]
            return self._done, self._skipped, self.total_chunks, activities, self._status, list(self._completed)  # list[tuple[str, float]]


# ---------------------------------------------------------------------------
# Multi-year display — owns the Rich Live context
# ---------------------------------------------------------------------------

class TileProgress:
    """Holds one TileYearProgress per year and owns the Rich Live display."""

    def __init__(
        self,
        tile_id: str,
        years: list[int],
        refresh_per_second: float = 4.0,
    ) -> None:
        self._tile_id = tile_id
        self._years   = years
        self._refresh = refresh_per_second
        self._rows: dict[int, TileYearProgress] = {
            yr: TileYearProgress(tile_id, yr) for yr in years
        }
        self._live = None

    def year(self, yr: int) -> TileYearProgress:
        """Return the TileYearProgress for *yr*, creating a stub row if missing."""
        if yr not in self._rows:
            self._rows[yr] = TileYearProgress(self._tile_id, yr)
            self._years.append(yr)
        return self._rows[yr]

    # ------------------------------------------------------------------ context manager

    def __enter__(self) -> "TileProgress":
        try:
            from rich.live import Live
            from rich.console import Console
            self._live = Live(
                get_renderable=self._render,
                console=Console(stderr=True),
                refresh_per_second=self._refresh,
                transient=False,
            )
            self._live.__enter__()
        except Exception:
            self._live = None
        return self

    def __exit__(self, *args) -> None:
        if self._live is not None:
            try:
                self._live.__exit__(*args)
            except Exception:
                pass

    def refresh(self) -> None:
        """Trigger an immediate repaint (e.g. after a chunk completes)."""
        if self._live is not None:
            try:
                self._live.refresh()
            except Exception:
                pass

    # ------------------------------------------------------------------ renderer

    # Column widths (chars) — chosen so the fixed region fits in 80 cols:
    #   indent(4) + chunk_id(9) + gap(2) + stage(10) + gap(2) + bar(20) + gap(2) + count(7) + gap(2) + elapsed(5)
    #   = 4+9+2+10+2+20+2+7+2+5 = 63  →  fits at 80 with 17 chars spare
    _CHUNK_W  = 9   # "r03_c02  "
    _STAGE_W  = 10  # "S2 fetch  "
    _BAR_W    = 20
    _COUNT_W  = 7   # "27/82  " or blank
    _ELAPS_W  = 5   # "341s "

    def _render(self):
        from rich.text import Text
        from rich.console import Group

        lines: list[Text] = []

        for yr, row in self._rows.items():
            done, skipped, total, activities, status, completed = row.snapshot()
            # Skip years that haven't started yet (no data and no status message).
            if not done and not skipped and not activities and not status and not completed:
                continue

            # ---- completed chunks (most recent first, above header) ---------
            for cid, t_done in completed:
                ts = time.strftime("%H:%M:%S", time.localtime(t_done))
                cl = Text(no_wrap=True)
                cl.append(f"  {ts} ", style="dim")
                cl.append("✓ ", style="bright_green")
                cl.append(cid, style="dim")
                lines.append(cl)

            # ---- header line ------------------------------------------------
            frac    = (done + skipped) / total if total else 0.0
            filled  = int(frac * self._BAR_W)
            bar_txt = "█" * filled + "░" * (self._BAR_W - filled)
            n_str   = (f"{done + skipped}/{total}" if total else "…").rjust(self._COUNT_W)

            header = Text(no_wrap=True)
            header.append(f"Fetch {self._tile_id} {yr}", style="bold")
            header.append("  ")
            header.append(bar_txt, style="green" if frac < 1.0 else "bright_green")
            header.append("  ")
            header.append(n_str, style="dim")
            if frac >= 1.0 and not activities:
                header.append("  done", style="bright_green")
            lines.append(header)

            # ---- status line (pre-pipeline phase) ---------------------------
            if status and not activities:
                st = Text(no_wrap=True)
                st.append(f"    {status}", style="dim italic")
                lines.append(st)

            # ---- sub-task lines ---------------------------------------------
            for act in activities:
                elapsed = act.elapsed_s()

                # Bar: real fill whenever total > 0, empty otherwise
                if act.total > 0:
                    sub_frac   = act.done / act.total
                    sub_filled = int(sub_frac * self._BAR_W)
                    # Show at least one partial block so early progress is visible
                    if sub_filled == 0 and act.done > 0:
                        sub_bar = "▒" + "░" * (self._BAR_W - 1)
                    else:
                        sub_bar = "█" * sub_filled + "░" * (self._BAR_W - sub_filled)
                    count_str  = f"{act.done}/{act.total}".rjust(self._COUNT_W)
                else:
                    sub_bar   = "░" * self._BAR_W
                    count_str = " " * self._COUNT_W

                elaps_str = f"{elapsed:.0f}s".rjust(self._ELAPS_W)

                sub = Text(no_wrap=True)
                sub.append("    ")
                sub.append(act.chunk_id.ljust(self._CHUNK_W), style="dim")
                sub.append("  ")
                sub.append(act.stage_name.ljust(self._STAGE_W), style="dim")
                sub.append("  ")
                sub.append(sub_bar, style="cyan")
                sub.append("  ")
                sub.append(count_str, style="dim")
                sub.append("  ")
                sub.append(elaps_str, style="dim")
                if act.label:
                    sub.append("  ")
                    sub.append(act.label, style="dim")
                lines.append(sub)

            # blank separator between years (skip after last)
            if yr != self._years[-1]:
                lines.append(Text(""))

        return Group(*lines)
