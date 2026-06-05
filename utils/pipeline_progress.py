"""utils/pipeline_progress.py — Rich live-display for the fetch pipeline.

Three fixed lines (one per thread) plus a scrolling completion history:

  13:04:17 ✓ 54LWH 2020 r05_c04  (4m 12s)
  13:11:32 ✓ 54LWH 2020 r05_c05  (6m 40s)
FETCH    54LWH 2020 r05_c06  S2 fetch  ████████████░░░░░░░░  79/116  640s
PROCESS  54LWH 2020 r05_c05  sort      ░░░░░░░░░░░░░░░░░░░░          166s
COPY     54LWH 2020 r05_c05            ████████████░░░░░░░░            8s
QUEUED   r05_c06  r05_c07

Each thread reports its state via a single update method.  The renderer reads
a snapshot of all three states under one lock and formats the fixed layout.

Exhaustive stage values per thread:
  FETCH:   "waiting" | "S2 fetch" | "S1 fetch" | "done"
  PROCESS: "waiting" | "S2 extract" | "S1 extract" | "reading" | "sorting" | "writing" | "done"
  COPY:    "waiting" | "copy" | "done"
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# State records
# ---------------------------------------------------------------------------

@dataclass
class ThreadState:
    """Current state of one pipeline thread."""
    thread:   str   = ""    # "FETCH" | "PROCESS" | "COPY"
    tile_id:  str   = ""
    year:     int   = 0
    chunk_id: str   = ""
    stage:    str   = "waiting"
    done:     int   = 0
    total:    int   = 0
    label:    str   = ""    # sub-label shown after elapsed (e.g. "82 files")
    t_start:  float = field(default_factory=time.perf_counter)


@dataclass
class _CompletedChunk:
    tile_id:   str
    year:      int
    chunk_id:  str
    wall_time: float    # time.time() at completion
    active_s:  float    # cumulative non-waiting thread-time across all three threads


# ---------------------------------------------------------------------------
# Per-tile progress tracker
# ---------------------------------------------------------------------------

class TileProgress:
    """Thread-safe progress state for one tile pipeline run.

    Worker threads call fetch_update / process_update / copy_update to report
    their current state.  The Rich renderer calls snapshot() to read all state
    atomically.
    """

    def __init__(
        self,
        tile_id: str,
        years: list[int],
        refresh_per_second: float = 4.0,
    ) -> None:
        self._tile_id   = tile_id
        self._years     = list(years)
        self._refresh   = refresh_per_second
        self._lock      = threading.Lock()
        self._live      = None

        # Thread states
        self._fetch   = ThreadState(thread="FETCH",   stage="waiting")
        self._process = ThreadState(thread="PROCESS", stage="waiting")
        self._copy    = ThreadState(thread="COPY",    stage="waiting")

        # Queue: (tile_id, year, chunk_id) tuples fetched but not yet picked up by process
        self._queued: list[tuple[str, int, str]] = []

        # Completion history
        self._completed: list[_CompletedChunk] = []

        # Chunk counters (for set_total / status compat)
        self._total_chunks: dict[int, int] = {}
        self._done_chunks:  dict[int, int] = {}
        self._status: str = ""

        # Per-chunk active-time accumulators (keyed by chunk_id)
        # Each thread adds its elapsed non-waiting time here before transitioning.
        self._chunk_active: dict[str, float] = {}

    # ------------------------------------------------------------------
    # FETCH thread API
    # ------------------------------------------------------------------

    def fetch_update(
        self,
        stage: str,
        tile_id: str = "",
        year: int = 0,
        chunk_id: str = "",
        done: int = 0,
        total: int = 0,
        label: str = "",
    ) -> None:
        with self._lock:
            s = self._fetch
            if stage != s.stage:
                # Accumulate elapsed time for the chunk we're leaving
                if s.stage not in ("waiting", "done") and s.chunk_id:
                    elapsed = time.perf_counter() - s.t_start
                    self._chunk_active[s.chunk_id] = (
                        self._chunk_active.get(s.chunk_id, 0.0) + elapsed
                    )
                s.t_start = time.perf_counter()
            s.stage    = stage
            s.tile_id  = tile_id  or s.tile_id
            s.year     = year     or s.year
            s.chunk_id = chunk_id or s.chunk_id
            s.done     = done
            s.total    = total
            s.label    = label

    def chunk_fetched(self, chunk_id: str) -> None:
        """Called by fetch thread after posting to _fetched_q."""
        with self._lock:
            self._queued.append((self._fetch.tile_id, self._fetch.year, chunk_id))

    # ------------------------------------------------------------------
    # PROCESS thread API
    # ------------------------------------------------------------------

    def process_update(
        self,
        stage: str,
        tile_id: str = "",
        year: int = 0,
        chunk_id: str = "",
        done: int = 0,
        total: int = 0,
        label: str = "",
    ) -> None:
        with self._lock:
            s = self._process
            if stage != s.stage:
                if s.stage not in ("waiting", "done") and s.chunk_id:
                    elapsed = time.perf_counter() - s.t_start
                    self._chunk_active[s.chunk_id] = (
                        self._chunk_active.get(s.chunk_id, 0.0) + elapsed
                    )
                s.t_start = time.perf_counter()
            s.stage    = stage
            s.tile_id  = tile_id  or s.tile_id
            s.year     = year     or s.year
            s.chunk_id = chunk_id or s.chunk_id
            s.done     = done
            s.total    = total
            s.label    = label

    def chunk_dequeued(self, chunk_id: str) -> None:
        """Called by process thread after taking work from _fetched_q."""
        with self._lock:
            self._queued = [(t, y, c) for t, y, c in self._queued if c != chunk_id]

    def chunk_done(self, chunk_id: str, tile_id: str = "", year: int = 0) -> None:
        """Called by process thread when a chunk's parquet is ready on NVMe."""
        with self._lock:
            active_s = self._chunk_active.pop(chunk_id, 0.0)
            # Add copy thread's elapsed if it already finished (unlikely but possible)
            _tid = tile_id or self._tile_id
            _yr  = year or self._process.year
            self._completed.append(_CompletedChunk(
                tile_id=_tid,
                year=_yr,
                chunk_id=chunk_id,
                wall_time=time.time(),
                active_s=active_s,
            ))
            self._done_chunks[_yr] = self._done_chunks.get(_yr, 0) + 1

    # ------------------------------------------------------------------
    # COPY thread API
    # ------------------------------------------------------------------

    def copy_update(
        self,
        stage: str,
        tile_id: str = "",
        year: int = 0,
        chunk_id: str = "",
    ) -> None:
        with self._lock:
            s = self._copy
            if stage != s.stage:
                if s.stage not in ("waiting", "done") and s.chunk_id:
                    elapsed = time.perf_counter() - s.t_start
                    self._chunk_active[s.chunk_id] = (
                        self._chunk_active.get(s.chunk_id, 0.0) + elapsed
                    )
                s.t_start = time.perf_counter()
            s.stage    = stage
            s.tile_id  = tile_id  or s.tile_id
            s.year     = year     or s.year
            s.chunk_id = chunk_id or s.chunk_id
            s.done     = 0
            s.total    = 0
            s.label    = ""

    # ------------------------------------------------------------------
    # Compat helpers (used by run_tile_pipeline_v2 set_total / set_status)
    # ------------------------------------------------------------------

    def set_total(self, year: int, total: int) -> None:
        with self._lock:
            self._total_chunks[year] = total
            self._status = ""

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def chunk_skipped(self, chunk_id: str, year: int = 0) -> None:
        with self._lock:
            _yr = year or self._process.year
            self._done_chunks[_yr] = self._done_chunks.get(_yr, 0) + 1
            self._completed.append(_CompletedChunk(
                tile_id=self._tile_id,
                year=_yr,
                chunk_id=chunk_id + " (skip)",
                wall_time=time.time(),
                active_s=self._chunk_active.pop(chunk_id, 0.0),
            ))

    def year(self, yr: int) -> "TileProgress":
        """Compat shim — returns self since TileProgress is now flat (one tile)."""
        return self

    # ------------------------------------------------------------------
    # Snapshot (renderer thread)
    # ------------------------------------------------------------------

    def snapshot(self):
        """Return (fetch, process, copy, queued, completed, total_by_yr, done_by_yr, status) under lock."""
        with self._lock:
            def _copy_state(s: ThreadState) -> ThreadState:
                return ThreadState(
                    thread=s.thread, tile_id=s.tile_id, year=s.year,
                    chunk_id=s.chunk_id, stage=s.stage,
                    done=s.done, total=s.total, label=s.label, t_start=s.t_start,
                )
            return (
                _copy_state(self._fetch),
                _copy_state(self._process),
                _copy_state(self._copy),
                list(self._queued),
                list(self._completed),
                dict(self._total_chunks),
                dict(self._done_chunks),
                self._status,
            )

    # ------------------------------------------------------------------
    # Rich Live context manager
    # ------------------------------------------------------------------

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
        if self._live is not None:
            try:
                self._live.refresh()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Renderer
    # ------------------------------------------------------------------

    # Column widths — label(8) + gap(1) + tile(6) + gap(1) + year(4) + gap(1) +
    #   chunk(9) + gap(2) + stage(10) + gap(2) + bar(20) + gap(2) + count(9) + gap(2) + elapsed(6)
    # = 8+1+6+1+4+1+9+2+10+2+20+2+9+2+6 = 83 chars
    _LABEL_W  = 8
    _TILE_W   = 6
    _YEAR_W   = 4
    _CHUNK_W  = 9
    _STAGE_W  = 10
    _BAR_W    = 20
    _COUNT_W  = 9
    _ELAPS_W  = 6

    _THREAD_STYLE = {
        "FETCH":   "bold blue",
        "PROCESS": "bold yellow",
        "COPY":    "bold magenta",
    }

    def _fmt_elapsed(self, seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s2 = divmod(s, 60)
        if m < 60:
            return f"{m}m{s2:02d}s"
        h, m2 = divmod(m, 60)
        return f"{h}h{m2:02d}m"

    def _render_thread_line(self, s: ThreadState) -> "Text":
        from rich.text import Text

        now    = time.perf_counter()
        elaps  = now - s.t_start
        style  = self._THREAD_STYLE.get(s.thread, "bold")
        line   = Text(no_wrap=True)

        label_str = s.thread.ljust(self._LABEL_W)

        if s.stage in ("waiting", "done", ""):
            line.append(label_str, style=style + " dim")
            if s.stage == "waiting":
                line.append(" —", style="dim")
            elif s.stage == "done":
                line.append(" done", style="dim")
            return line

        line.append(label_str, style=style)
        line.append(" ")
        line.append(s.tile_id.ljust(self._TILE_W), style="dim")
        line.append(" ")
        line.append(str(s.year).ljust(self._YEAR_W), style="dim")
        line.append(" ")
        line.append(s.chunk_id.ljust(self._CHUNK_W), style="dim")
        line.append("  ")
        line.append(s.stage.ljust(self._STAGE_W), style="dim")
        line.append("  ")

        # Progress bar
        if s.total > 0:
            frac   = min(s.done / s.total, 1.0)
            filled = int(frac * self._BAR_W)
            if filled == 0 and s.done > 0:
                bar = "▒" + "░" * (self._BAR_W - 1)
            else:
                bar = "█" * filled + "░" * (self._BAR_W - filled)
            count_str = f"{s.done}/{s.total}".rjust(self._COUNT_W)
        else:
            bar       = "░" * self._BAR_W
            count_str = " " * self._COUNT_W

        line.append(bar, style="cyan")
        line.append("  ")
        line.append(count_str, style="dim")
        line.append("  ")
        line.append(self._fmt_elapsed(elaps).rjust(self._ELAPS_W), style="dim")
        if s.label:
            line.append("  ")
            line.append(s.label, style="dim")
        return line

    def _render(self):
        from rich.text import Text
        from rich.console import Group

        fetch, process, copy, queued, completed, total_by_yr, done_by_yr, status = self.snapshot()
        lines: list[Text] = []

        # ---- completed chunk history (most recent first) --------------------
        for cc in completed:
            ts = time.strftime("%H:%M:%S", time.localtime(cc.wall_time))
            active_str = self._fmt_elapsed(cc.active_s)
            cl = Text(no_wrap=True)
            cl.append(f"  {ts} ", style="dim")
            cl.append("✓ ", style="bright_green")
            cl.append(f"{cc.tile_id} {cc.year} {cc.chunk_id}  ({active_str})", style="dim")
            lines.append(cl)

        # ---- overall progress header (if total known) -----------------------
        for yr in self._years:
            total = total_by_yr.get(yr, 0)
            done  = done_by_yr.get(yr, 0)
            if total > 0:
                frac    = min(done / total, 1.0)
                filled  = int(frac * self._BAR_W)
                bar_txt = "█" * filled + "░" * (self._BAR_W - filled)
                n_str   = f"{done}/{total}".rjust(self._COUNT_W)
                hdr = Text(no_wrap=True)
                hdr.append(f"{self._tile_id} {yr}", style="bold")
                hdr.append("  ")
                hdr.append(bar_txt, style="green" if frac < 1.0 else "bright_green")
                hdr.append("  ")
                hdr.append(n_str, style="dim")
                if frac >= 1.0:
                    hdr.append("  done", style="bright_green")
                lines.append(hdr)

        # ---- status (pre-pipeline) ------------------------------------------
        if status:
            st = Text(no_wrap=True)
            st.append(f"  {status}", style="dim italic")
            lines.append(st)

        # ---- three fixed thread lines ---------------------------------------
        lines.append(self._render_thread_line(fetch))
        lines.append(self._render_thread_line(process))
        lines.append(self._render_thread_line(copy))

        # ---- queued / pending line -----------------------------------------
        # "process queue": chunks fully fetched, waiting for process to pick up
        # "fetch remaining": total − done − in-process − in-copy − in-process-queue
        total_all = sum(total_by_yr.values())
        done_all  = sum(done_by_yr.values())
        in_flight = (
            (1 if process.stage not in ("waiting", "done", "") else 0)
            + (1 if copy.stage    not in ("waiting", "done", "") else 0)
            + len(queued)
        )
        fetch_remaining = max(0, total_all - done_all - in_flight)

        show_queued  = bool(queued)
        show_pending = fetch_remaining > 0 and fetch.stage not in ("done", "")

        if show_queued or show_pending:
            q = Text(no_wrap=True)
            q.append("QUEUED  ".ljust(self._LABEL_W + 1), style="dim")
            parts: list[str] = []
            if show_queued:
                cids = "  ".join(f"{t}:{y}:{c}" for t, y, c in queued)
                parts.append(f"{cids}  → process")
            if show_pending:
                parts.append(f"{fetch_remaining} remaining to fetch")
            q.append("  ·  ".join(parts), style="dim")
            lines.append(q)

        return Group(*lines)


# ---------------------------------------------------------------------------
# Backward-compat shim — TileYearProgress was used in a few old paths
# ---------------------------------------------------------------------------

class TileYearProgress:
    """Thin shim that delegates to TileProgress for backward compatibility."""

    def __init__(self, tile_id: str, year: int, total_chunks: int = 0) -> None:
        self.tile_id      = tile_id
        self.year         = year
        self.total_chunks = total_chunks
        self._owner: TileProgress | None = None

    def _bind(self, owner: TileProgress) -> None:
        self._owner = owner

    def set_total(self, total: int) -> None:
        if self._owner:
            self._owner.set_total(self.year, total)

    def set_status(self, status: str) -> None:
        if self._owner:
            self._owner.set_status(status)

    def chunk_done(self, chunk_id: str = "") -> None:
        if self._owner:
            self._owner.chunk_done(chunk_id, year=self.year)

    def chunk_skipped(self, chunk_id: str = "") -> None:
        if self._owner:
            self._owner.chunk_skipped(chunk_id, year=self.year)
