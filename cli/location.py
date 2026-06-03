"""cli/location.py — Location registry introspection and fetch triggering.

Usage
-----
  python cli/location.py list
  python cli/location.py info <id>
  python cli/location.py bbox <id>
  python cli/location.py fetch <id> --years YYYY [YYYY ...]
                                     [--cloud-max N] [--no-nbar]
  python cli/location.py training list
  python cli/location.py training fetch [--regions ID ...] [--all]
                                         [--cloud-max N] [--no-nbar]
  python cli/location.py training verify
  python cli/location.py validate <id> [--year YYYY ...] [--verbose]

Examples
--------
  python cli/location.py list
  python cli/location.py info longreach
  python cli/location.py bbox muttaburra
  python cli/location.py fetch longreach --years 2020 2021 2022
  python cli/location.py fetch longreach --years 2024
  python cli/location.py training list
  python cli/location.py training fetch --all
  python cli/location.py training fetch --regions lake_mueller_presence barcoorah_presence
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Delightful progress formatter
# ---------------------------------------------------------------------------

_BAR_WIDTH = 28

# Progress log patterns — matched in _FetchHandler.emit() before formatting.
# Each pattern's group 1 is the kind label shown in the progress line.
_PAT_ITEM_PROGRESS = re.compile(
    r"(S[12] scenes)\s+shard (\d+)/(\d+)\s+item (\d+)/(\d+)\s+(\d+) rows(?:\s+workers (\d+)/(\d+))?"
)
_PAT_CHIP_PROGRESS = re.compile(
    r"(S2 chips)\s+fetch (\d+)/(\d+) patches done"
)
_PAT_CONCAT_PROGRESS = re.compile(
    r"(S2 merge)\s+concat (\d+)/(\d+) row groups \(([\d.]+)%\)\s+([\d,]+) rows written"
)


def _bar(done: int, total: int, width: int = _BAR_WIDTH) -> str:
    frac = done / total if total else 0
    filled = round(frac * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _strip_ansi(s: str) -> str:
    return re.sub(r"\033\[[^m]*m", "", s)


# ---------------------------------------------------------------------------
# Shared progress state — aggregates across all concurrent shards/strips/years
# ---------------------------------------------------------------------------

import threading as _threading


class _ProgressState:
    """Thread-safe aggregate of in-flight extraction progress across all units.

    Each "unit" is identified by a (thread_id, shard_i) key.  On every
    progress tick a unit updates its slot.  The renderer reads all slots to
    produce one consolidated \r line showing combined throughput across every
    active shard running concurrently.
    """

    def __init__(self) -> None:
        self._lock = _threading.Lock()
        self._slots: dict[tuple, dict] = {}

    def update_item(
        self,
        label: str,
        shard_i: int, shard_n: int,
        item_i: int, item_n: int,
        rows: int,
        workers_active: int | None,
        workers_max: int | None,
    ) -> None:
        key = (_threading.get_ident(), shard_i)
        with self._lock:
            self._slots[key] = dict(
                kind="item", label=label,
                shard_i=shard_i, shard_n=shard_n,
                item_i=item_i, item_n=item_n,
                rows=rows,
                workers_active=workers_active,
                workers_max=workers_max,
            )

    def update_chips(self, label: str, done: int, total: int) -> None:
        key = (_threading.get_ident(), -2)
        with self._lock:
            self._slots[key] = dict(kind="chips", label=label, done=done, total=total)

    def update_concat(self, label: str, rg_i: int, rg_n: int, rows: int) -> None:
        key = (_threading.get_ident(), -1)
        with self._lock:
            self._slots[key] = dict(kind="concat", label=label, rg_i=rg_i, rg_n=rg_n, rows=rows)

    def complete_unit(self, shard_i: int) -> None:
        key = (_threading.get_ident(), shard_i)
        with self._lock:
            self._slots.pop(key, None)

    def clear_chips(self) -> None:
        with self._lock:
            self._slots.pop((_threading.get_ident(), -2), None)

    def clear_concat(self) -> None:
        with self._lock:
            self._slots.pop((_threading.get_ident(), -1), None)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(slots=dict(self._slots))

    def set_total(self, n: int) -> None:
        pass

    def reset(self) -> None:
        with self._lock:
            self._slots.clear()


_progress = _ProgressState()


class _FetchFormatter(logging.Formatter):
    """ANSI-coloured, human-readable formatter for the fetch pipeline."""

    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _DIM    = "\033[2m"
    _CYAN   = "\033[36m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"
    _RED    = "\033[31m"
    _BLUE   = "\033[34m"
    _MAG    = "\033[35m"

    # Static patterns for normal (newline-terminated) lines.
    _PATTERNS: list[tuple[re.Pattern, str, str, int]] = [
        # ── Memory / budget ──────────────────────────────────────────────────
        (re.compile(r"System memory detected.*?([\d.]+) GB"),
         "ram", _CYAN, 0),
        (re.compile(r"fetch_spec \S+: memory budget ([\d.]+) GB.*strip_height=([^,]+),.*max_extract_years=(\d+)"),
         "cfg", _CYAN, 0),

        # ── Phase A (fetch patches) ──────────────────────────────────────────
        (re.compile(r"fetch_spec (\S+): (\d+) years.*Phase A.*Phase B \((\d+) concurrent\)"),
         "fetch", _BLUE, 0),
        (re.compile(r"fetch_spec (\S+) year (\d+): Phase A.*fetching (\d+) strips \((\d+) concurrent"),
         "fetch", _BLUE, 0),
        (re.compile(r"Fetching (\d+)/(\d+) items not yet in cache"),
         "fetch", _BLUE, 2),
        (re.compile(r"All \d+ items already in cache"),
         "cache", _GREEN, 2),
        (re.compile(r"(\d+) items? (found|loaded from cache)"),
         "stac", _CYAN, 2),
        (re.compile(r"STAC search: (\S+) → (\S+)  cloud < (\d+)%"),
         "stac", _CYAN, 2),
        (re.compile(r"All \d+ shards already complete — skipping fetch"),
         "cache", _GREEN, 2),
        (re.compile(r"collect: fetch-only phase complete"),
         "done", _GREEN, 2),

        # ── Shard lifecycle ──────────────────────────────────────────────────
        (re.compile(r"Shard (\d+)/(\d+): (\d+) points"),
         "shard", _MAG, 0),
        (re.compile(r"shard (\d+)/(\d+) complete: ([\d,]+) rows"),
         "shard", _GREEN, 0),
        (re.compile(r"Shard (\d+)/(\d+) already complete, skipping"),
         "cache", _GREEN, 2),
        (re.compile(r"sorting shard (\d+)/(\d+) →"),
         "sort", _CYAN, 2),
        (re.compile(r"shard (\d+)/(\d+) sorted"),
         "sort", _GREEN, 2),
        # Progress ticks — shown as plain text in non-TTY (TTY renders via _progress)
        (re.compile(r"S[12] scenes\s+shard (\d+)/(\d+)\s+item (\d+)/(\d+)\s+[\d,]+ rows"),
         "item", _CYAN, 2),
        (re.compile(r"concat (\d+)/(\d+) row groups"),
         "merge", _GREEN, 2),

        # ── Tile counter ─────────────────────────────────────────────────────
        (re.compile(r"fetch_spec \S+ year \d+: tile (\d+)/(\d+) — (\S+)"),
         "tile", _MAG, 0),

        # ── Per-tile pipeline (proxy/_pipeline.py) ───────────────────────────
        (re.compile(r"\[(?:v2 tile )?(\S+) (\d+)\] (?:\[chunk \d+_\d+\] )?STAC search"),
         "stac", _CYAN, 0),
        (re.compile(r"\[(?:v2 tile )?(\S+) (\d+)\] (?:\[chunk \d+_\d+\] )?(\d+) STAC items"),
         "stac", _CYAN, 0),
        (re.compile(r"\[(?:v2 tile )?(\S+) (\d+)\] (\d+) chunks \((\d+)x(\d+) px each\)"),
         "chunks", _MAG, 0),
        (re.compile(r"\[(?:v2 tile )?(\S+) (\d+)\] \[chunk \d+_\d+\] skipping"),
         "cache", _GREEN, 0),
        (re.compile(r"\[(?:v2 tile )?(\S+) (\d+)\] \[chunk \d+_\d+\] fetch_tiffs"),
         "fetch", _BLUE, 0),
        (re.compile(r"\[(?:v2 tile )?(\S+) (\d+)\] \[chunk \d+_\d+\] extract_scenes"),
         "extract", _MAG, 0),
        (re.compile(r"\[(?:v2 tile )?(\S+) (\d+)\] \[chunk \d+_\d+\] no scene data"),
         "empty", _DIM, 0),
        (re.compile(r"\[(?:v2 tile )?(\S+) (\d+)\] \[chunk \d+_\d+\] ready →"),
         "chunk", _GREEN, 0),

        # ── Per-tile pipeline (tile_pipeline.py) ─────────────────────────────
        (re.compile(r"\[(\S+) (\d+)\] already done"),
         "cache", _GREEN, 0),
        (re.compile(r"\[(\S+) (\d+)\] resuming from chunk"),
         "resume", _CYAN, 0),
        (re.compile(r"\[(\S+) (\d+)\] chunk \(\d+,\d+\) written"),
         "chunk", _GREEN, 0),

        # ── Phase B (extract) ────────────────────────────────────────────────
        (re.compile(r"fetch_spec \S+: Phase A complete"),
         "done", _GREEN, 0),
        (re.compile(r"fetch_spec \S+ year \d+: Phase A complete"),
         "done", _GREEN, 0),
        (re.compile(r"fetch_spec (\S+) year (\d+): patches ready, queuing extract"),
         "extract", _MAG, 0),
        (re.compile(r"fetch_spec (\S+) year (\d+): Phase B.*extracting (\d+) strips \((\d+) concurrent\)"),
         "extract", _MAG, 0),
        (re.compile(r"Extraction workers: (\d+) \(pixel count ([\d,]+), target ([\d.]+) GB\)"),
         "workers", _CYAN, 2),
        (re.compile(r"Pixel grid: (\d+) × (\d+) = ([\d,]+) points"),
         "grid", _DIM, 2),
        (re.compile(r"Concatenating \d+ sorted shards"),
         "merge", _CYAN, 0),

        # ── S1 ───────────────────────────────────────────────────────────────
        (re.compile(r"S1 .* written"),
         "s1", _CYAN, 2),
        (re.compile(r"sorting \d+ shards"),
         "s1", _CYAN, 0),
        (re.compile(r"merging \d+ sorted shards"),
         "s1", _CYAN, 0),
        (re.compile(r"(?:sort|merge) done →"),
         "s1", _GREEN, 0),
        (re.compile(r"wrote \S+\.s1\.parquet"),
         "s1", _GREEN, 0),

        # ── Tile merge (DuckDB sort-merge) ───────────────────────────────────
        (re.compile(r"merge_tile: sort-merging"),
         "merge", _CYAN, 0),
        (re.compile(r"merge_tile: wrote"),
         "done", _GREEN, 0),
        (re.compile(r"merge_tile: .* already up-to-date"),
         "cache", _GREEN, 0),

        # ── Strip merge / written ────────────────────────────────────────────
        (re.compile(r"fetch_spec (\S+) year (\d+): merging (\d+) strips → (\S+)"),
         "merge", _GREEN, 0),
        (re.compile(r"Written: \S+\.s2\.parquet\s+\([\d,]+ rows\)"),
         "done", _GREEN, 2),
        (re.compile(r"fetch_spec (\S+) year (\d+): written (\d+) tile"),
         "done", _GREEN, 0),
    ]

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color
        self._start = time.monotonic()

    def _c(self, text: str, code: str) -> str:
        if self._use_color:
            return f"{code}{text}{self._RESET}"
        return text

    def _elapsed(self) -> str:
        secs = time.monotonic() - self._start
        return f"{secs:5.0f}s" if secs < 3600 else f"{secs/3600:5.1f}h"

    # Matches [v2 tile TILE YEAR] [chunk ROW_COL] or [TILE YEAR] [chunk ROW_COL]
    _PAT_TILE_CTX  = re.compile(r"^\[(?:v2 tile )?(\w+) (\d{4})\](?: \[chunk (\d+_\d+)\])?")

    def _render(self, label: str, colour: str, msg: str, indent: int) -> str:
        elapsed = self._c(self._elapsed(), self._DIM)

        # Extract tile/year/chunk context from the message and strip it from body.
        m = self._PAT_TILE_CTX.match(msg)
        if m:
            tile, year, chunk = m.group(1), m.group(2), m.group(3)
            ctx_str = f"{tile} {year}" + (f" chunk {chunk}" if chunk else "")
            ctx     = self._c(f"[{ctx_str:<22}]", self._DIM)
            body    = msg[m.end():].lstrip()
        else:
            ctx  = self._c(f"{'':24}", self._DIM)
            body = msg

        prefix = self._c(f"[{label:>7}]", colour + self._BOLD)
        return f"  {elapsed}  {ctx}  {prefix}  {body}"

    def render_progress(self, snap: dict) -> str:
        """Render one consolidated \r progress line from a _ProgressState snapshot."""
        slots = snap["slots"]

        elapsed = self._c(self._elapsed(), self._DIM)

        if not slots:
            return ""

        item_slots   = [s for s in slots.values() if s["kind"] == "item"]
        concat_slots = [s for s in slots.values() if s["kind"] == "concat"]
        chip_slots   = [s for s in slots.values() if s["kind"] == "chips"]

        if concat_slots:
            label    = concat_slots[0]["label"]
            rg_done  = sum(s["rg_i"] for s in concat_slots)
            rg_total = sum(s["rg_n"] for s in concat_slots)
            rows     = sum(s["rows"] for s in concat_slots)
            bar      = _bar(rg_done, rg_total)
            pct      = f"{100 * rg_done / rg_total:.0f}%" if rg_total else "?"
            prefix   = self._c("[ merge]", self._GREEN + self._BOLD)
            bar_col  = self._c(bar, self._GREEN)
            label_col = self._c(label, self._GREEN + self._BOLD)
            rows_str  = self._c(f"{rows:,} rows", self._DIM)
            rg_str    = self._c(f"{rg_done}/{rg_total} rg", self._RESET)
            return f"  {elapsed}  {prefix}  {label_col}  {bar_col} {pct:>4}  {rg_str}  {rows_str}"

        if chip_slots:
            label     = chip_slots[0]["label"]
            done      = sum(s["done"]  for s in chip_slots)
            total     = sum(s["total"] for s in chip_slots)
            bar       = _bar(done, total)
            pct       = f"{100 * done / total:.0f}%" if total else "?"
            prefix    = self._c("[ fetch]", self._BLUE + self._BOLD)
            bar_col   = self._c(bar, self._BLUE)
            label_col = self._c(label, self._BLUE + self._BOLD)
            patch_str = self._c(f"{done}/{total} patches", self._RESET)
            return f"  {elapsed}  {prefix}  {label_col}  {bar_col} {pct:>4}  {patch_str}"

        # Item-extraction phase: aggregate items and workers across all active shards
        label      = item_slots[0]["label"] if item_slots else "S2 scenes"
        item_done  = sum(s["item_i"] for s in item_slots)
        item_total = sum(s["item_n"] for s in item_slots)
        rows       = sum(s["rows"]   for s in item_slots)

        bar      = _bar(item_done, item_total)
        pct      = f"{100 * item_done / item_total:.0f}%" if item_total else "?"
        prefix   = self._c("[ shard]", self._MAG + self._BOLD)
        bar_col  = self._c(bar, self._CYAN)
        label_col = self._c(label, self._MAG + self._BOLD)
        rows_str  = self._c(f"{rows:,} rows", self._DIM)
        items_str = self._c(f"{item_done}/{item_total} items", self._RESET)

        w_active = sum(s["workers_active"] for s in item_slots if s.get("workers_active") is not None)
        w_max    = sum(s["workers_max"]    for s in item_slots if s.get("workers_max")    is not None)
        if w_max > 0:
            worker_dots = "●" * min(w_active, w_max) + "○" * max(0, w_max - w_active)
            workers_col = self._c(f"[{worker_dots}]", self._CYAN)
            return (f"  {elapsed}  {prefix}  {label_col}  {bar_col} {pct:>4}  {items_str}"
                    f"  {workers_col}  {rows_str}")
        return (f"  {elapsed}  {prefix}  {label_col}  {bar_col} {pct:>4}  {items_str}"
                f"  {rows_str}")

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()

        if record.levelno >= logging.ERROR:
            elapsed = self._c(self._elapsed(), self._DIM)
            prefix  = self._c("[  error]", self._RED + self._BOLD)
            body    = self._c(msg, self._RED)
            line    = f"  {elapsed}  {prefix}  {body}"
            if record.exc_info:
                import traceback
                tb = "".join(traceback.format_exception(*record.exc_info)).rstrip()
                dim_tb = "\n".join(self._c(f"           {l}", self._DIM) for l in tb.splitlines())
                line = f"{line}\n{dim_tb}"
            return line

        if record.levelno == logging.WARNING:
            elapsed = self._c(self._elapsed(), self._DIM)
            prefix  = self._c("[   warn]", self._YELLOW + self._BOLD)
            body    = self._c(msg, self._YELLOW)
            return f"  {elapsed}  {prefix}  {body}"

        for pat, label, colour, indent in self._PATTERNS:
            if pat.search(msg):
                return self._render(label, colour, msg, indent)

        # Unmatched — dim pass-through
        elapsed = self._c(self._elapsed(), self._DIM)
        prefix  = self._c("[   info]", self._DIM)
        m = self._PAT_TILE_CTX.match(msg)
        if m:
            tile, year, chunk = m.group(1), m.group(2), m.group(3)
            ctx_str = f"{tile} {year}" + (f" chunk {chunk}" if chunk else "")
            ctx  = self._c(f"[{ctx_str:<22}]", self._DIM)
            body = msg[m.end():].lstrip()
        else:
            ctx  = self._c(f"{'':24}", self._DIM)
            body = msg
        return f"  {elapsed}  {ctx}  {prefix}  {self._c(body, self._DIM)}"


class _FetchHandler(logging.StreamHandler):
    """StreamHandler that renders aggregated progress as a single \r line.

    Progress messages (item extraction ticks, concat ticks) update the shared
    _ProgressState and trigger a re-render of the single consolidated line.
    This means any number of concurrent shards/strips/years all contribute to
    one line — no interleaving, no flickering.

    Non-progress messages always get a \n, preceded by a closing \n if the
    last write was a progress line.
    """

    def __init__(self, stream, use_color: bool) -> None:
        super().__init__(stream)
        self._use_color = use_color
        self._on_progress_line = False
        self._lock = _threading.Lock()

    def _write_progress(self, line: str) -> None:
        if not line:
            return
        try:
            import shutil
            cols = shutil.get_terminal_size().columns
        except Exception:
            cols = 120
        visible_len = len(_strip_ansi(line))
        padded = line + " " * max(0, cols - visible_len - 1)
        self.stream.write(f"\r{padded}")
        self.stream.flush()
        self._on_progress_line = True

    # Per-shard S1 lifecycle lines that are redundant when the progress bar is
    # showing — suppressed on TTY only (non-TTY logs them for CI/file output).
    _TTY_SUPPRESS = re.compile(
        r"S1: shard \d+/\d+ — (?:\d+ points|fetching patches)"
        r"|fetch_patches: phase [12] —"
        r"|fetch_patches complete:"
        r"|S1: shard \d+/\d+ complete —"
    )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            fmt: _FetchFormatter = self.formatter  # type: ignore[assignment]

            if self._use_color and self._TTY_SUPPRESS.search(msg):
                # Retire completed shard slots even when suppressing the line.
                m_done = re.search(r"S1: shard (\d+)/\d+ complete", msg)
                if m_done:
                    with self._lock:
                        _progress.complete_unit(int(m_done.group(1)))
                if re.search(r"fetch_patches complete:", msg):
                    with self._lock:
                        _progress.clear_chips()
                return

            m_item   = _PAT_ITEM_PROGRESS.search(msg)   if self._use_color else None
            m_chips  = _PAT_CHIP_PROGRESS.search(msg)   if self._use_color else None
            m_concat = _PAT_CONCAT_PROGRESS.search(msg) if self._use_color else None
            m_total  = re.search(r"S1: (\d+) shards total", msg) if self._use_color else None

            with self._lock:
                if m_total:
                    _progress.set_total(int(m_total.group(1)))
                    # fall through — let the line render normally

                if m_item:
                    label   = m_item.group(1)
                    shard_i = int(m_item.group(2))
                    shard_n = int(m_item.group(3))
                    item_i  = int(m_item.group(4))
                    item_n  = int(m_item.group(5))
                    rows    = int(m_item.group(6))
                    wa = int(m_item.group(7)) if m_item.group(7) else None
                    wm = int(m_item.group(8)) if m_item.group(8) else None
                    _progress.update_item(label, shard_i, shard_n, item_i, item_n, rows, wa, wm)
                    self._write_progress(fmt.render_progress(_progress.snapshot()))
                    return

                if m_chips:
                    label = m_chips.group(1)
                    done  = int(m_chips.group(2))
                    total = int(m_chips.group(3))
                    _progress.update_chips(label, done, total)
                    self._write_progress(fmt.render_progress(_progress.snapshot()))
                    return

                if m_concat:
                    label = m_concat.group(1)
                    rg_i  = int(m_concat.group(2))
                    rg_n  = int(m_concat.group(3))
                    rows  = int(m_concat.group(5).replace(",", ""))
                    _progress.update_concat(label, rg_i, rg_n, rows)
                    self._write_progress(fmt.render_progress(_progress.snapshot()))
                    return

                # Normal line — close progress line first, then check for
                # shard-complete so we can retire the slot from _progress.
                if self._on_progress_line:
                    self.stream.write("\n")
                    self._on_progress_line = False

                # Retire completed slots so stale progress doesn't bleed into next phase.
                m_done = re.search(r"shard (\d+)/\d+ complete:", msg)
                if m_done:
                    _progress.complete_unit(int(m_done.group(1)))
                if re.search(r"fetch_patches complete:", msg):
                    _progress.clear_chips()
                if re.search(r"Written: \S+\.s2\.parquet", msg):
                    _progress.clear_concat()
                if re.search(r"Phase A complete", msg):
                    _progress.reset()

                self.stream.write(fmt.format(record) + "\n")
                self.stream.flush()

        except Exception:
            self.handleError(record)


def _setup_fetch_logging() -> None:
    """Configure beautiful progress logging for the fetch pipeline."""
    use_color = sys.stderr.isatty()
    _progress.reset()

    handler = _FetchHandler(sys.stderr, use_color=use_color)
    handler.setFormatter(_FetchFormatter(use_color=use_color))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    for name in ("rasterio", "urllib3", "requests", "botocore", "boto3",
                 "s3transfer", "asyncio", "fiona", "pyproj"):
        logging.getLogger(name).setLevel(logging.WARNING)


from utils.location import all_locations, get  # noqa: E402


def _fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1e12:
        return f"{n_bytes/1e12:.1f} TB"
    if n_bytes >= 1e9:
        return f"{n_bytes/1e9:.1f} GB"
    if n_bytes >= 1e6:
        return f"{n_bytes/1e6:.0f} MB"
    return f"{n_bytes/1e3:.0f} KB"


def _dir_size(p: "Path") -> int:
    import os
    total = 0
    for entry in os.scandir(p):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
        elif entry.is_dir(follow_symlinks=False):
            total += _dir_size(Path(entry.path))
    return total


def cmd_list(args: argparse.Namespace) -> None:
    locs = sorted(all_locations(), key=lambda l: l.id)

    rows = []
    for loc in locs:
        chips = loc.chips_path()
        years = loc.parquet_years()
        if years:
            years_str = f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])
            tile_paths = loc.parquet_tile_paths()
            total_bytes = sum(p.stat().st_size for ps in tile_paths.values() for p in ps)
            parquet_str = _fmt_size(total_bytes)
        else:
            years_str   = "—"
            parquet_str = "—"
        chips_str  = _fmt_size(_dir_size(chips)) if chips.exists() else "—"
        area_str   = f"{loc.area_km2:.1f}"
        pixels_str = f"{loc.pixel_count:,}"
        rows.append((loc.id, years_str, area_str, pixels_str, chips_str, parquet_str))

    headers = ("ID", "YEARS", "AREA km²", "PIXELS", "CHIPS", "PARQUET")
    # columns 0,1 are left-aligned; 2-5 are right-aligned
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cols):
        id_col, yr, area, px, chips, parq = cols
        w = widths
        return (f"  {id_col:<{w[0]}} {yr:<{w[1]}} {area:>{w[2]}}  {px:>{w[3]}}  "
                f"{chips:>{w[4]}}  {parq:>{w[5]}}")

    print(fmt_row(headers))
    print("  " + "-" * (sum(widths) + 12))
    for row in rows:
        print(fmt_row(row))


def cmd_info(args: argparse.Namespace) -> None:
    try:
        loc = get(args.id)
    except KeyError:
        print(f"Unknown location: {args.id!r}", file=sys.stderr)
        sys.exit(1)

    print(loc.summary())

    years = loc.parquet_years()
    if not years:
        return

    import polars as pl

    print()
    print(f"  Fetched years: {', '.join(str(y) for y in years)}")

    tile_paths_by_year = loc.parquet_tile_paths()
    for year in years:
        tile_paths = tile_paths_by_year.get(year, [])
        total_size = sum(p.stat().st_size for p in tile_paths)
        size_str = _fmt_size(total_size)
        dfs = [pl.read_parquet(p, columns=["date"]) for p in tile_paths]
        df = pl.concat(dfs) if dfs else pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})
        counts = (
            df.with_columns(pl.col("date").cast(pl.Date).dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(pl.col("date").n_unique().alias("n"))
            .sort("month")
        )
        print()
        print(f"  {year}  ({size_str})  — scene count per month")
        for row in counts.iter_rows(named=True):
            label = row["month"].strftime("%b %Y").upper()
            n = row["n"]
            bar = "#" * n
            print(f"    {label:<12} {n:>4}  {bar}")


def cmd_bbox(args: argparse.Namespace) -> None:
    try:
        print(get(args.id).bbox_cli)
    except KeyError:
        print(f"Unknown location: {args.id!r}", file=sys.stderr)
        sys.exit(1)


def cmd_fetch(args: argparse.Namespace) -> None:
    _setup_fetch_logging()

    try:
        loc = get(args.id)
    except KeyError:
        print(f"Unknown location: {args.id!r}", file=sys.stderr)
        sys.exit(1)

    use_color = sys.stderr.isatty()
    _DIM   = "\033[2m"  if use_color else ""
    _BOLD  = "\033[1m"  if use_color else ""
    _CYAN  = "\033[36m" if use_color else ""
    _GREEN = "\033[32m" if use_color else ""
    _RESET = "\033[0m"  if use_color else ""

    years_str = " ".join(str(y) for y in sorted(args.years))
    tiles_str = " ".join(loc.tile_ids())
    print(file=sys.stderr)
    print(f"  {_BOLD}{_CYAN}Fetch  {_RESET}{_BOLD}{loc.name}{_RESET}{_DIM}  ({loc.id}){_RESET}", file=sys.stderr)
    print(f"  {_DIM}{'─' * 60}{_RESET}", file=sys.stderr)
    print(f"  {_DIM}years   {_RESET}{years_str}", file=sys.stderr)
    print(f"  {_DIM}tiles   {_RESET}{tiles_str}", file=sys.stderr)
    print(f"  {_DIM}area    {_RESET}{loc.area_km2:.1f} km²  ·  ~{loc.pixel_count:,} pixels  ·  cloud ≤ {args.cloud_max}%", file=sys.stderr)
    print(f"  {_DIM}nbar    {_RESET}{'on' if not args.no_nbar else 'off'}", file=sys.stderr)
    print(f"  {_DIM}{'─' * 60}{_RESET}", file=sys.stderr)
    print(file=sys.stderr)

    output_dir = Path(args.output_dir) if args.output_dir else None
    work_dir   = Path(args.work_dir) if getattr(args, "work_dir", None) else None

    t0 = time.monotonic()
    try:
        written = loc.fetch(
            years=args.years,
            cloud_max=args.cloud_max,
            apply_nbar=not args.no_nbar,
            n_workers=args.workers,
            tiles=getattr(args, "tiles", None),
            output_dir=output_dir,
            work_dir=work_dir,
        )
    except KeyboardInterrupt:
        print(file=sys.stderr)
        print(f"\n  {_DIM}Interrupted.{_RESET}", file=sys.stderr)
        import os as _os
        _os._exit(1)
    elapsed = time.monotonic() - t0

    print(file=sys.stderr)
    print(f"  {_DIM}{'─' * 60}{_RESET}", file=sys.stderr)
    if written:
        total_mb = sum(p.stat().st_size for p in written if p.exists()) / 1e6
        print(f"  {_GREEN}{_BOLD}Done{_RESET}  {len(written)} tile(s) written  ·  {total_mb:.0f} MB  ·  {elapsed:.0f}s elapsed", file=sys.stderr)
        for path in written:
            print(f"Written: {path}")
    else:
        print(f"  {_DIM}No output files written.{_RESET}", file=sys.stderr)
    print(file=sys.stderr)


def cmd_training_list(args: argparse.Namespace) -> None:
    from utils.regions import load_regions
    from utils.location import _bbox_pixel_count

    yaml_path = Path(args.yaml) if args.yaml else None
    regions = load_regions(yaml_path) if yaml_path else load_regions()
    totals: dict[str, int] = {}

    print(f"  {'ID':<40} {'LABEL':<10} {'YEARS':<12} {'PIXELS':>8}")
    print("  " + "-" * 74)
    for r in regions:
        n = _bbox_pixel_count(r.bbox)
        totals[r.label] = totals.get(r.label, 0) + n
        year_str = f"{min(r.years)}–{max(r.years)}" if r.years else "—"
        print(f"  {r.id:<40} {r.label:<10} {year_str:<12} {n:>8,}")

    print("  " + "-" * 68)
    for label, total in sorted(totals.items()):
        print(f"  {'Total ' + label:<40} {'':10} {'':6} {total:>8,}")


def cmd_training_fetch(args: argparse.Namespace) -> None:
    _setup_fetch_logging()
    from utils.regions import load_regions, select_regions
    from utils.training_collector import ensure_training_pixels

    yaml_path = Path(args.yaml) if args.yaml else None
    if args.all:
        regions = load_regions(yaml_path) if yaml_path else load_regions()
    else:
        regions = select_regions(args.regions, yaml_path) if yaml_path else select_regions(args.regions)
    ensure_training_pixels(
        regions=regions,
        cloud_max=args.cloud_max,
        apply_nbar=not args.no_nbar,
        max_concurrent=args.max_concurrent,
        max_region_workers=args.max_region_workers,
    )


def cmd_training_verify(args: argparse.Namespace) -> None:
    import polars as pl
    from utils.regions import load_regions
    from utils.training_collector import _region_parquet_path, tile_parquet_path, _load_index

    BAND_COLS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    yaml_path = Path(args.yaml) if args.yaml else None
    regions = load_regions(yaml_path) if yaml_path else load_regions()
    if args.prefix:
        regions = [r for r in regions if r.id.startswith(args.prefix)]
        if not regions:
            print(f"  No regions match prefix {args.prefix!r}.")
            return
    issues: list[str] = []
    rows = []

    for r in regions:
        path = _region_parquet_path(r.id)
        if not path.exists():
            issues.append(f"MISSING  {r.id} — parquet not found at {path}")
            continue

        df = pl.read_parquet(path)
        s2 = df.filter(pl.col("source") == "S2") if "source" in df.columns else df

        n_pixels = s2["point_id"].n_unique()
        n_obs    = len(s2)
        dupes    = len(s2) - len(s2.unique(subset=["point_id", "date"]))

        # Observations per pixel
        obs_per_pixel = s2.group_by("point_id").agg(pl.len().alias("n"))["n"]
        obs_min = int(obs_per_pixel.min())
        obs_med = int(obs_per_pixel.median())
        obs_max = int(obs_per_pixel.max())

        # Band stats — mean and std across S2 obs only
        band_means = [s2[c].mean() for c in BAND_COLS]
        band_stds  = [s2[c].std()  for c in BAND_COLS]
        nan_counts = sum(s2[c].is_null().sum() for c in BAND_COLS)
        mean_of_means = sum(band_means) / len(band_means)
        mean_of_stds  = sum(band_stds)  / len(band_stds)

        # Flag suspicious values
        region_issues = []
        if dupes > 0:
            region_issues.append(f"DUPES    {r.id} — {dupes} duplicate (point_id, date) rows")
        if n_pixels < 5:
            region_issues.append(f"SPARSE   {r.id} — only {n_pixels} pixels")
        if obs_min < 4:
            region_issues.append(f"LOW_OBS  {r.id} — some pixels have <4 observations (min={obs_min})")
        if nan_counts > 0:
            region_issues.append(f"NAN      {r.id} — {nan_counts} NaN band values")
        bm_min, bm_max = min(band_means), max(band_means)
        if bm_max > 1.5 or bm_min < -0.5:
            region_issues.append(f"RANGE    {r.id} — band means outside expected range [{bm_min:.2f}, {bm_max:.2f}]")

        # S1 backscatter sanity checks — VH lives in the tile parquet, not the region parquet
        index = _load_index()
        tile_ids = index.filter(pl.col("region_id") == r.id)["tile_id"].to_list()
        for tile_id in tile_ids:
            tp = tile_parquet_path(tile_id)
            if not tp.exists():
                region_issues.append(f"S1_MISS  {r.id} — tile parquet {tile_id} not built yet")
                continue
            tile_df = pl.read_parquet(tp, columns=["point_id", "vh"])
            region_prefix = r.id + "_"
            vh = tile_df.filter(pl.col("point_id").str.starts_with(region_prefix))["vh"].drop_nulls()
            if len(vh) == 0:
                region_issues.append(f"S1_MISS  {r.id} — no S1 vh data in tile {tile_id}")
            elif vh.median() > 1.0:
                region_issues.append(
                    f"S1_SCALE {r.id} — vh median={vh.median():.1f} in tile {tile_id} looks like raw GRD DN, not linear power"
                )

        if not region_issues and args.prefix:
            issues.append(f"OK       {r.id}")
        else:
            issues.extend(region_issues)

        rows.append((r.id, r.label, n_pixels, n_obs, obs_min, obs_med, obs_max,
                     f"{mean_of_means:.3f}", f"{mean_of_stds:.3f}"))

    # Print table
    headers = ("REGION", "LABEL", "PIXELS", "OBS", "MIN_OBS", "MED_OBS", "MAX_OBS", "MEAN_BAND", "STD_BAND")
    widths  = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]

    def fmt(cols):
        return (f"  {str(cols[0]):<{widths[0]}}  {str(cols[1]):<{widths[1]}}"
                f"  {str(cols[2]):>{widths[2]}}  {str(cols[3]):>{widths[3]}}"
                f"  {str(cols[4]):>{widths[4]}}  {str(cols[5]):>{widths[5]}}"
                f"  {str(cols[6]):>{widths[6]}}  {str(cols[7]):>{widths[7]}}"
                f"  {str(cols[8]):>{widths[8]}}")

    print(fmt(headers))
    print("  " + "-" * (sum(widths) + 18))
    for row in rows:
        print(fmt(row))

    # Summary
    if rows:
        total_pixels = sum(r[2] for r in rows)
        presence_px  = sum(r[2] for r in rows if r[1] == "presence")
        absence_px   = sum(r[2] for r in rows if r[1] == "absence")
        print()
        print(f"  Total pixels: {total_pixels:,}  (presence: {presence_px:,}  absence: {absence_px:,}  ratio: 1:{absence_px//max(presence_px,1)})")

    if issues:
        print()
        print(f"  {len(issues)} issue(s) found:")
        for iss in issues:
            marker = " " if iss.startswith("OK") else "!"
            print(f"    {marker} {iss}")
    else:
        print()
        print("  No issues found.")


def _print_validate_table(reports: list, verbose: bool) -> None:
    from utils.parquet_validator import Status

    use_color = sys.stdout.isatty()
    _COLOR = {Status.PASS: "\033[32m", Status.WARN: "\033[33m", Status.FAIL: "\033[31m"}
    _RESET = "\033[0m"

    def colored(text: str, status: "Status") -> str:
        if use_color:
            return f"{_COLOR[status]}{text}{_RESET}"
        return text

    header = f"  {'TILE':<12} {'YEAR':<6} {'ROWS':>12} {'PIXELS':>9} {'DATES':>6} {'S1':<5} STATUS"
    print(header)
    print("  " + "-" * (len(header) - 2))

    issue_lines: list[str] = []
    for r in reports:
        rows_str   = f"{r.n_rows:>12,}" if r.n_rows else f"{'—':>12}"
        pixels_str = f"{r.n_pixels:>9,}" if r.n_pixels else f"{'—':>9}"
        dates_str  = f"{r.n_dates:>6}" if r.n_dates else f"{'—':>6}"
        if r.path is None or not r.path.exists():
            s1_str = f"{'—':<5}"
        elif r.s1_old_format:
            s1_str = colored(f"{'OLD':<5}", Status.FAIL)
        elif r.has_s1:
            s1_str = f"{'YES':<5}"
        else:
            s1_str = colored(f"{'NO':<5}", Status.WARN)

        status_str = colored(r.status.value, r.status)
        issue_names = ", ".join(i.name for i in r.issues)
        print(f"  {r.tile_id:<12} {r.year:<6} {rows_str} {pixels_str} {dates_str} {s1_str} {status_str}  {issue_names}")

        for i in r.issues:
            issue_lines.append(f"    {r.tile_id} ({r.year})  {colored(i.status.value, i.status)}  {i.name}: {i.message}")

    if verbose and issue_lines:
        print()
        print("  Issues:")
        for line in issue_lines:
            print(line)


def cmd_validate(args: argparse.Namespace) -> None:
    from utils.parquet_validator import validate_location, Status

    try:
        loc = get(args.id)
    except KeyError:
        print(f"Unknown location: {args.id!r}", file=sys.stderr)
        sys.exit(1)

    years = args.years if args.years else None
    reports = validate_location(loc, years=years)

    if not reports:
        print(f"  No parquet data found for {args.id!r}.")
        sys.exit(0)

    _print_validate_table(reports, verbose=args.verbose)

    any_fail = any(r.status == Status.FAIL for r in reports)
    sys.exit(1 if any_fail else 0)


def cmd_training(args: argparse.Namespace) -> None:
    {
        "list":   cmd_training_list,
        "fetch":  cmd_training_fetch,
        "verify": cmd_training_verify,
    }[args.training_cmd](args)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python cli/location.py",
        description="Inspect and fetch Parkinsonia analysis locations.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all known locations")

    pi = sub.add_parser("info", help="Print full summary for a location")
    pi.add_argument("id", help="Location id (e.g. longreach, muttaburra)")

    pb = sub.add_parser("bbox", help="Print bbox as 'lon_min,lat_min,lon_max,lat_max'")
    pb.add_argument("id", help="Location id")

    pf = sub.add_parser("fetch", help="Fetch Sentinel-2 pixel observations")
    pf.add_argument("id", help="Location id")
    pf.add_argument("--years", nargs="+", type=int, required=True, metavar="YYYY",
                    help="Calendar years to fetch (e.g. --years 2020 2021 2022)")
    pf.add_argument("--cloud-max", type=int, default=30, metavar="N",
                    help="Max cloud cover %% (default: 30)")
    pf.add_argument("--no-nbar", action="store_true",
                    help="Disable BRDF NBAR c-factor correction")
    pf.add_argument("--workers", type=int, default=None, metavar="N",
                    help="Concurrent item extraction workers (default: auto-scaled by pixel count)")
    pf.add_argument("--tiles", nargs="+", metavar="TILE_ID", default=None,
                    help="Only fetch these MGRS tile IDs (default: all tiles for the location)")
    pf.add_argument("--output-dir", type=str,
                    default=os.environ.get("CHUNKSTORE_DIR", "/mnt/external/chunkstore"),
                    metavar="DIR",
                    help="Root directory for final chunk parquets "
                         "(default: $CHUNKSTORE_DIR or /mnt/external/chunkstore). "
                         "Chunks are written to <DIR>/<location_id>/<year>/<tile_id>/.")
    pf.add_argument("--work-dir", type=str, default=None, metavar="DIR",
                    help="Root directory for temporary working data "
                         "(default: same as --output-dir). "
                         "Should be on fast local NVMe — intermediate files are deleted after each chunk.")

    pv = sub.add_parser("validate", help="Validate parquet data quality for a location")
    pv.add_argument("id", help="Location id (e.g. longreach, flinders0)")
    pv.add_argument("--year", dest="years", nargs="+", type=int, metavar="YYYY",
                    help="Year(s) to validate; default: all fetched years")
    pv.add_argument("--verbose", action="store_true",
                    help="Print full issue details below the summary table")

    pt = sub.add_parser("training", help="Manage training regions and pixel collection")
    tsub = pt.add_subparsers(dest="training_cmd", required=True)

    tls = tsub.add_parser("list", help="List all training regions with estimated pixel counts")
    tls.add_argument("--yaml", metavar="PATH", default=None,
                     help="Path to regions YAML (default: data/locations/training.yaml)")

    tvfy = tsub.add_parser("verify", help="Verify training data quality and flag issues")
    tvfy.add_argument("--yaml", metavar="PATH", default=None,
                      help="Path to regions YAML (default: data/locations/training.yaml)")
    tvfy.add_argument("--prefix", metavar="STR",
                      help="Only verify regions whose id starts with this prefix")

    tf = tsub.add_parser("fetch", help="Fetch pixels for training regions")
    tf.add_argument("--yaml", metavar="PATH", default=None,
                    help="Path to regions YAML (default: data/locations/training.yaml)")
    grp = tf.add_mutually_exclusive_group(required=True)
    grp.add_argument("--regions", nargs="+", metavar="ID",
                     help="Region IDs to fetch")
    grp.add_argument("--all", action="store_true",
                     help="Fetch all regions in the YAML")
    tf.add_argument("--cloud-max", type=int, default=80, metavar="N")
    tf.add_argument("--no-nbar", action="store_true")
    tf.add_argument("--max-concurrent", type=int, default=32, metavar="N",
                    help="Max concurrent HTTP patch fetches per tile (default: 32)")
    tf.add_argument("--max-region-workers", type=int, default=4, metavar="N",
                    help="Max regions fetched in parallel (default: 4)")

    args = p.parse_args()
    {
        "list":     cmd_list,
        "info":     cmd_info,
        "bbox":     cmd_bbox,
        "fetch":    cmd_fetch,
        "validate": cmd_validate,
        "training": cmd_training,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
