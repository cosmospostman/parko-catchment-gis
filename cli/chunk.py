"""cli/chunk.py — Chunkstore introspection and integrity verification.

Usage
-----
  python cli/chunk.py verify [--year YYYY] [--tile TILE] [--root DIR] [--all]

Examples
--------
  python cli/chunk.py verify                       # all chunks under CHUNKSTORE_DIR
  python cli/chunk.py verify --year 2025           # one year
  python cli/chunk.py verify --tile 55KCB          # one tile, all years
  python cli/chunk.py verify --year 2025 --tile 55KCB

By default only chunks with issues are listed; pass --all to print every chunk.
Exit code is non-zero when any chunk has an issue (so it can gate a rebuild).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.chunk_verify import (  # noqa: E402
    MIN_S1_OBS_PER_YEAR,
    verify_chunkstore,
)

try:
    # config loads .env, so this honours CHUNKSTORE_DIR set there.
    from config import CHUNKSTORE_DIR as _DEFAULT_ROOT
except Exception:
    _DEFAULT_ROOT = Path(os.environ.get("CHUNKSTORE_DIR", "/mnt/external/chunkstore"))


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.exists():
        print(f"  Chunkstore root not found: {root}")
        return 2

    # Progress bar on stderr (auto-off when not a TTY, or with --no-progress) so
    # the results table and exit-code gating on stdout stay clean and pipeable.
    use_progress = args.progress and sys.stderr.isatty()
    bar = None

    def _on_start(total: int) -> None:
        nonlocal bar
        if use_progress and total:
            from tqdm import tqdm
            bar = tqdm(total=total, desc="Verifying chunks", unit="chunk",
                       file=sys.stderr, leave=False)

    _fails = [0]

    def _on_progress(done: int, rep) -> None:
        if bar is not None:
            if not rep.ok:
                _fails[0] += 1
                bar.set_postfix(fail=_fails[0])
            bar.update(1)

    reports = verify_chunkstore(
        root, year=args.year, tile=args.tile,
        on_start=_on_start, on_progress=_on_progress,
    )
    if bar is not None:
        bar.close()

    # Machine-readable mode: print one FAIL chunk parquet path per line to stdout
    # (nothing else), so it can feed a delete/refetch script.  Still exits non-zero
    # when failures exist.
    if args.fail_paths:
        fails = [r for r in reports if not r.ok]
        for r in fails:
            print(r.path)
        return 1 if fails else 0

    if not reports:
        scope = f"year={args.year} " if args.year else ""
        scope += f"tile={args.tile} " if args.tile else ""
        print(f"  No chunk parquets found under {root} {scope}".rstrip())
        return 0

    headers = ("YEAR", "TILE", "CHUNK", "PIXELS", "S1_DATES",
               "MED_FRAC", "MAX_FRAC", f"%>={MIN_S1_OBS_PER_YEAR}OBS", "STATUS")
    rows = []
    for r in reports:
        rows.append((
            str(r.year) if r.year is not None else "-",
            r.tile,
            f"r{r.chunk_row:02d}_c{r.chunk_col:02d}" if r.chunk_row >= 0 else "-",
            f"{r.n_pixels:,}",
            str(r.n_s1_dates),
            f"{r.s1_med_frac:.2f}",
            f"{r.s1_max_frac:.2f}",
            f"{r.pct_ge_min_obs:.1f}",
            "OK" if r.ok else "FAIL",
        ))

    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, h in enumerate(headers)]

    def fmt(cols):
        return "  " + "  ".join(
            f"{c:<{widths[i]}}" if i in (1, 2, 8) else f"{c:>{widths[i]}}"
            for i, c in enumerate(cols)
        )

    show = reports if args.all else [r for r in reports if not r.ok]
    if show:
        print(fmt(headers))
        print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
        show_ids = {id(r) for r in show}
        for r, row in zip(reports, rows):
            if id(r) in show_ids:
                print(fmt(row))

    issues = [iss for r in reports for iss in r.issues]
    notes = [n for r in reports for n in r.notes]
    n_fail = sum(1 for r in reports if not r.ok)
    print()
    print(f"  {len(reports)} chunk(s) checked — {len(reports) - n_fail} OK, "
          f"{n_fail} with issues, {len(notes)} note(s).")
    if issues:
        print()
        for iss in issues:
            print(f"    ! {iss}")
    if notes:
        print()
        for n in notes:
            print(f"    · {n}")
    return 1 if n_fail else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunkstore integrity tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="Verify chunk parquet integrity (S1 truncation, emptiness)")
    v.add_argument("--root", default=str(_DEFAULT_ROOT), help="Chunkstore root (default: $CHUNKSTORE_DIR)")
    v.add_argument("--year", type=int, default=None, help="Limit to one year")
    v.add_argument("--tile", default=None, help="Limit to one MGRS tile id")
    v.add_argument("--all", action="store_true", help="Print every chunk, not just failures")
    v.add_argument("--no-progress", dest="progress", action="store_false",
                   help="Disable the live progress bar (auto-off when stderr is not a TTY)")
    v.add_argument("--fail-paths", action="store_true",
                   help="Print only failing chunk parquet paths to stdout (one per line), "
                        "for piping into a delete/refetch script")
    v.set_defaults(progress=True)
    v.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
