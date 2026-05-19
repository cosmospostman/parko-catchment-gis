"""signals/eval_s1.py — Evaluate S1 SAR signals across priority Parkinsonia sites.

Site comparisons are defined in docs/S1-SIGNALS.md. Run with:
    python signals/eval_s1.py

Uses presence_min_vh_dry_db=-21.0 to match TAM training defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signals.eval import SiteSpec, evaluate
from signals.s1 import VHSignal, VVSignal, VHVVSignal, RVISignal


SITES = [
    SiteSpec("etna", [
        ("etna_presence_2",  "presence"),
        ("etna_presence_5",  "presence"),
        ("etna_presence_6",  "presence"),
        ("etna_absence_6",   "absence"),
        ("etna_absence_7",   "absence"),
        ("etna_absence_8",   "absence"),
    ]),
    SiteSpec("landsend", [
        ("landsend_sparse_presence_1", "presence"),
        ("landsend_sparse_presence_2", "presence"),
        ("landsend_absence_grass_1",   "absence"),
        ("landsend_absence_riverbed_1","absence"),
    ]),
    SiteSpec("frenchs", [
        ("frenchs_presence_1",               "presence"),
        ("frenchs_presence_2",               "presence"),
        ("frenchs_absence_riparian",         "absence"),
        ("frenchs_absence_riparian_woodland","absence"),
    ]),
    SiteSpec("burdekin", [
        ("burdekin_presence_1", "presence"),
        ("burdekin_absence_4",  "absence"),
        ("burdekin_absence_5",  "absence"),
        ("burdekin_absence_8",  "absence"),
    ]),
]

SIGNALS = [VHSignal(), VVSignal(), VHVVSignal(), RVISignal()]

PRESENCE_MIN_VH_DRY_DB = -21.0


def main() -> None:
    for signal in SIGNALS:
        print(f"\n{'='*70}")
        print(f"Signal: {signal.name}")
        print("="*70)
        evaluate(
            signal,
            SITES,
            rank_key="dry_mean",
            presence_min_vh_dry_db=PRESENCE_MIN_VH_DRY_DB,
            verbose=True,
        )


if __name__ == "__main__":
    main()
