"""pipelines/train.py — Training pipeline orchestrator.

Subcommands
-----------
load-testdata
    Fetch and stage a small set of fixture chips from the STAC archive for use
    by pytest. Writes a .fixture_commit sentinel to tests/fixtures/ so conftest
    can detect stale test data.

    This is a prerequisite for pytest, not part of the production run path.
    Analogy: manage.py in Django — administrative operations clearly separated
    from the main pipeline execution.

run (future session)
    Full training pipeline: Stage 0 fetch → extraction → quality → waveform →
    feature assembly → RF fit → spatial validation gate → artefact writes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SCL_BAND
from stage0.fetch import fetch_chips
from utils.stac import search_sentinel2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixture point set — small (~5 points) Parkinsonia presence locations in the
# Mitchell catchment area (north Queensland). Used only by load-testdata.
# Source: ALA occurrences filtered to lon 141–145, lat -15 to -20.
# ---------------------------------------------------------------------------

FIXTURE_POINTS: list[tuple[str, float, float]] = [
    ("fixture_001", 141.24363,  -18.35002),
    ("fixture_002", 142.86670,  -18.20000),
    ("fixture_003", 141.75000,  -17.16670),
    ("fixture_004", 143.05203,  -17.63864),
    ("fixture_005", 141.54217,  -15.79650),
]

# Sentinel-2 bands to fetch for fixture chips: spectral + SCL + AOT
FIXTURE_BANDS: list[str] = ["B03", "B04", "B08", "B8A", "B11", SCL_BAND, "AOT"]

# Search parameters for fixture STAC query
FIXTURE_BBOX: list[float] = [141.0, -19.0, 143.5, -15.5]  # covers all fixture points
FIXTURE_START: str = "2022-07-01"
FIXTURE_END:   str = "2022-10-31"
FIXTURE_CLOUD_MAX: int = 30

FIXTURE_DIR  = PROJECT_ROOT / "tests" / "fixtures"
SENTINEL_FILE = FIXTURE_DIR / ".fixture_commit"
INPUTS_DIR   = PROJECT_ROOT / "inputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Subcommand: load-testdata
# ---------------------------------------------------------------------------

def cmd_load_testdata(args: argparse.Namespace) -> None:
    """Fetch fixture chips and write the staleness sentinel."""
    import config as _config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    logger.info("load-testdata: searching STAC for fixture items")
    items = search_sentinel2(
        bbox=FIXTURE_BBOX,
        start=FIXTURE_START,
        end=FIXTURE_END,
        cloud_cover_max=FIXTURE_CLOUD_MAX,
        endpoint=_config.STAC_ENDPOINT_ELEMENT84,
        collection=_config.S2_COLLECTION,
    )
    if not items:
        logger.error("No STAC items found for fixture search parameters.")
        sys.exit(1)

    logger.info("load-testdata: fetching chips for %d items × %d points × %d bands",
                len(items), len(FIXTURE_POINTS), len(FIXTURE_BANDS))

    asyncio.run(fetch_chips(
        points=FIXTURE_POINTS,
        items=items,
        bands=FIXTURE_BANDS,
        window_px=5,
        inputs_dir=INPUTS_DIR,
        scl_filter=True,
        max_concurrent=32,
    ))

    # Write sentinel: current git commit hash
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    commit = _current_git_commit()
    SENTINEL_FILE.write_text(commit + "\n")
    logger.info("load-testdata complete — sentinel written: %s (commit %s)",
                SENTINEL_FILE, commit[:12])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Spectral time series training pipeline",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # load-testdata
    p_ltd = sub.add_parser(
        "load-testdata",
        help="Fetch fixture chips from STAC and write the pytest sentinel.",
    )
    p_ltd.set_defaults(func=cmd_load_testdata)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
