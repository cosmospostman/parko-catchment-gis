"""utils/report.py — VerificationReport dataclass with atomic JSON roundtrip."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    step: str
    year: int
    status: str  # "PASS" or "FAIL"
    checks_passed: int
    checks_failed: int
    messages: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)


def save_report(report: VerificationReport, path: Path) -> None:
    """Atomically append a VerificationReport to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing reports
    existing: List[Dict] = []
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Could not read existing report file %s: %s", path, exc)
            existing = []

    existing.append(asdict(report))

    # Write atomically
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp_path, path)
    logger.info("Verification report saved: %s [%s]", path, report.status)


def load_report(path: Path) -> List[VerificationReport]:
    """Load all VerificationReport entries from a JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return [VerificationReport(**entry) for entry in data]
