#!/usr/bin/env bash
# Shared helpers for the S1-truncation rebuild campaign.
# Source this from a per-part script.  Nothing deletes unless --apply is passed.
set -euo pipefail

: "${CHUNKSTORE_DIR:=/mnt/gis-archive/chunkstore}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FAILLISTS="$REPO_ROOT/scripts/rebuild/faillists"
TRAINING_TILES="$REPO_ROOT/data/training/tiles"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then APPLY=1; fi

_run() {  # echo, and run only when --apply
  echo "  \$ $*"
  if [[ $APPLY -eq 1 ]]; then "$@"; fi
}

# Delete the FAIL chunk parquets (+ pixel_count sidecars) listed for a tile.
delete_chunks() {
  local tile="$1" list="$FAILLISTS/$1.txt"
  [[ -f "$list" ]] || { echo "  !! no fail list: $list"; exit 1; }
  echo "  Deleting $(wc -l < "$list") FAIL chunk parquets for $tile:"
  while IFS= read -r p; do
    if [[ -z "$p" ]]; then continue; fi
    _run rm -f -- "$p"
    _run rm -f -- "${p%.parquet}.pixel_count"
  done < "$list"
}

banner() {
  echo; echo "=== $* ==="
  if [[ $APPLY -eq 0 ]]; then
    echo "  (DRY RUN — pass --apply to execute)"
  fi
}
