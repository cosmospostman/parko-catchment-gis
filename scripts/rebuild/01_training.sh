#!/usr/bin/env bash
# Part 1 — training tiles: 55KEU, 53LND, 54KXA, 54KXB, 54KWC.
#
# Training parquets are cheap to rebuild, so we wipe ALL of them (rather than
# mapping FAIL chunks to specific regions).  `training fetch --all` then rebuilds
# every region, refetching the deleted chunkstore chunks through the guarded
# pipeline.  We still delete the FAIL chunkstore chunks explicitly so the
# idempotent chunk-level skip inside the fetch actually re-fetches them.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

banner "Part 1 — training tiles"

echo "  Deleting ALL training region + tile parquets (cheap to rebuild):"
_run rm -rf -- "$TRAINING_TILES/regions"
# Remove the rolled-up per-tile parquets but keep the regions/ dir layout sane.
for f in "$TRAINING_TILES"/*.parquet; do
  if [[ ! -e "$f" ]]; then continue; fi
  _run rm -f -- "$f"
done

echo
echo "  Deleting FAIL chunkstore chunks for training tiles:"
for t in 55KEU 53LND 54KXA 54KXB 54KWC; do
  delete_chunks "$t"
done

echo
echo "  Refetch (rebuilds region parquets + missing chunkstore chunks):"
echo "    python cli/location.py training fetch --all"
echo
echo "  Then re-verify:"
echo "    python cli/chunk.py verify --tile 55KEU"
echo "    python cli/chunk.py verify --tile 53LND"
echo "    python cli/chunk.py verify --tile 54KXA"
echo "    python cli/chunk.py verify --tile 54KXB"
echo "    python cli/chunk.py verify --tile 54KWC"
