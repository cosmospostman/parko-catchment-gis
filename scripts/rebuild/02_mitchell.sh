#!/usr/bin/env bash
# Part 2 — Mitchell tiles: 55KCB, 55KBB.
#
# location.py fetch (mitchell) -> run_tile_pipeline_v2 skips chunks whose output
# parquet already exists, so deleting the FAIL chunks is enough to re-trigger them.
#
# Affected years (from the verify scan):
#   55KCB: 2017-2025  (persistent partial r05_c0x each year + whole-tile 2025)
#   55KBB: 2025 only  (the June-11 whole-tile build)
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

banner "Part 2 — Mitchell tiles (55KCB, 55KBB)"

delete_chunks 55KCB
delete_chunks 55KBB

echo
echo "  Refetch (only the affected years per tile):"
echo "    python cli/location.py fetch mitchell --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 --tiles 55KCB"
echo "    python cli/location.py fetch mitchell --years 2025 --tiles 55KBB"
echo
echo "  Then re-verify:"
echo "    python cli/chunk.py verify --tile 55KCB"
echo "    python cli/chunk.py verify --tile 55KBB"
