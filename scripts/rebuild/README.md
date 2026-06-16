# S1-truncation rebuild campaign

Generated from `cli/chunk.py verify` (see docs/S1-COVERAGE.md, docs/CHUNKSTORE-VERIFY.md).
Each script deletes the FAIL chunk parquets (and, for training tiles, the affected
region/tile training parquets that would otherwise cause `training fetch` to skip the
tile entirely), then refetches through the guarded pipeline.

Order: smallest/safest first. After each, re-run:

    python cli/chunk.py verify --tile <TILE>

and confirm 0 failures before moving on.

Tiles & owners:
  training regions : 55KEU 53LND 54KXA 54KXB 54KWC
  mitchell fetch   : 55KCB 55KBB

Set CHUNKSTORE_DIR (or rely on .env). Scripts assume it via $CHUNKSTORE_DIR.
