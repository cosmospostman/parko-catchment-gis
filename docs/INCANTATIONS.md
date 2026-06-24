# SCORE:

python3 -m tam.pipeline score \
  --checkpoint outputs/models/tam-v10 \
  --location mitchell \
  --pixel-dir /mnt/gis-archive/chunkstore \
  --years 2025 \
  --out-parquet \
  --pmtiles outputs/scores/mitchell \
  --tile-id 54LWJ

# FETCH:
python cli/location.py fetch staaten --years 2025 --tiles 54KWG

# TRAIN:
python -m tam.pipeline train --experiment v10