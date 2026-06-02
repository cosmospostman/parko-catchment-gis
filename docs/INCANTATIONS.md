Score:

python3 -m tam.pipeline sco`re \
  --checkpoint outputs/models/tam-v10-0530 \
  --location mitchell \
  --pixel-dir /mnt/external/mitchell \
  --years 2025 \
  --out-parquet \
  --pmtiles outputs/scores/mitchell \
  --tile-id 54LWH