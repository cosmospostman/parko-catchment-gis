#!/usr/bin/env bash
# scripts/s1_ebs_sync.sh — sync Sentinel-1 GRD assets from S3 to a local EBS volume.
#
# Usage:
#   ./scripts/s1_ebs_sync.sh --manifest manifest_s1.txt --dest /mnt/s1cache
#
# Reads s3://sentinel-s1-l1c/... URIs from the manifest one per line, copies each
# to --dest preserving the key path, skips files already present.
# Runs copies in parallel (xargs -P 16).

set -euo pipefail

MANIFEST=""
DEST=""
PARALLEL=16

usage() {
    echo "Usage: $0 --manifest <path> --dest <path> [--parallel N]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --dest)     DEST="$2";     shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$MANIFEST" || -z "$DEST" ]] && usage

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 1
fi

mkdir -p "$DEST"

TOTAL=$(grep -c . "$MANIFEST" || true)
echo "Syncing $TOTAL files → $DEST (parallel=$PARALLEL)"

START_TIME=$(date +%s)
TMPDIR_COUNTS=$(mktemp -d)
trap 'rm -rf "$TMPDIR_COUNTS"' EXIT

COPY_SCRIPT=$(mktemp /tmp/s1_copy_XXXXXX.sh)
trap 'rm -f "$COPY_SCRIPT"; rm -rf "$TMPDIR_COUNTS"' EXIT

cat > "$COPY_SCRIPT" <<'COPY_EOF'
#!/usr/bin/env bash
set -euo pipefail
URI="$1"
DEST="$2"
COUNTS_DIR="$3"

# s3://sentinel-s1-l1c/GRD/... → sentinel-s1-l1c/GRD/...
KEY="${URI#s3://}"
LOCAL_PATH="${DEST}/${KEY}"

if [[ -f "$LOCAL_PATH" ]]; then
    echo "skip:$LOCAL_PATH" >> "${COUNTS_DIR}/skipped"
    exit 0
fi

mkdir -p "$(dirname "$LOCAL_PATH")"

# Convert s3://sentinel-s1-l1c/... to public HTTPS URL
BUCKET="sentinel-s1-l1c"
OBJECT_KEY="${KEY#${BUCKET}/}"
HTTPS_URL="https://${BUCKET}.s3.amazonaws.com/${OBJECT_KEY}"

SIZE_BYTES=$(curl -sI "$HTTPS_URL" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r')
if [[ -n "$SIZE_BYTES" ]]; then
    SIZE_MB=$(awk "BEGIN {printf \"%.0f\", $SIZE_BYTES / 1048576}")
    echo "download: $KEY (${SIZE_MB} MB)"
else
    echo "download: $KEY"
fi

curl -fsSL --retry 5 --retry-delay 2 "$HTTPS_URL" -o "$LOCAL_PATH"
echo "sync:$LOCAL_PATH" >> "${COUNTS_DIR}/synced"
COPY_EOF

chmod +x "$COPY_SCRIPT"

while IFS= read -r uri; do
    [[ -z "$uri" ]] && continue
    printf '%s\0%s\0%s\0' "$uri" "$DEST" "$TMPDIR_COUNTS"
done < "$MANIFEST" | xargs -0 -n3 -P "$PARALLEL" bash "$COPY_SCRIPT"

SYNCED=$(wc -l < "${TMPDIR_COUNTS}/synced" 2>/dev/null || echo 0)
SKIPPED=$(wc -l < "${TMPDIR_COUNTS}/skipped" 2>/dev/null || echo 0)

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

TOTAL_BYTES=0
if [[ -s "${TMPDIR_COUNTS}/synced" ]]; then
    while IFS= read -r path; do
        [[ -f "$path" ]] && TOTAL_BYTES=$(( TOTAL_BYTES + $(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0) ))
    done < "${TMPDIR_COUNTS}/synced"
fi
TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_BYTES / 1073741824}")

echo ""
echo "--- Sync complete ---"
echo "  Synced:  $SYNCED files (${TOTAL_GB} GB)"
echo "  Skipped: $SKIPPED files (already present)"
echo "  Elapsed: ${ELAPSED}s"
