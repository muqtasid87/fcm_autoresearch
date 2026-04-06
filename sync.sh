#!/bin/bash
# Sync helper for remote GPU training workflow.
# Code is versioned with git; data and results are synced with rsync.
#
# Usage:
#   ./sync.sh push-data       # Push data/ to remote
#   ./sync.sh pull-results    # Pull experiments/ and mlruns/ from remote
#   ./sync.sh pull-all        # Pull everything (experiments + mlruns + data)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$SCRIPT_DIR/sync.conf"

if [ ! -f "$CONF" ]; then
    echo "Error: sync.conf not found. Create it with:"
    echo "  REMOTE_HOST=user@gpu-server"
    echo "  REMOTE_DIR=/path/to/hiwi_actual_work"
    exit 1
fi

source "$CONF"

if [ -z "${REMOTE_HOST:-}" ] || [ -z "${REMOTE_DIR:-}" ]; then
    echo "Error: REMOTE_HOST and REMOTE_DIR must be set in sync.conf"
    exit 1
fi

case "${1:-}" in
    push-data)
        echo "Pushing data/ to $REMOTE_HOST:$REMOTE_DIR/data/"
        rsync -avz --progress "$SCRIPT_DIR/data/" "$REMOTE_HOST:$REMOTE_DIR/data/"
        ;;
    pull-results)
        echo "Pulling experiments/ and mlruns/ from $REMOTE_HOST"
        rsync -avz --progress "$REMOTE_HOST:$REMOTE_DIR/experiments/" "$SCRIPT_DIR/experiments/"
        rsync -avz --progress "$REMOTE_HOST:$REMOTE_DIR/mlruns/" "$SCRIPT_DIR/mlruns/" 2>/dev/null || true
        ;;
    pull-all)
        echo "Pulling experiments/, mlruns/, and data/ from $REMOTE_HOST"
        rsync -avz --progress "$REMOTE_HOST:$REMOTE_DIR/experiments/" "$SCRIPT_DIR/experiments/"
        rsync -avz --progress "$REMOTE_HOST:$REMOTE_DIR/mlruns/" "$SCRIPT_DIR/mlruns/" 2>/dev/null || true
        rsync -avz --progress "$REMOTE_HOST:$REMOTE_DIR/data/" "$SCRIPT_DIR/data/"
        ;;
    *)
        echo "Usage: ./sync.sh {push-data|pull-results|pull-all}"
        exit 1
        ;;
esac

echo "Done."
