#!/usr/bin/env bash
# sync_vastai.sh ACTION
#
# Sync results/configs between local machine and active Vast.ai instance.
# Reads HOST and PORT from .vastai_host / .vastai_port (written by setup_vastai.sh).
#
# Commands:
#   ./sync_vastai.sh pull-results   Pull results.tsv + runs/ back locally
#   ./sync_vastai.sh pull-all       Pull results + runs + mlruns/
#   ./sync_vastai.sh push-configs   Push fcm_autoresearch/*.json + *.md + configs/
#   ./sync_vastai.sh status         Show active trials (tail results.tsv on remote)

set -euo pipefail

ACTION="${1:-}"
REMOTE_DIR="fcm"

# ── Load connection details ────────────────────────────────────────────────────
if [ -f ".vastai_host" ] && [ -f ".vastai_port" ]; then
    HOST=$(cat .vastai_host)
    PORT=$(cat .vastai_port)
else
    # Fallback: parse from ~/.ssh/config
    HOST=$(grep -A5 "^Host vastai$" ~/.ssh/config | grep HostName | awk '{print $2}')
    PORT=$(grep -A5 "^Host vastai$" ~/.ssh/config | grep Port | awk '{print $2}')
fi

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "ERROR: No Vast.ai connection info found."
    echo "  Run: ./setup_vastai.sh HOST PORT first"
    exit 1
fi

echo "Connecting to vastai ($HOST:$PORT)..."
RSYNC="rsync -avz --progress -e \"ssh -p $PORT -o StrictHostKeyChecking=no\""
SSH="ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST"

case "$ACTION" in

  pull-results)
    echo "Pulling fcm_autoresearch/results.tsv and runs/..."
    rsync -avz --progress -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "root@$HOST:~/$REMOTE_DIR/fcm_autoresearch/results.tsv" \
        ./fcm_autoresearch/results.tsv 2>/dev/null || echo "  (results.tsv not found yet)"
    rsync -avz --progress -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "root@$HOST:~/$REMOTE_DIR/fcm_autoresearch/runs/" \
        ./fcm_autoresearch/runs/ 2>/dev/null || echo "  (runs/ is empty)"
    echo "Done. Runs pulled to ./fcm_autoresearch/runs/"
    ;;

  pull-all)
    echo "Pulling results + runs + mlruns..."
    rsync -avz --progress -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "root@$HOST:~/$REMOTE_DIR/fcm_autoresearch/results.tsv" \
        ./fcm_autoresearch/results.tsv 2>/dev/null || true
    rsync -avz --progress -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "root@$HOST:~/$REMOTE_DIR/fcm_autoresearch/runs/" \
        ./fcm_autoresearch/runs/ 2>/dev/null || true
    rsync -avz --progress -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "root@$HOST:~/$REMOTE_DIR/mlruns/" \
        ./mlruns/ 2>/dev/null || echo "  (mlruns/ not found)"
    echo "Done. Run: mlflow ui to browse results."
    ;;

  push-configs)
    echo "Pushing configs and autoresearch files..."
    rsync -avz --progress -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        ./fcm_autoresearch/program.md \
        ./fcm_autoresearch/current_config.json \
        ./fcm_autoresearch/run_trial.sh \
        "root@$HOST:~/$REMOTE_DIR/fcm_autoresearch/"
    rsync -avz --progress -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        ./configs/ \
        "root@$HOST:~/$REMOTE_DIR/configs/"
    echo "Done. Configs pushed."
    ;;

  status)
    echo "Remote results.tsv (last 10 trials):"
    ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
        "tail -10 ~/$REMOTE_DIR/fcm_autoresearch/results.tsv 2>/dev/null || echo '(no results yet)'"
    echo ""
    echo "Remote GPU status:"
    ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo '(nvidia-smi not available)'"
    ;;

  *)
    echo "Usage: $0 {pull-results|pull-all|push-configs|status}"
    echo ""
    echo "  pull-results  Pull results.tsv + runs/ from remote"
    echo "  pull-all      Also pull mlruns/ for MLflow"
    echo "  push-configs  Push updated configs and program.md to remote"
    echo "  status        Show last 10 results and GPU utilization"
    exit 1
    ;;

esac
