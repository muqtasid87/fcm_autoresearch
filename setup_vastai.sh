#!/usr/bin/env bash
# setup_vastai.sh HOST PORT [REPO_URL]
#
# One-command setup for a fresh Vast.ai GPU instance.
# Run from your LOCAL machine terminal after renting an instance.
#
# Usage:
#   ./setup_vastai.sh 123.456.789.0 22222
#   ./setup_vastai.sh 123.456.789.0 22222 https://github.com/youruser/yourrepo.git
#
# After this: open VS Code → Remote-SSH → Connect to Host → vastai

set -euo pipefail

HOST="${1:-}"
PORT="${2:-}"
REPO_URL="${3:-}"
REMOTE_DIR="~/fcm"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "Usage: $0 HOST PORT [REPO_URL]"
    echo "  HOST     Vast.ai instance IP (e.g. 123.456.789.0)"
    echo "  PORT     SSH port shown in Vast.ai dashboard (e.g. 22222)"
    echo "  REPO_URL Optional git remote URL. If omitted, reads from git remote get-url origin"
    exit 1
fi

# Auto-detect repo URL from local git config if not provided
if [ -z "$REPO_URL" ]; then
    REPO_URL=$(git remote get-url origin 2>/dev/null || true)
    if [ -z "$REPO_URL" ]; then
        echo "ERROR: No REPO_URL provided and no git remote 'origin' found."
        echo "  Set one with: git remote add origin https://github.com/youruser/yourrepo.git"
        exit 1
    fi
    echo "Using repo URL from git remote: $REPO_URL"
fi

# ── 1. Write ~/.ssh/config entry ───────────────────────────────────────────────
SSH_CONFIG="$HOME/.ssh/config"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

# Remove old 'vastai' entry if it exists
if grep -q "^Host vastai$" "$SSH_CONFIG" 2>/dev/null; then
    echo "Updating existing 'vastai' SSH config entry..."
    python3 - "$SSH_CONFIG" "$HOST" "$PORT" <<'EOF'
import sys, re

config_path, host, port = sys.argv[1], sys.argv[2], sys.argv[3]
with open(config_path, 'r') as f:
    content = f.read()

# Remove existing vastai block
content = re.sub(
    r'^Host vastai\n(?:[ \t]+[^\n]*\n)*',
    '',
    content,
    flags=re.MULTILINE
)
with open(config_path, 'w') as f:
    f.write(content.strip() + '\n')
EOF
fi

cat >> "$SSH_CONFIG" <<EOF

Host vastai
  HostName $HOST
  Port $PORT
  User root
  StrictHostKeyChecking no
  ServerAliveInterval 60
EOF
echo "SSH config written: Host vastai → root@$HOST:$PORT"

# Also save host/port for sync_vastai.sh
echo "${HOST}" > .vastai_host
echo "${PORT}" > .vastai_port
echo "Saved to .vastai_host and .vastai_port"

# ── 2. Remote setup via SSH ────────────────────────────────────────────────────
echo ""
echo "Setting up remote instance..."

ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" bash -s "$REPO_URL" "$REMOTE_DIR" <<'REMOTE_SCRIPT'
REPO_URL="$1"
REMOTE_DIR="$2"

set -euo pipefail

echo "=== Remote setup on $(hostname) ==="
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU detected or nvidia-smi missing)"

# Clone or update repo
if [ -d "$REMOTE_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    cd "$REMOTE_DIR"
    git pull origin master 2>/dev/null || git pull origin main 2>/dev/null
else
    echo "Cloning $REPO_URL → $REMOTE_DIR..."
    git clone "$REPO_URL" "$REMOTE_DIR"
    cd "$REMOTE_DIR"
fi

# Install Python package
echo "Installing Python package..."
pip install --upgrade pip setuptools -q
pip install --ignore-installed . -q

# Create working directories
mkdir -p fcm_autoresearch/data fcm_autoresearch/runs experiments mlruns

# Install Claude Code CLI
echo "Installing Claude Code CLI..."
curl -sL https://dist.claude.ai/linux-x64/latest | tar xz -C /usr/local/bin/ 2>/dev/null || echo "  (claude CLI install skipped)"

echo ""
echo "=== Remote setup complete ==="
echo "Python: $(python3 --version)"
echo "Packages installed: $(pip show mlflow 2>/dev/null | grep Version || echo 'mlflow not found')"
echo "Claude Code: $(which claude && echo 'installed' || echo 'not found — install manually if needed')"
echo "Working dir: $(pwd)"
REMOTE_SCRIPT

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  VS Code: Ctrl+Shift+P → Remote-SSH: Connect to Host → vastai"
echo "  Then open terminal and run:"
echo "    cd fcm && claude --model claude-opus-4-6 --dangerously-skip-permissions"
echo "============================================================"
