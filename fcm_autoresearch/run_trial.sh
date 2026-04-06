#!/usr/bin/env bash
# FCM Autoresearch: run one trial (generate data → train → report val_loss)
# Usage: bash fcm_autoresearch/run_trial.sh
# Called by the agent loop from the project root.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="$SCRIPT_DIR/current_config.json"
DATA_DIR="$SCRIPT_DIR/data"
RUNS_DIR="$SCRIPT_DIR/runs"
TRIAL_ID="$(date +%s)"
TRIAL_DIR="$RUNS_DIR/$TRIAL_ID"

mkdir -p "$DATA_DIR" "$TRIAL_DIR"

# ── 1. Parse current_config.json ──────────────────────────────────────────────
eval "$(python3 - <<'EOF'
import json, sys
c = json.load(open("fcm_autoresearch/current_config.json"))
print(f'NUM_DISTANCES={c.get("num_distances", 12)}')
print(f'TARGET_SPACING={c.get("target_spacing", "auto")}')
print(f'CUT_TYPE={c.get("cut_type", "both")}')
print(f'INCLUDE_ARC={str(c.get("include_arc_features", False)).lower()}')
print(f'NUM_SAMPLES={c.get("num_samples", 50000)}')
print(f'NUM_INPUTS={c.get("num_distances", 12) + (2 if c.get("include_arc_features", False) else 0)}')
EOF
)"

echo "============================================================"
echo "  FCM AUTORESEARCH TRIAL $TRIAL_ID"
echo "  distances=$NUM_DISTANCES  spacing=$TARGET_SPACING  cut=$CUT_TYPE"
echo "  arc_features=$INCLUDE_ARC  samples=$NUM_SAMPLES"
echo "============================================================"

# ── 2. Generate data ───────────────────────────────────────────────────────────
HALF_SAMPLES=$(( NUM_SAMPLES / 2 ))

# Generate line cuts
echo ""
echo "Generating LINE cuts (${HALF_SAMPLES} samples)..."
LINE_OUTPUT=$(python3 scripts/generate_data.py \
    --cut-type line \
    --num-distances "$NUM_DISTANCES" \
    --target-spacing "$TARGET_SPACING" \
    --num-samples "$HALF_SAMPLES" \
    --project-name "$DATA_DIR/trial_line" \
    --dataset-name "fcm_auto_line" \
    $([ "$INCLUDE_ARC" = "true" ] && echo "--include-arc-features") \
    2>&1)
echo "$LINE_OUTPUT"
LINE_CSV=$(echo "$LINE_OUTPUT" | grep "Output:" | tail -1 | awk '{print $2}')

# Generate arc cuts
echo ""
echo "Generating ARC cuts (${HALF_SAMPLES} samples)..."
ARC_OUTPUT=$(python3 scripts/generate_data.py \
    --cut-type arc \
    --num-distances "$NUM_DISTANCES" \
    --target-spacing "$TARGET_SPACING" \
    --num-samples "$HALF_SAMPLES" \
    --project-name "$DATA_DIR/trial_arc" \
    --dataset-name "fcm_auto_arc" \
    $([ "$INCLUDE_ARC" = "true" ] && echo "--include-arc-features") \
    2>&1)
echo "$ARC_OUTPUT"
ARC_CSV=$(echo "$ARC_OUTPUT" | grep "Output:" | tail -1 | awk '{print $2}')

if [ -z "$LINE_CSV" ] || [ ! -f "$LINE_CSV" ]; then
    echo "ERROR: Line CSV not found. Generation failed." >&2
    exit 1
fi
if [ -z "$ARC_CSV" ] || [ ! -f "$ARC_CSV" ]; then
    echo "ERROR: Arc CSV not found. Generation failed." >&2
    exit 1
fi

# ── 3. Merge and split into train/valid ────────────────────────────────────────
TRAIN_CSV="$DATA_DIR/train.csv"
VALID_CSV="$DATA_DIR/valid.csv"

python3 - <<EOF
import numpy as np
import pandas as pd

line = pd.read_csv("$LINE_CSV", header=None)
arc  = pd.read_csv("$ARC_CSV",  header=None)
data = pd.concat([line, arc], ignore_index=True)
data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

split = int(len(data) * 0.8)
data.iloc[:split].to_csv("$TRAIN_CSV", header=False, index=False)
data.iloc[split:].to_csv("$VALID_CSV", header=False, index=False)
print(f"  Merged {len(line)} line + {len(arc)} arc = {len(data)} total")
print(f"  Train: {split} rows  |  Valid: {len(data)-split} rows")
EOF

# ── 4. Write train_config.json from model portion of current_config.json ───────
TRAIN_CONFIG="$TRIAL_DIR/train_config.json"
python3 - <<EOF
import json

MODEL_FIELDS = {
    "model_type", "model_width", "model_depth", "activation", "dropout_rate",
    "learning_rate", "num_epochs", "early_stopping_patience", "loss_type",
    "loss_alpha", "loss_beta", "batch_size", "seed",
    "d_model", "num_heads", "num_layers", "d_ff",
}

c = json.load(open("fcm_autoresearch/current_config.json"))
model_cfg = {k: v for k, v in c.items() if k in MODEL_FIELDS}
model_cfg["num_inputs"] = $NUM_INPUTS
model_cfg["num_outputs"] = 4
model_cfg["mlflow_experiment_name"] = "fcm_autoresearch"

with open("$TRAIN_CONFIG", "w") as f:
    json.dump(model_cfg, f, indent=2)
print(f"  Train config: {model_cfg}")
EOF

# ── 5. Train ───────────────────────────────────────────────────────────────────
echo ""
echo "Training..."
python3 scripts/train.py "$TRAIN_CONFIG" \
    --train-data "$TRAIN_CSV" \
    --valid-data "$VALID_CSV" \
    --output-dir "$TRIAL_DIR" \
    --num-gpus 1 \
    --cpu-only \
    --mlflow-experiment fcm_autoresearch

# ── 6. Extract metric ──────────────────────────────────────────────────────────
RESULTS_JSON=$(python3 -c "
import json, glob
matches = sorted(glob.glob('$TRIAL_DIR/*/results.json'))
if not matches:
    matches = glob.glob('$TRIAL_DIR/results.json')
if matches:
    r = json.load(open(matches[-1]))
    print(r['best_val_loss'])
else:
    print('NOT_FOUND')
")

if [ "$RESULTS_JSON" = "NOT_FOUND" ]; then
    echo "ERROR: results.json not found after training." >&2
    exit 1
fi

echo ""
echo "============================================================"
echo "VAL_LOSS=$RESULTS_JSON"
echo "TRIAL_DIR=$TRIAL_DIR"
echo "============================================================"
