#!/usr/bin/env bash
# FCM Autoresearch: run one trial on a specific GPU with a specific config
# Usage: bash fcm_autoresearch/run_trial_gpu.sh <config_file> <gpu_id> [label]
#   config_file : path to JSON config (relative to project root)
#   gpu_id      : 0, 1, 2, or 3
#   label       : optional human label for output
#
# Writes: fcm_autoresearch/runs/<trial_id>/  (standard layout)
#         fcm_autoresearch/gpu_results/gpu<gpu_id>.txt  (VAL_LOSS=... line for agent to read)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-fcm_autoresearch/current_config.json}"
GPU_ID="${2:-0}"
LABEL="${3:-gpu${GPU_ID}}"

DATA_DIR="$SCRIPT_DIR/data_gpu${GPU_ID}"
RUNS_DIR="$SCRIPT_DIR/runs"
TRIAL_ID="$(date +%s)_gpu${GPU_ID}"
TRIAL_DIR="$RUNS_DIR/$TRIAL_ID"
GPU_RESULTS_DIR="$SCRIPT_DIR/gpu_results"

mkdir -p "$DATA_DIR" "$TRIAL_DIR" "$GPU_RESULTS_DIR"

# ── 1. Parse config ────────────────────────────────────────────────────────────
eval "$(python3 - <<EOF
import json, sys
c = json.load(open("$CONFIG"))
print(f'NUM_DISTANCES={c.get("num_distances", 12)}')
print(f'TARGET_SPACING={c.get("target_spacing", "auto")}')
print(f'CUT_TYPE={c.get("cut_type", "both")}')
print(f'INCLUDE_ARC={str(c.get("include_arc_features", False)).lower()}')
print(f'NUM_SAMPLES={c.get("num_samples", 50000)}')
print(f'NUM_INPUTS={c.get("num_distances", 12) + (2 if c.get("include_arc_features", False) else 0)}')
EOF
)"

echo "============================================================"
echo "  FCM AUTORESEARCH TRIAL $TRIAL_ID  [GPU $GPU_ID] [$LABEL]"
echo "  distances=$NUM_DISTANCES  spacing=$TARGET_SPACING  cut=$CUT_TYPE"
echo "  arc_features=$INCLUDE_ARC  samples=$NUM_SAMPLES"
echo "============================================================"

# Use trial-unique dataset names so appendDataset=True never collides with prior runs
DS_LINE="line_${TRIAL_ID}"
DS_ARC="arc_${TRIAL_ID}"
PROJ_LINE="$DATA_DIR/trial_line_${TRIAL_ID}"
PROJ_ARC="$DATA_DIR/trial_arc_${TRIAL_ID}"

# ── 2. Generate data ───────────────────────────────────────────────────────────
HALF_SAMPLES=$(( NUM_SAMPLES / 2 ))

echo ""
echo "[GPU $GPU_ID] Generating LINE cuts (${HALF_SAMPLES} samples)..."
LINE_OUTPUT=$(python3 scripts/generate_data.py \
    --cut-type line \
    --num-distances "$NUM_DISTANCES" \
    --target-spacing "$TARGET_SPACING" \
    --num-samples "$HALF_SAMPLES" \
    --project-name "$PROJ_LINE" \
    --dataset-name "$DS_LINE" \
    --workers 8 \
    $([ "$INCLUDE_ARC" = "true" ] && echo "--include-arc-features") \
    2>&1)
echo "$LINE_OUTPUT"
LINE_CSV=$(echo "$LINE_OUTPUT" | grep "Output:" | tail -1 | awk '{print $2}')

echo ""
echo "[GPU $GPU_ID] Generating ARC cuts (${HALF_SAMPLES} samples)..."
ARC_OUTPUT=$(python3 scripts/generate_data.py \
    --cut-type arc \
    --num-distances "$NUM_DISTANCES" \
    --target-spacing "$TARGET_SPACING" \
    --num-samples "$HALF_SAMPLES" \
    --project-name "$PROJ_ARC" \
    --dataset-name "$DS_ARC" \
    --workers 8 \
    $([ "$INCLUDE_ARC" = "true" ] && echo "--include-arc-features") \
    2>&1)
echo "$ARC_OUTPUT"
ARC_CSV=$(echo "$ARC_OUTPUT" | grep "Output:" | tail -1 | awk '{print $2}')

if [ -z "$LINE_CSV" ] || [ ! -f "$LINE_CSV" ]; then
    echo "ERROR: Line CSV not found. Generation failed." >&2
    echo "VAL_LOSS=ERROR" > "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
    exit 1
fi
if [ -z "$ARC_CSV" ] || [ ! -f "$ARC_CSV" ]; then
    echo "ERROR: Arc CSV not found. Generation failed." >&2
    echo "VAL_LOSS=ERROR" > "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
    exit 1
fi

# ── 3. Merge and split ─────────────────────────────────────────────────────────
TRAIN_CSV="$DATA_DIR/train.csv"
VALID_CSV="$DATA_DIR/valid.csv"

python3 - <<EOF
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

# ── 4. Write train config ──────────────────────────────────────────────────────
TRAIN_CONFIG="$TRIAL_DIR/train_config.json"
python3 - <<EOF
import json
MODEL_FIELDS = {
    "model_type", "model_width", "model_depth", "activation", "dropout_rate",
    "learning_rate", "num_epochs", "early_stopping_patience", "loss_type",
    "loss_alpha", "loss_beta", "batch_size", "batch_size_train", "batch_size_valid",
    "weight_initializer", "seed",
    "d_model", "num_heads", "num_layers", "d_ff",
}
c = json.load(open("$CONFIG"))
model_cfg = {k: v for k, v in c.items() if k in MODEL_FIELDS}
model_cfg["num_inputs"] = $NUM_INPUTS
model_cfg["num_outputs"] = 4
model_cfg["mlflow_experiment_name"] = "fcm_autoresearch"
with open("$TRAIN_CONFIG", "w") as f:
    json.dump(model_cfg, f, indent=2)
print(f"  Train config: {model_cfg}")
EOF

# ── 5. Train on single GPU ─────────────────────────────────────────────────────
echo ""
echo "[GPU $GPU_ID] Training..."
CUDA_VISIBLE_DEVICES="$GPU_ID" python3 scripts/train.py "$TRAIN_CONFIG" \
    --train-data "$TRAIN_CSV" \
    --valid-data "$VALID_CSV" \
    --output-dir "$TRIAL_DIR" \
    --num-gpus 1 \
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

echo ""
echo "============================================================"
echo "VAL_LOSS=$RESULTS_JSON"
echo "TRIAL_DIR=$TRIAL_DIR"
echo "GPU=$GPU_ID"
echo "LABEL=$LABEL"
echo "CONFIG=$CONFIG"
echo "============================================================"

# Write result file for agent to read
echo "VAL_LOSS=$RESULTS_JSON" > "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
echo "TRIAL_DIR=$TRIAL_DIR" >> "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
echo "LABEL=$LABEL" >> "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
echo "CONFIG=$CONFIG" >> "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
echo "NUM_DISTANCES=$NUM_DISTANCES" >> "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
echo "TARGET_SPACING=$TARGET_SPACING" >> "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
echo "INCLUDE_ARC=$INCLUDE_ARC" >> "$GPU_RESULTS_DIR/gpu${GPU_ID}.txt"
