# FCM Autoresearch Program

You are an autonomous research agent. Your goal is to find the best way to model and solve the FCM (Finite Cell Method) cut cell quadrature weight prediction problem. You run 24/7, improving one thing at a time, tracking every result.

## Problem Description

A unit cell [0,1]² is cut by either a straight line or an arc. We need to predict 4 quadrature weights (for basis functions {1, x, y, xy}) using Green's theorem applied to the cut cell boundary.

**Inputs**: N signed distances from the cut line/arc to N target points on the cell.
**Outputs**: 4 quadrature weights.
**All datasets**: include BOTH line cuts and arc cuts (`cut_type = "both"`).

The key scientific question: **what is the best way to represent the problem as inputs?**
- More distances? Fewer? Where to place the measurement points?
- Near the boundary (log-spaced) or uniformly spread (evenly-spaced)?
- For arc cuts: do raw distances suffice, or do curvature features (radius, direction) help?

## Your Single Metric

`best_val_loss` (lower is better) from `fcm_autoresearch/runs/<timestamp>/results.json`.

## What You Control

Edit `fcm_autoresearch/current_config.json`. All fields are fair game.

### Data fields (problem representation):
- `num_distances`: integer — try 4, 8, 12, 16, 20, 28
  - Formula: evenly-spaced supports 4+4k (8, 12, 16, 20, 24, 28)
  - Formula: log-spaced supports 4+8k (12, 20, 28)
- `target_spacing`: "log" (near-boundary bias) or "even" (uniform)
  - Note: 8 and 16 can only be "even"; 12, 20, 28 can be either
- `include_arc_features`: true → adds 2 extra inputs (radius_ratio, direction) for arc cuts

### Model fields:
- `model_type`: "fnn" or "transformer"
- `model_width`: 64, 128, 256, 512
- `model_depth`: 2, 3, 4, 5, 6
- `learning_rate`: 1e-4 to 5e-3
- `activation`: "relu", "gelu", "tanh"
- `dropout_rate`: 0.0 to 0.3

### Training fields (usually leave these alone):
- `num_samples`: 50000 for exploration; increase to 500000 to confirm a good data config
- `num_epochs`: 500
- `early_stopping_patience`: 50

## The Loop (NEVER STOP)

```
LOOP FOREVER:
  1. Read current_config.json and results.tsv (see where you are)
  2. Form a hypothesis: "I think changing X to Y will help because..."
  3. Edit current_config.json with your change
  4. git add fcm_autoresearch/current_config.json && git commit -m "experiment: <hypothesis>"
  5. bash fcm_autoresearch/run_trial.sh
  6. Read the printed VAL_LOSS=<value>
  7. Compare to best_val_loss in results.tsv
  8. If improved (>0.5% better): keep the commit, record "KEEP" in results.tsv
     If not improved: git reset HEAD~1, restore config, record "REVERT" in results.tsv
  9. Append row to results.tsv
  10. Go to step 1
```

## Decision Rules

1. **Simplicity bias**: prefer fewer distances if loss is within 1% — simpler inputs generalize better
2. **One change at a time**: change either the data config OR the model, not both together
3. **Fast-eval first**: use `num_samples: 50000` to explore quickly
4. **Confirm data changes**: before committing to a new data representation, re-run with `num_samples: 500000` to confirm it holds on more data
5. **Revert threshold**: revert if loss is more than 0.5% worse than current best
6. **Model vs data**: first find a good data representation (first ~20 trials), then optimize the model

## Exploration Strategy

### Phase 1: Data Representation (first ~20 trials)
Start with the baseline (12 distances, log-spacing). Try:
- `num_distances: 8` with `target_spacing: "even"` — fewer, uniform
- `num_distances: 12` with `target_spacing: "even"` — same count, uniform vs log
- `num_distances: 20` with `target_spacing: "log"` — more, near-boundary
- `num_distances: 20` with `target_spacing: "even"` — more, uniform
- `include_arc_features: true` with current best distances — does curvature help?
- `num_distances: 16` with `target_spacing: "even"` — intermediate count

### Phase 2: Model Optimization (after finding best data config)
With the best data config locked in, try:
- Wider/deeper FNN
- Transformer (`model_type: "transformer"`)
- Different activations (gelu often beats relu for smooth functions)
- Dropout tuning

### Phase 3: Scale Up
Once best (data, model) pair found: increase `num_samples` to 500000 and run final validation.

## results.tsv Format

Append one tab-separated row per trial:
```
timestamp\tnum_distances\tspacing\tarc_features\twidth\tdepth\tlr\tactivation\tdropout\tval_loss\tdelta_pct\tstatus\tnotes
```
- `delta_pct`: percentage change from previous best (negative = improvement)
- `status`: KEEP or REVERT
- `notes`: your 1-sentence hypothesis that motivated this trial

## Important Constraints

- NEVER edit data generation source code (parameters.py, job.py, mesh.py, etc.)
- NEVER edit training source code (train_single.py, data_loading.py, etc.)
- ONLY edit `fcm_autoresearch/current_config.json`
- If run_trial.sh fails with an error, diagnose it, fix current_config.json (don't commit), retry
- Keep going — don't stop unless the terminal is interrupted
