# FCM Autoresearch — Next Session Guide

## TL;DR
Best result: **val_loss = 1.527e-07** (reference: 1.13e-07, gap likely closes at 500K samples)  
Champion config: `w256 d5 relu mse he_normal dropout=0 10K_epochs 50K_samples`

---

## Quick Start (new machine)

```bash
git clone https://github.com/muqtasid87/fcm_autoresearch.git
cd fcm_autoresearch

# Install git-lfs to pull the CSV data files
git lfs pull

# Activate environment
source .venv/bin/activate   # or: pip install -e .

# Confirm champion config
cat fcm_autoresearch/current_config.json

# View all 24 prior results
cat fcm_autoresearch/results.tsv

# Run 4-GPU parallel experiments (see Priority Experiments below)
bash fcm_autoresearch/run_trial_gpu.sh fcm_autoresearch/config_gpu0.json 0 "label0" &
bash fcm_autoresearch/run_trial_gpu.sh fcm_autoresearch/config_gpu1.json 1 "label1" &
bash fcm_autoresearch/run_trial_gpu.sh fcm_autoresearch/config_gpu2.json 2 "label2" &
bash fcm_autoresearch/run_trial_gpu.sh fcm_autoresearch/config_gpu3.json 3 "label3" &

# Monitor progress
watch -n 30 "cat fcm_autoresearch/gpu_results/gpu{0,1,2,3}.txt 2>/dev/null"
```

---

## All Discoveries (ordered chronologically)

| Finding | Before | After | Delta |
|---------|--------|-------|-------|
| Even spacing > log spacing | 3.84e-04 | 2.41e-04 | -37% |
| 12 distances optimal (vs 4, 8, 16, 20, 28) | — | — | best of all tested |
| Arc features (radius, direction) don't help | 2.41e-04 | 3.66e-04 | +52% (worse) |
| dropout=0.0 is critical | 3.82e-05 | 2.10e-05 | -45% |
| MSE loss > combined loss | key for long training | — | significant |
| he_normal initializer | key for deep relu | — | significant |
| **10K epochs + patience=500** | 2.10e-05 | 1.527e-07 | **-99.3%** |
| relu+d5 > gelu variants (with long training) | 4.886e-07 | 1.527e-07 | 3× better |

### Key insight on training duration
The original `run_trial.sh` used 500 epochs with patience=100 — ~8× too few.  
With `num_epochs=10000, early_stopping_patience=500`, the champion converged at epoch 2493.  
**Short training completely misleads architecture search** — relu/depth=5 looked terrible at 500 epochs,
but is the best at 10K epochs. This invalidates most early-session architecture comparisons.

---

## Champion Config

```json
{
  "num_distances": 12,
  "target_spacing": "even",
  "cut_type": "both",
  "include_arc_features": false,
  "num_samples": 50000,
  "model_type": "fnn",
  "model_width": 256,
  "model_depth": 5,
  "learning_rate": 1e-3,
  "activation": "relu",
  "dropout_rate": 0.0,
  "weight_initializer": "he_normal",
  "loss_type": "mse",
  "batch_size_train": 16384,
  "batch_size_valid": 65536,
  "num_epochs": 10000,
  "early_stopping_patience": 500
}
```
→ val_loss = **1.527e-07** (best at epoch 2493)  
Reference result (muqtasid87's manual run): val_loss = **1.127e-07** at epoch ~3500 (same config, presumably more data or different seed)

---

## Priority Experiments for Next Session

Update `config_gpu*.json` with these configs before running.

### 1. Scale to 500K samples (highest priority)
Almost certain to beat the 1.127e-07 reference. The gap is likely a data-size issue.

```json
{ ...champion..., "num_samples": 500000 }
```
**ETA**: ~4× longer data generation + similar training (~3h total per GPU). Run on GPU0.

### 2. Lower learning rate lr=5e-4 with champion config
Was running (GPU0 of last session) but killed at ep=1151/10K (val_loss still dropping).
```json
{ ...champion..., "learning_rate": 5e-4 }
```
Run on GPU1.

### 3. Depth=6 (w256+d6+relu)
Was running (GPU1 of last session) at ep=185/10K — showed fast early convergence.
May beat d5 with more depth.
```json
{ ...champion..., "model_depth": 6 }
```
Run on GPU2.

### 4. Width=512 + depth=5 (w512+d5+relu)
Was running (GPU2 of last session) but barely started (ep=61/10K). More capacity may help.
```json
{ ...champion..., "model_width": 512, "model_depth": 5 }
```
Run on GPU3.

### 5. Transformer architecture (not explored yet)
Self-attention over the 12 distance features may capture geometric relationships better than FNN.
Requires implementing `model_type: "transformer"` in train.py first.

### 6. Wider model: w384+d5
Intermediate between w256 and w512.
```json
{ ...champion..., "model_width": 384 }
```

---

## Infrastructure Notes

### run_trial_gpu.sh usage
```bash
bash fcm_autoresearch/run_trial_gpu.sh <config_json> <gpu_id> [label]
# Results written to: fcm_autoresearch/gpu_results/gpu{N}.txt
# Per-GPU data dir:   fcm_autoresearch/data_gpu{N}/
# Unique datasets per trial: avoids appendDataset=True contamination bug
```

### appendDataset bug (FIXED)
`parameters.py` has `appendDataset=True` hardcoded — generate_data.py accumulates CSV rows
across reruns if the same project/dataset name is reused. The fix in `run_trial_gpu.sh`:
unique `--dataset-name "line_${TRIAL_ID}"` and `--project-name "$DATA_DIR/trial_line_${TRIAL_ID}"`
per trial. Do NOT change this or you'll get contaminated data and NaN losses.

### Data directories
- `fcm_autoresearch/data_gpu{0-3}/` — per-GPU data dirs, each with `train.csv` (120MB) + `valid.csv` (30MB)
- `fcm_autoresearch/data/` — original shared data dir, `train.csv` (211MB) + `valid.csv` (53MB)
- All tracked via git-lfs (run `git lfs pull` to download)
- Data was generated with `num_samples=50000` (50K per split: 40K train + 10K valid)

### Monitoring long runs
For 10K-epoch runs, monitor via:
```bash
# Check GPU-specific progress files
cat fcm_autoresearch/gpu_results/gpu0.txt

# Or directly via process output files (if using background tasks)
# Progress is printed every epoch to stdout
```

### Adding results to results.tsv
After each run, manually add a row to `fcm_autoresearch/results.tsv`:
```
<timestamp>  <num_distances>  <spacing>  <arc_features>  fnn  <width>  <depth>  <lr>  <activation>  <dropout>  <val_loss>  <delta_pct>  KEEP/REVERT  <notes>
```
Use the `delta_pct` relative to the current champion.

---

## File Structure Reference

```
fcm_autoresearch/
├── program.md              # Autoresearch loop instructions (read this first)
├── current_config.json     # Current champion config
├── results.tsv             # 24 rows of all experimental results
├── NEXT_SESSION.md         # This file
├── run_trial.sh            # Original single-GPU runner (uses --auto, all GPUs)
├── run_trial_gpu.sh        # New multi-GPU runner (CUDA_VISIBLE_DEVICES, recommended)
├── config_gpu{0-3}.json    # Last session's per-GPU experiment configs
├── gpu_results/            # Output from last GPU runs
├── data_gpu{0-3}/          # Per-GPU datasets (train.csv + valid.csv via LFS)
├── data/                   # Original shared dataset (train.csv + valid.csv via LFS)
└── runs/                   # Training run artifacts (results.json, loss curves, model weights)
```

---

## Session History Summary

**Session 1 (2026-04-06 to 2026-04-07)**

Started at val_loss=3.84e-04 (baseline: 12 log-spaced distances, w256 d4 relu dropout=0.1, 500 epochs).

Systematic sweep findings:
- Representation: 12 even-spaced distances, no arc features → -37%
- Architecture: dropout=0.0 critical → 59% improvement
- Loss: MSE > combined loss
- Init: he_normal for deep relu networks
- **Most impactful**: training duration — 10K epochs + patience=500 → -99.3%

Final runs started (killed when machine shut down, incomplete at ep<200):
- GPU0: w256+d5+relu+lr=5e-4 (1151 epochs completed, still improving)
- GPU1: w256+d6+relu (185 epochs, fast early convergence)
- GPU2: w512+d5+relu (61 epochs, barely started)
- GPU3: w384+d5+relu (106 epochs, promising)
