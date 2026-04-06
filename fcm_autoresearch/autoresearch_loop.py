#!/usr/bin/env python3
"""
Autonomous FCM autoresearch loop.
Reads program.md for instructions, edits current_config.json, runs trials, logs results.
Run: python fcm_autoresearch/autoresearch_loop.py
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import random

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "current_config.json"
RESULTS_PATH = SCRIPT_DIR / "results.tsv"
RUN_SCRIPT = SCRIPT_DIR / "run_trial.sh"

def read_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def write_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)

def run_trial():
    """Run one trial and return val_loss, or None on failure."""
    try:
        result = subprocess.run(
            ["bash", str(RUN_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=900  # 15 min max
        )
        # Parse VAL_LOSS=X from output
        for line in result.stdout.split('\n'):
            if line.startswith('VAL_LOSS='):
                return float(line.split('=')[1])
        print(f"ERROR: VAL_LOSS not found in output:\n{result.stdout}\n{result.stderr}")
        return None
    except Exception as e:
        print(f"ERROR running trial: {e}")
        return None

def git_commit(msg):
    subprocess.run(["git", "add", "fcm_autoresearch/current_config.json"],
                   capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], capture_output=True)

def git_reset():
    subprocess.run(["git", "reset", "HEAD~1"], capture_output=True)
    subprocess.run(["git", "checkout", "fcm_autoresearch/current_config.json"],
                   capture_output=True)

def get_best_loss():
    """Read best val_loss from results.tsv."""
    if not RESULTS_PATH.exists():
        return float('inf')
    with open(RESULTS_PATH) as f:
        lines = [l for l in f.readlines()[1:] if l.strip()]  # Skip header
    if not lines:
        return float('inf')
    return float(lines[-1].split('\t')[9])  # Column 9 is val_loss

def log_result(num_distances, spacing, arc_features, model_type, width, depth, lr,
               activation, dropout, val_loss, best_loss, status, notes):
    """Append result to results.tsv."""
    timestamp = datetime.now().isoformat()
    delta = ((val_loss - best_loss) / best_loss * 100) if best_loss != float('inf') else 0
    line = f"{timestamp}\t{num_distances}\t{spacing}\t{arc_features}\t{model_type}\t{width}\t{depth}\t{lr}\t{activation}\t{dropout}\t{val_loss:.6e}\t{delta:.1f}\t{status}\t{notes}\n"
    with open(RESULTS_PATH, 'a') as f:
        f.write(line)

def explore():
    """Main autoresearch loop."""
    print("=" * 70)
    print("  FCM AUTORESEARCH LOOP")
    print("=" * 70)

    best_loss = get_best_loss()
    trial_count = 0

    while True:
        trial_count += 1
        cfg = read_config()
        best_loss = get_best_loss()

        # Phase-based strategy
        if trial_count < 20:
            # Phase 1: data representation search
            strategies = [
                ("Try fewer distances", {"num_distances": 8, "target_spacing": "even"}),
                ("Try uniform spacing instead of log", {"num_distances": 12, "target_spacing": "even"}),
                ("Try more distances, log", {"num_distances": 20, "target_spacing": "log"}),
                ("Try more distances, uniform", {"num_distances": 20, "target_spacing": "even"}),
                ("Try even more distances", {"num_distances": 28, "target_spacing": "log"}),
                ("Add arc features", {"include_arc_features": True}),
                ("Try intermediate count", {"num_distances": 16, "target_spacing": "even"}),
                ("Back to baseline", {"num_distances": 12, "target_spacing": "log", "include_arc_features": False}),
            ]
            hyp, changes = random.choice(strategies)
        else:
            # Phase 2: model optimization
            if cfg["model_type"] == "fnn":
                strategies = [
                    ("Wider model", {"model_width": cfg["model_width"] + 128}),
                    ("Deeper model", {"model_depth": cfg["model_depth"] + 1}),
                    ("Try GELU activation", {"activation": "gelu"}),
                    ("Increase dropout", {"dropout_rate": min(0.3, cfg.get("dropout_rate", 0.1) + 0.1)}),
                    ("Lower learning rate", {"learning_rate": cfg["learning_rate"] / 2}),
                    ("Try transformer", {"model_type": "transformer"}),
                ]
            else:
                strategies = [
                    ("Larger transformer", {"d_model": cfg.get("d_model", 128) + 64}),
                    ("More heads", {"num_heads": cfg.get("num_heads", 4) + 2}),
                    ("Back to FNN", {"model_type": "fnn"}),
                ]
            hyp, changes = random.choice(strategies)

        # Apply changes
        for k, v in changes.items():
            cfg[k] = v
        write_config(cfg)

        print(f"\nTrial {trial_count}: {hyp}")
        print(f"  Config: {changes}")

        # Run trial
        val_loss = run_trial()
        if val_loss is None:
            print("  FAILED - reverting")
            git_reset()
            continue

        # Decide: keep or revert
        best_loss_before = get_best_loss()
        improve_pct = (best_loss_before - val_loss) / best_loss_before * 100 if best_loss_before != float('inf') else 0

        if improve_pct > 0.5:  # >0.5% improvement
            git_commit(f"experiment: {hyp}")
            log_result(cfg["num_distances"], cfg["target_spacing"], cfg["include_arc_features"],
                      cfg["model_type"], cfg["model_width"], cfg["model_depth"], cfg["learning_rate"],
                      cfg["activation"], cfg["dropout_rate"], val_loss, best_loss_before, "KEEP", hyp)
            print(f"  ✓ KEEP (+{improve_pct:.1f}%)")
        else:
            git_reset()
            log_result(cfg["num_distances"], cfg["target_spacing"], cfg["include_arc_features"],
                      cfg["model_type"], cfg["model_width"], cfg["model_depth"], cfg["learning_rate"],
                      cfg["activation"], cfg["dropout_rate"], val_loss, best_loss_before, "REVERT", hyp)
            print(f"  ✗ REVERT ({improve_pct:+.1f}%)")

if __name__ == "__main__":
    explore()
