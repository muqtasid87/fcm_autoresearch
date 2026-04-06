#!/usr/bin/env python
"""
Monitor training progress for parallel model training.

Usage:
    python monitor_progress.py moment_loss_search/run_YYYYMMDD_HHMMSS
    python monitor_progress.py moment_loss_search/run_YYYYMMDD_HHMMSS --interval 30
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


def get_progress_files(base_dir):
    """Find all progress.json files in the output directory."""
    progress_files = []
    for model_dir in sorted(Path(base_dir).iterdir()):
        if model_dir.is_dir() and model_dir.name.startswith('model_'):
            progress_file = model_dir / 'progress.json'
            if progress_file.exists():
                progress_files.append((model_dir.name, progress_file))
    return progress_files


def read_progress(progress_file):
    """Read progress from a JSON file."""
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_progress(base_dir, clear=True):
    """Print current progress of all models."""
    if clear:
        # Clear screen
        print("\033[2J\033[H", end="")

    print("=" * 90)
    print(f"TRAINING PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directory: {base_dir}")
    print("=" * 90)

    progress_files = get_progress_files(base_dir)

    if not progress_files:
        print("No progress files found yet. Training may still be starting...")
        return

    # Categorize by status
    running = []
    complete = []
    starting = []
    not_started = 0

    for model_name, pf in progress_files:
        progress = read_progress(pf)
        if progress is None:
            not_started += 1
            continue

        status = progress.get('status', 'unknown')
        if status == 'complete':
            complete.append((model_name, progress))
        elif status == 'running':
            running.append((model_name, progress))
        elif status == 'starting':
            starting.append((model_name, progress))
        else:
            starting.append((model_name, progress))

    # Print running models
    print(f"\n[RUNNING] {len(running)} models")
    print("-" * 90)
    if running:
        print(f"{'Model':<45} {'Epoch':>8} {'%':>6} {'Loss':>12} {'Val Loss':>12} {'ETA':>10}")
        print("-" * 90)
        for model_name, p in sorted(running, key=lambda x: -x[1].get('percent', 0)):
            name_short = model_name[9:54] if len(model_name) > 54 else model_name[9:]  # Remove 'model_XX_'
            epoch = f"{p.get('epoch', 0)}/{p.get('total_epochs', '?')}"
            pct = p.get('percent', 0)
            loss = p.get('loss', 0)
            val_loss = p.get('val_loss', 0)
            eta = format_time(p.get('eta_seconds', 0))
            print(f"{name_short:<45} {epoch:>8} {pct:>5.1f}% {loss:>12.6e} {val_loss:>12.6e} {eta:>10}")

    # Print completed models
    print(f"\n[COMPLETE] {len(complete)} models")
    if complete:
        print("-" * 90)
        for model_name, p in complete[:5]:  # Show first 5
            name_short = model_name[9:54] if len(model_name) > 54 else model_name[9:]
            val_loss = p.get('val_loss', 0)
            elapsed = format_time(p.get('elapsed_seconds', 0))
            early = " (early stopped)" if p.get('stopped_early', False) else ""
            print(f"  {name_short}: val_loss={val_loss:.6e}, time={elapsed}{early}")
        if len(complete) > 5:
            print(f"  ... and {len(complete) - 5} more")

    # Print summary
    print(f"\n[STARTING] {len(starting)} models initializing")
    print(f"[PENDING] {not_started} models not started")

    total = len(running) + len(complete) + len(starting) + not_started
    print(f"\nTotal: {len(complete)}/{total} complete")


def main():
    parser = argparse.ArgumentParser(description='Monitor parallel training progress')
    parser.add_argument('directory', type=str, help='Training output directory')
    parser.add_argument('--interval', type=int, default=10,
                       help='Refresh interval in seconds (default: 10)')
    parser.add_argument('--once', action='store_true',
                       help='Print once and exit (no continuous monitoring)')

    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    if args.once:
        print_progress(args.directory, clear=False)
    else:
        print(f"Monitoring {args.directory} (refresh every {args.interval}s, Ctrl+C to stop)")
        try:
            while True:
                print_progress(args.directory)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


if __name__ == '__main__':
    main()
