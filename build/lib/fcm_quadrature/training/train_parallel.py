"""
Parallel training orchestrator for neural networks on multi-GPU or CPU.

This module provides ParallelGPUHyperparameterSearch, which distributes
training jobs across GPUs (round-robin) or CPU cores using subprocesses.

Not intended to be run directly — use train.py as the entry point.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import glob as glob_module


# Suppress TensorFlow warnings in main process
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_single_training(args):
    """
    Run a single training job as a subprocess on a specific GPU.

    Parameters
    ----------
    args : tuple
        (config, job_idx, output_dir, data_paths, gpu_id, threads_per_job)

    Returns
    -------
    dict or None
        Results dictionary or None if failed
    """
    config, job_idx, output_dir, data_paths, gpu_id, threads_per_job = args

    model_name = config['model_name']
    model_output_dir = os.path.join(output_dir, model_name)

    # Create output dir and config file
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    config_file = os.path.join(model_output_dir, 'input_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

    # Build command
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'train_single.py')

    cmd = [
        sys.executable, worker_script,
        '--config', config_file,
        '--train-data', data_paths['train'],
        '--valid-data', data_paths['valid'],
        '--output-dir', model_output_dir,
        '--num-threads', str(threads_per_job),
    ]
    if gpu_id is not None:
        cmd += ['--gpu-id', str(gpu_id)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        results_file = os.path.join(model_output_dir, 'results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            print(f"\n[FAILED] {model_name} (GPU {gpu_id}): No results file")
            if result.stdout:
                print(f"  STDOUT (last 300): {result.stdout[-300:]}")
            if result.stderr:
                print(f"  STDERR (last 500): {result.stderr[-500:]}")
            return None

    except Exception as e:
        print(f"\n[ERROR] {model_name}: {str(e)}")
        return None


class ParallelGPUHyperparameterSearch:
    """
    Orchestrates parallel hyperparameter search across multiple GPUs.

    Models are distributed round-robin across GPUs. Multiple models can
    share a GPU (memory growth is enabled in the worker).
    """

    def __init__(self, output_dir='gpu_search', num_gpus=8,
                 max_parallel=30, threads_per_job=3, cpu_only=False,
                 mlflow_experiment_name=None):
        """
        Parameters
        ----------
        output_dir : str
            Base output directory
        num_gpus : int
            Number of GPUs available
        max_parallel : int
            Maximum concurrent training jobs
        threads_per_job : int
            CPU threads per job for data pipeline
        cpu_only : bool
            If True, train on CPU only (no GPU)
        mlflow_experiment_name : str, optional
            MLflow experiment name. If set, each worker logs to this experiment.
        """
        self.num_gpus = num_gpus
        self.max_parallel = max_parallel
        self.threads_per_job = threads_per_job
        self.cpu_only = cpu_only
        self.mlflow_experiment_name = mlflow_experiment_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f'search_{timestamp}')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.results = []
        self._stop_monitor = False
        self._monitor_thread = None

        mode = "CPU-ONLY" if cpu_only else "GPU"
        if mlflow_experiment_name:
            print(f"  MLflow experiment : {mlflow_experiment_name}")
        print("=" * 70)
        print(f"  PARALLEL {mode} HYPERPARAMETER SEARCH")
        print("=" * 70)
        print(f"  Output directory : {self.output_dir}")
        if cpu_only:
            print(f"  Device           : CPU only")
        else:
            print(f"  GPUs available   : {num_gpus}")
        print(f"  Max parallel jobs: {max_parallel}")
        print(f"  Threads per job  : {threads_per_job}")
        print(f"  Total CPU threads: {max_parallel * threads_per_job}")
        print(f"  Available CPUs   : {os.cpu_count() or '?'}")
        print("=" * 70)

    @classmethod
    def auto_configured(cls, output_dir='gpu_search', num_models=1, **kwargs):
        """Create an instance with auto-detected hardware settings.

        Parameters
        ----------
        output_dir : str
            Base output directory.
        num_models : int
            Number of models to be trained (affects parallelism).
        **kwargs
            Additional kwargs passed to __init__ (overrides auto settings).
        """
        from fcm_quadrature.utils.hardware import detect_hardware, auto_configure_training

        hw = detect_hardware()
        config = auto_configure_training(hw, num_models=num_models)

        init_args = {
            'output_dir': output_dir,
            'num_gpus': config['num_gpus'],
            'max_parallel': config['max_parallel'],
            'threads_per_job': config['threads_per_job'],
            'cpu_only': config['cpu_only'],
        }
        init_args.update(kwargs)
        return cls(**init_args)

    def _read_all_progress(self):
        """Read progress from all model directories."""
        progress_files = glob_module.glob(
            os.path.join(self.output_dir, '*/progress.json'))
        all_progress = []
        for pf in progress_files:
            try:
                with open(pf, 'r') as f:
                    all_progress.append(json.load(f))
            except:
                pass
        return all_progress

    def _display_progress(self, total_jobs):
        """Display live overall progress summary with ETA."""
        while not self._stop_monitor:
            time.sleep(3)

            progress_list = self._read_all_progress()
            if not progress_list:
                continue

            running = [p for p in progress_list if p.get('status') == 'running']
            complete = [p for p in progress_list if p.get('status') == 'complete']
            starting = [p for p in progress_list if p.get('status') == 'starting']

            n_complete = len(complete)
            n_running = len(running)
            n_starting = len(starting)

            # Overall progress: completed jobs + average progress of running jobs
            if running:
                avg_running_pct = sum(p['percent'] for p in running) / len(running)
                # ETA based on running jobs
                etas = [p.get('eta_seconds', 0) for p in running]
                max_eta = max(etas) if etas else 0
            else:
                avg_running_pct = 0
                max_eta = 0

            # Overall percentage
            overall_pct = 100.0 * (n_complete + (n_running * avg_running_pct / 100.0)) / total_jobs

            # Best val loss so far
            all_val_losses = []
            for p in complete:
                if 'val_loss' in p:
                    all_val_losses.append(p['val_loss'])
            for p in running:
                if 'val_loss' in p and p['val_loss'] > 0:
                    all_val_losses.append(p['val_loss'])
            best_loss_str = f"{min(all_val_losses):.2e}" if all_val_losses else "N/A"

            # GPU utilization
            gpu_usage = {}
            for p in running:
                gid = p.get('gpu_id', '?')
                gpu_usage[gid] = gpu_usage.get(gid, 0) + 1

            # Build progress bar
            bar_len = 40
            filled = int(bar_len * overall_pct / 100)
            bar = '#' * filled + '-' * (bar_len - filled)

            # Format ETA
            if max_eta > 3600:
                eta_str = f"{max_eta / 3600:.1f}h"
            elif max_eta > 60:
                eta_str = f"{max_eta / 60:.0f}m"
            else:
                eta_str = f"{max_eta:.0f}s"

            status = (
                f"\r[{bar}] {overall_pct:5.1f}% | "
                f"Done: {n_complete}/{total_jobs} | "
                f"Running: {n_running} | "
                f"Best Loss: {best_loss_str} | "
                f"ETA(slowest): {eta_str}   "
            )
            print(status, end='', flush=True)

    def run(self, configs, data_paths):
        """
        Run parallel hyperparameter search.

        Parameters
        ----------
        configs : list
            List of configuration dictionaries (exactly 30)
        data_paths : dict
            {'train': path, 'valid': path}

        Returns
        -------
        list
            All results
        """
        total_jobs = len(configs)
        if self.cpu_only:
            print(f"\nStarting {total_jobs} training jobs on CPU...")
        else:
            print(f"\nStarting {total_jobs} training jobs across {self.num_gpus} GPUs...")
            print(f"Models per GPU: ~{total_jobs // self.num_gpus} "
                  f"(+{total_jobs % self.num_gpus} extra distributed)")
        print()

        # Start progress monitor thread
        self._stop_monitor = False
        self._monitor_thread = threading.Thread(
            target=self._display_progress, args=(total_jobs,), daemon=True)
        self._monitor_thread.start()

        # Save all configs
        configs_path = os.path.join(self.output_dir, 'all_configs.json')
        with open(configs_path, 'w') as f:
            json.dump(configs, f, indent=2)

        start_time = time.time()
        completed = 0
        failed = 0

        # Assign GPUs round-robin, or None for CPU-only
        # Inject mlflow_experiment_name into each config if set
        job_args = []
        for idx, config in enumerate(configs):
            if self.mlflow_experiment_name and 'mlflow_experiment_name' not in config:
                config['mlflow_experiment_name'] = self.mlflow_experiment_name
            gpu_id = None if self.cpu_only else (idx % self.num_gpus)
            job_args.append((
                config, idx, self.output_dir, data_paths,
                gpu_id, self.threads_per_job
            ))

        # Run with process pool
        with ProcessPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_config = {
                executor.submit(run_single_training, args): args[0]
                for args in job_args
            }

            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result is not None:
                        self.results.append(result)
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"\n[EXCEPTION] {config.get('model_name', '?')}: {e}")
                    failed += 1

                # Save intermediate results periodically
                if (completed + failed) % 5 == 0:
                    self._save_results()

        total_time = time.time() - start_time

        # Stop progress monitor
        self._stop_monitor = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        print()  # New line after progress bar

        mode = "CPU" if self.cpu_only else "GPU"
        print(f"\n{'=' * 70}")
        print(f"  PARALLEL {mode} SEARCH COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Total wall time     : {total_time:.1f}s ({total_time / 3600:.2f} hours)")
        print(f"  Completed           : {completed}/{total_jobs}")
        print(f"  Failed              : {failed}/{total_jobs}")
        if self.results:
            total_train_time = sum(r['training_time_seconds'] for r in self.results)
            print(f"  Total compute-time  : {total_train_time:.1f}s ({total_train_time / 3600:.2f} hours)")
            print(f"  Effective speedup   : {total_train_time / total_time:.1f}x")
            early_stopped = sum(1 for r in self.results if r.get('stopped_early', False))
            print(f"  Early stopped       : {early_stopped}/{completed}")
        print(f"{'=' * 70}")

        # Final save and summary
        self._save_results()
        self._generate_summary()

        return self.results

    def _save_results(self):
        """Save current results to JSON."""
        results_path = os.path.join(self.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _generate_summary(self):
        """Generate summary of results."""
        if not self.results:
            return

        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])

        summary_path = os.path.join(self.output_dir, 'SUMMARY.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PARALLEL GPU HYPERPARAMETER SEARCH SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total models trained: {len(self.results)}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"GPUs used: {self.num_gpus}\n\n")

            f.write("=" * 70 + "\n")
            f.write("ALL MODELS RANKED BY VALIDATION LOSS\n")
            f.write("=" * 70 + "\n\n")

            for i, r in enumerate(sorted_results):
                early = " [EARLY STOP]" if r.get('stopped_early', False) else ""
                f.write(f"Rank {i + 1:2d}: {r['model_name']}{early}\n")
                f.write(f"  Val Loss       : {r['best_val_loss']:.6e}\n")
                f.write(f"  Mean Rel Error : {r['mean_abs_rel_error']:.6f}\n")
                f.write(f"  Train Time     : {r['training_time_seconds']:.1f}s\n")
                f.write(f"  Actual Epochs  : {r.get('actual_epochs', r['config']['num_epochs'])}\n")
                f.write(f"  Best Epoch     : {r['best_epoch']}\n")
                f.write(f"  Params         : {r['total_params']:,}\n")
                f.write(f"  GPU            : {r.get('gpu_id', '?')}\n")
                f.write(f"  Config: w={r['config']['model_width']}, "
                        f"d={r['config']['model_depth']}, "
                        f"lr={r['config']['learning_rate']:.0e}, "
                        f"bs={r['config']['batch_size_train']}, "
                        f"act={r['config']['activation']}, "
                        f"init={r['config'].get('weight_initializer', 'glorot_uniform')}\n\n")

            f.write("=" * 70 + "\n")
            f.write("BEST MODEL\n")
            f.write("=" * 70 + "\n\n")

            best = sorted_results[0]
            f.write(f"Model               : {best['model_name']}\n")
            f.write(f"Validation Loss     : {best['best_val_loss']:.6e}\n")
            f.write(f"Mean Abs Rel Error  : {best['mean_abs_rel_error']:.6f}\n")
            f.write(f"Median Abs Rel Error: {best['median_abs_rel_error']:.6f}\n")
            f.write(f"Max Abs Rel Error   : {best['max_abs_rel_error']:.6f}\n")
            f.write(f"Mean Abs Error      : {best.get('mean_abs_error', 'N/A')}\n")
            f.write(f"Training Time       : {best['training_time_seconds']:.1f}s\n")
            f.write(f"Inference Time      : {best['inference_time_mean']:.4f}s\n")
            f.write(f"Total Parameters    : {best['total_params']:,}\n")
            f.write(f"Best Epoch          : {best['best_epoch']}\n")
            f.write(f"Early Stopped       : {best.get('stopped_early', False)}\n\n")

            f.write("Configuration:\n")
            for k, v in best['config'].items():
                f.write(f"  {k}: {v}\n")

        print(f"\nSummary saved to: {summary_path}")

        best = sorted_results[0]
        print(f"\nBEST MODEL: {best['model_name']}")
        print(f"  Validation Loss : {best['best_val_loss']:.6e}")
        print(f"  Mean Rel Error  : {best['mean_abs_rel_error']:.6f}")
        print(f"  Location        : {best['output_dir']}")


