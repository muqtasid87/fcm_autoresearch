"""Hardware detection and auto-configuration for training.

Detects GPUs, CPU cores, and memory to automatically configure
batch sizes, parallelism, and thread counts.
"""

import os
import math


def detect_hardware() -> dict:
    """Detect available hardware resources.

    Returns
    -------
    dict with keys:
        gpu_count : int
        gpu_names : list of str
        gpu_memory_mb : list of int (per GPU)
        cpu_count : int
        system_ram_gb : float
    """
    hw = {
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory_mb': [],
        'cpu_count': os.cpu_count() or 1,
        'system_ram_gb': _get_system_ram_gb(),
    }

    # Detect GPUs via TensorFlow
    try:
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        hw['gpu_count'] = len(gpus)
        hw['gpu_names'] = [g.name for g in gpus]

        # Try to get GPU memory
        for gpu in gpus:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                mem_mb = details.get('device_memory_size', 0)
                if isinstance(mem_mb, str):
                    # Sometimes returns a string like "24576 MB"
                    mem_mb = int(mem_mb.split()[0])
                hw['gpu_memory_mb'].append(mem_mb)
            except Exception:
                hw['gpu_memory_mb'].append(0)  # unknown
    except ImportError:
        pass

    # Fallback GPU detection via nvidia-smi
    if hw['gpu_count'] == 0:
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                hw['gpu_count'] = len(lines)
                for line in lines:
                    parts = line.split(',')
                    hw['gpu_names'].append(parts[0].strip())
                    try:
                        hw['gpu_memory_mb'].append(int(parts[1].strip()))
                    except (ValueError, IndexError):
                        hw['gpu_memory_mb'].append(0)
        except Exception:
            pass

    return hw


def _get_system_ram_gb() -> float:
    """Get system RAM in GB."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except Exception:
        pass

    # Fallback for non-Linux
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        return 0.0


def suggest_batch_size(gpu_memory_mb: int, num_params: int = 500000,
                       input_dim: int = 12) -> int:
    """Suggest a training batch size based on available GPU memory.

    Heuristic: larger GPU memory → larger batch size.
    Conservative estimates to avoid OOM.
    """
    if gpu_memory_mb <= 0:
        return 8192  # default

    # Rough estimate: model uses ~4 bytes per param, batch uses ~input_dim*4 bytes per sample
    # Reserve 2GB for model + framework overhead
    available_mb = max(gpu_memory_mb - 2048, 1024)

    # Each sample in batch ≈ (input_dim + num_outputs) * 4 bytes * 3 (fwd+bwd+optimizer)
    bytes_per_sample = (input_dim + 4) * 4 * 3
    max_batch = int(available_mb * 1024 * 1024 / bytes_per_sample)

    # Round down to nearest power of 2 for efficiency
    batch_size = 2 ** int(math.log2(max_batch))
    # Clamp between 1024 and 131072
    return max(1024, min(batch_size, 131072))


def auto_configure_training(hw: dict, num_models: int = 1) -> dict:
    """Auto-configure training parameters based on hardware.

    Parameters
    ----------
    hw : dict
        Output from detect_hardware().
    num_models : int
        Number of models to train in parallel.

    Returns
    -------
    dict with keys:
        num_gpus : int
        max_parallel : int
        threads_per_job : int
        cpu_only : bool
        batch_size : int
        summary : str
    """
    gpu_count = hw['gpu_count']
    cpu_count = hw['cpu_count']

    if gpu_count > 0:
        # GPU mode
        cpu_only = False
        num_gpus = gpu_count

        # Max parallel: 2-4 models per GPU depending on memory
        min_gpu_mem = min(hw['gpu_memory_mb']) if hw['gpu_memory_mb'] else 0
        if min_gpu_mem > 16000:
            models_per_gpu = 4
        elif min_gpu_mem > 8000:
            models_per_gpu = 3
        else:
            models_per_gpu = 2

        max_parallel = min(num_gpus * models_per_gpu, max(num_models, 1))

        # Threads: distribute remaining CPU cores
        threads_per_job = max(1, cpu_count // max_parallel)

        # Batch size from GPU memory
        batch_size = suggest_batch_size(
            min_gpu_mem if min_gpu_mem > 0 else 8192
        )

        gpu_names = ', '.join(set(hw['gpu_names'])) if hw['gpu_names'] else 'unknown'
        summary = (f"{gpu_count} GPU(s) [{gpu_names}], "
                   f"{cpu_count} CPUs, "
                   f"max_parallel={max_parallel}, "
                   f"threads/job={threads_per_job}, "
                   f"batch_size={batch_size}")
    else:
        # CPU-only mode
        cpu_only = True
        num_gpus = 0

        # On CPU: limit parallelism to avoid thrashing
        max_parallel = max(1, cpu_count // 4)
        threads_per_job = max(1, cpu_count // max_parallel)
        batch_size = 4096  # smaller for CPU

        summary = (f"CPU-only ({cpu_count} cores), "
                   f"max_parallel={max_parallel}, "
                   f"threads/job={threads_per_job}, "
                   f"batch_size={batch_size}")

    return {
        'num_gpus': num_gpus,
        'max_parallel': max_parallel,
        'threads_per_job': threads_per_job,
        'cpu_only': cpu_only,
        'batch_size': batch_size,
        'summary': summary,
    }


def print_hardware_summary(hw: dict = None):
    """Print a formatted hardware summary."""
    if hw is None:
        hw = detect_hardware()

    print("=" * 60)
    print("  HARDWARE SUMMARY")
    print("=" * 60)
    print(f"  CPUs          : {hw['cpu_count']}")
    print(f"  System RAM    : {hw['system_ram_gb']} GB")
    print(f"  GPUs          : {hw['gpu_count']}")
    for i, name in enumerate(hw['gpu_names']):
        mem = hw['gpu_memory_mb'][i] if i < len(hw['gpu_memory_mb']) else '?'
        print(f"    GPU {i}: {name} ({mem} MB)")
    print("=" * 60)
