"""
Data generation script for FCM quadrature weight prediction.

Supports line cuts, arc cuts, or both. Configurable number of distance
features and dataset size.

Usage:
    # Quick test (line cuts, 12 distances, small dataset)
    python scripts/generate_data.py --mode test

    # Arc cuts with 16 distance features
    python scripts/generate_data.py --cut-type arc --num-distances 16 --mode production

    # Both line + arc, 1M samples
    python scripts/generate_data.py --cut-type both --num-samples 1000000

    # Custom configuration
    python scripts/generate_data.py --cut-type line --num-distances 20 --num-samples 500000 --workers 64
"""

import os
import sys
import json
import time
import math
import argparse
import multiprocessing as mp

from fcm_quadrature.data_generation.job import Project
from fcm_quadrature.data_generation.parameters import (
    ProjectParameter,
    QuickTestConfig,
    ProductionConfig,
    DebugConfig,
    MillionSamplesConfig,
    ArcConfig,
    ArcWithFeaturesConfig,
    CombinedConfig,
)


def estimate_samples_per_edge(target_samples, use_all_edges=True):
    """Compute numSamplesStartEdge to get approximately target_samples.

    Formula: total = (edges * n) * (3 * n)
    where edges = 4 if use_all_edges else 1.
    """
    edges = 4 if use_all_edges else 1
    # total = edges * n * 3n = 3 * edges * n^2
    n = math.sqrt(target_samples / (3 * edges))
    # Must be odd
    n = int(n)
    if n % 2 == 0:
        n += 1
    return max(3, n)


def build_config(args):
    """Build a ProjectParameter config from CLI arguments."""
    # Start with a preset if mode is given
    if args.mode == 'test':
        par = QuickTestConfig()
    elif args.mode == 'debug':
        par = DebugConfig()
    elif args.mode == 'production':
        par = ProductionConfig()
    else:
        par = ProjectParameter(projectName='Custom', datasetName='Dataset')

    # Override project/dataset names
    cut_label = args.cut_type.capitalize()
    dist_label = f"{args.num_distances}d"
    par.projectName = args.project_name or f'FCM_{cut_label}_{dist_label}'
    par.datasetName = args.dataset_name or f'{cut_label}_{dist_label}_Data'

    # Cut type
    par.cutSectionType = args.cut_type if args.cut_type != 'both' else 'line'

    # Dynamic distances and spacing strategy
    strategy = getattr(args, 'target_spacing', 'auto')
    if args.num_distances != 12 or strategy != 'auto':
        par.configure_targets(args.num_distances, strategy=strategy)

    # Dataset size
    if args.num_samples:
        n = estimate_samples_per_edge(args.num_samples, use_all_edges=True)
        par.numSamplesStartEdge = n
        par.numSamplesEndEdge = n
        par.useAllEdgesAsStart = True

    # Arc parameters
    if args.cut_type in ('arc', 'both'):
        par.numRadiusRatio = args.arc_num_radius
        par.minRadiusRatio = args.arc_min_radius
        par.maxRadiusRatio = args.arc_max_radius

    # Include arc features (curvature columns)
    par.includeArcFeatures = args.include_arc_features

    # Workers
    if args.workers:
        par.numWorkers = args.workers
        par.subListLength = args.workers * 5

    return par


def save_metadata(par, output_csv_path, elapsed, num_samples_actual, cut_type, target_spacing='auto'):
    """Save a JSON metadata file alongside the CSV."""
    meta = {
        'cut_type': cut_type,
        'num_inputs': par.get_num_distances() + (2 if par.includeArcFeatures else 0),
        'num_outputs': 4,
        'num_distance_features': par.get_num_distances(),
        'include_arc_features': par.includeArcFeatures,
        'num_samples': num_samples_actual,
        'num_samples_start_edge': par.numSamplesStartEdge,
        'num_samples_end_edge': par.numSamplesEndEdge,
        'use_all_edges_as_start': par.useAllEdgesAsStart,
        'target_types': par.targets,
        'target_spacing': target_spacing,
        'generation_time_seconds': round(elapsed, 2),
    }
    if cut_type in ('arc', 'both'):
        meta['arc_min_radius'] = par.minRadiusRatio
        meta['arc_max_radius'] = par.maxRadiusRatio
        meta['arc_num_radius'] = par.numRadiusRatio

    meta_path = output_csv_path.replace('.csv', '_metadata.json')
    if not meta_path.endswith('.json'):
        meta_path = output_csv_path + '.metadata.json'

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_path}")
    return meta_path


def run_generation(par, label=''):
    """Run data generation with the given parameters. Returns (csv_path, elapsed, job_count)."""
    if os.path.exists(par.logName):
        os.remove(par.logName)
    if os.path.exists(par.projectName + '.Project'):
        os.remove(par.projectName + '.Project')

    print(f"\n{'=' * 70}")
    print(f"  DATA GENERATION{' - ' + label if label else ''}")
    print(f"{'=' * 70}")
    print(f"  Project        : {par.projectName}")
    print(f"  Cut type       : {par.cutSectionType}")
    print(f"  Distances      : {par.get_num_distances()}")
    print(f"  Arc features   : {par.includeArcFeatures}")
    print(f"  Edge samples   : {par.numSamplesStartEdge} × {par.numSamplesEndEdge}")
    print(f"  All edges      : {par.useAllEdgesAsStart}")
    print(f"  Workers        : {par.numWorkers}")
    print(f"{'=' * 70}")

    start = time.perf_counter()
    project = Project(par)
    project.parallelExecute()
    elapsed = time.perf_counter() - start

    csv_path = project.dataset.csvPath
    job_count = project.jobCounter
    error_count = project.errorCounter

    print(f"\n  Completed: {job_count} jobs, {error_count} errors, {elapsed:.1f}s")
    print(f"  Output: {csv_path}")

    return csv_path, elapsed, job_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate FCM quadrature training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_data.py --mode test
  python scripts/generate_data.py --cut-type arc --num-distances 16 --mode production
  python scripts/generate_data.py --cut-type both --num-samples 1000000 --workers 64
  python scripts/generate_data.py --cut-type arc --include-arc-features --num-samples 500000
        """
    )

    parser.add_argument('--mode', type=str, default='production',
                        choices=['test', 'debug', 'production', 'custom'],
                        help='Preset mode (default: production)')
    parser.add_argument('--cut-type', type=str, default='line',
                        choices=['line', 'arc', 'both'],
                        help='Cut type (default: line)')
    parser.add_argument('--num-distances', type=int, default=12,
                        help='Number of distance input features (default: 12)')
    parser.add_argument('--target-spacing', type=str, default='auto',
                        choices=['auto', 'log', 'even'],
                        help='Target point placement strategy: log (near-boundary, 4+8k counts), '
                             'even (uniform, 4+4k counts), auto (log if possible, else even). '
                             'Use even to get uniform spacing for 12/20/28 distances instead of default log.')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Approximate number of samples to generate')
    parser.add_argument('--include-arc-features', action='store_true',
                        help='Append arc curvature features (radius, direction) to input')
    parser.add_argument('--arc-num-radius', type=int, default=10,
                        help='Number of radius samples for arc cuts (default: 10)')
    parser.add_argument('--arc-min-radius', type=float, default=0.5,
                        help='Minimum radius ratio for arcs (default: 0.5)')
    parser.add_argument('--arc-max-radius', type=float, default=10.0,
                        help='Maximum radius ratio for arcs (default: 10)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--project-name', type=str, default=None,
                        help='Override project name')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Override dataset name')
    parser.add_argument('--inspect', action='store_true',
                        help='Inspect output CSV after generation')

    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    if args.cut_type == 'both':
        # Generate line data
        par_line = build_config(args)
        par_line.cutSectionType = 'line'
        par_line.projectName += '_Line'
        par_line.datasetName += '_Line'
        csv_line, elapsed_line, count_line = run_generation(par_line, 'LINE CUTS')
        save_metadata(par_line, csv_line, elapsed_line, count_line, 'line', args.target_spacing)

        # Generate arc data
        par_arc = build_config(args)
        par_arc.cutSectionType = 'arc'
        par_arc.projectName += '_Arc'
        par_arc.datasetName += '_Arc'
        csv_arc, elapsed_arc, count_arc = run_generation(par_arc, 'ARC CUTS')
        save_metadata(par_arc, csv_arc, elapsed_arc, count_arc, 'arc', args.target_spacing)

        print(f"\n{'=' * 70}")
        print(f"  COMBINED GENERATION COMPLETE")
        print(f"  Line data: {csv_line} ({count_line} samples)")
        print(f"  Arc data:  {csv_arc} ({count_arc} samples)")
        print(f"  Total time: {elapsed_line + elapsed_arc:.1f}s")
        print(f"{'=' * 70}")
    else:
        par = build_config(args)
        csv_path, elapsed, count = run_generation(par, args.cut_type.upper() + ' CUTS')
        save_metadata(par, csv_path, elapsed, count, args.cut_type, args.target_spacing)

        if args.inspect:
            inspect_output_csv(csv_path, par.get_num_distances())


def inspect_output_csv(csv_path, num_distances=12):
    """Inspect the generated CSV file."""
    import numpy as np

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    data = np.loadtxt(csv_path, delimiter=',')

    print(f"\n{'=' * 70}")
    print(f"CSV INSPECTION: {csv_path}")
    print(f"{'=' * 70}")
    print(f"  Shape: {data.shape}")
    print(f"  Columns 0-{num_distances-1}: Signed distances ({num_distances} features)")
    print(f"  Columns {data.shape[1]-4}-{data.shape[1]-1}: Quadrature weights")
    print(f"\n  Weight stats (last 4 cols):")
    weights = data[:, -4:]
    for i, label in enumerate(['w1(1)', 'w2(x)', 'w3(y)', 'w4(xy)']):
        print(f"    {label}: mean={weights[:,i].mean():.6f}, "
              f"std={weights[:,i].std():.6f}, "
              f"range=[{weights[:,i].min():.6f}, {weights[:,i].max():.6f}]")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
