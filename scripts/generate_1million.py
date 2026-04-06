"""
Generate 1 million moment fitting data points.

Configuration calculation:
- Original mode: numSamplesStartEdge × (numSamplesEndEdge × 3 edges) = n × 3n = 3n²
- All-edges mode: 4 × numSamplesStartEdge × (numSamplesEndEdge × 3) = 12n²

For 1 million samples:
- Original mode: n = sqrt(1,000,000 / 3) ≈ 577 → 577 × 1731 = 998,787 samples
- All-edges mode: n = sqrt(1,000,000 / 12) ≈ 289 → 4 × 289 × 867 = 1,001,412 samples

We'll use all-edges mode with n=289 for better coverage of cut configurations.
"""

import os
import sys
import time
import multiprocessing as mp


from fcm_quadrature.data_generation.parameters import ProjectParameter, MillionSamplesConfig
from fcm_quadrature.data_generation.job import Project


def calculate_total_samples(par):
    """Calculate total number of samples that will be generated."""
    if par.useAllEdgesAsStart:
        # 4 start edges × n_start × (3 end edges × n_end)
        return 4 * par.numSamplesStartEdge * (3 * par.numSamplesEndEdge)
    else:
        # 1 start edge × n_start × (3 end edges × n_end)
        return par.numSamplesStartEdge * (3 * par.numSamplesEndEdge)


def main():
    """Run the 1 million sample generation."""

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Create configuration
    par = MillionSamplesConfig()

    # Calculate expected samples
    expected_samples = calculate_total_samples(par)

    print("\n" + "="*70)
    print("MOMENT FITTING DATA GENERATION: 1 MILLION SAMPLES")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Project name: {par.projectName}")
    print(f"  Dataset name: {par.datasetName}")
    print(f"  Samples per start edge: {par.numSamplesStartEdge}")
    print(f"  Samples per end edge: {par.numSamplesEndEdge}")
    print(f"  All-edges mode: {par.useAllEdgesAsStart}")
    print(f"  Save polygon vertices: {par.savePolygonVertices}")
    print(f"  Number of workers: {par.numWorkers}")
    print(f"  Batch size: {par.subListLength}")
    print(f"\nExpected total samples: {expected_samples:,}")
    print(f"Output file: Data/{par.projectName}/{par.datasetName}.csv")
    print("="*70)

    # Skip confirmation for automated runs
    # print("\nThis will generate approximately 1 million data points.")
    # response = input("\nProceed? (y/n): ").strip().lower()
    # if response != 'y':
    #     print("Aborted.")
    #     return

    # Clean up previous run files
    if os.path.exists(par.logName):
        os.remove(par.logName)
    if os.path.exists(par.projectName + '.Project'):
        os.remove(par.projectName + '.Project')

    # Run generation
    print("\nStarting data generation...")
    start_time = time.perf_counter()

    project = Project(par)
    project.parallelExecute()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Print summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    print(f"Jobs completed: {project.jobCounter:,}")
    print(f"Jobs failed: {project.errorCounter:,}")
    print(f"Success rate: {100*project.jobCounter/(project.jobCounter+project.errorCounter):.2f}%")
    print(f"Speed: {project.jobCounter/elapsed:.1f} samples/second")
    print(f"\nOutput saved to: {project.dataset.csvPath}")

    # Check file size
    csv_path = project.dataset.csvPath
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.1f} MB")
    print("="*70)


if __name__ == '__main__':
    main()
