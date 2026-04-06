"""
Example script for running moment fitting data generation pipeline.

This script demonstrates how to use the modified Job.py to generate
a dataset with quadrature weights instead of eigenvalues.

Usage:
    python run_moment_fitting_pipeline.py --mode test
    python run_moment_fitting_pipeline.py --mode production
    python run_moment_fitting_pipeline.py --mode debug
"""

import os
import time
import argparse
import multiprocessing as mp
from tqdm import tqdm

# Import the modified classes
from fcm_quadrature.data_generation.job import Project
from fcm_quadrature.data_generation.parameters import (
    ProjectParameter, 
    QuickTestConfig, 
    ProductionConfig,
    DebugConfig
)


def run_pipeline(config_mode='test'):
    """
    Run the moment fitting data generation pipeline.
    
    Parameters
    ----------
    config_mode : str
        Configuration mode: 'test', 'production', or 'debug'
    """
    
    mp.set_start_method('spawn', force=True)
    
    # Select configuration
    if config_mode == 'test':
        par = QuickTestConfig(
            projectName='MomentFit_Test',
            datasetName='Test_Dataset'
        )
        print("\n" + "="*70)
        print("RUNNING IN TEST MODE")
        print("Small dataset for quick testing")
        print("="*70)
        
    elif config_mode == 'debug':
        par = DebugConfig(
            projectName='MomentFit_Debug',
            datasetName='Debug_Dataset'
        )
        print("\n" + "="*70)
        print("RUNNING IN DEBUG MODE")
        print("Very small dataset with visualizations enabled")
        print("="*70)
        
    elif config_mode == 'production':
        par = ProductionConfig(
            projectName='MomentFit_Production',
            datasetName='Training_Dataset'
        )
        print("\n" + "="*70)
        print("RUNNING IN PRODUCTION MODE")
        print("Large dataset for training")
        print("="*70)
        
    else:
        raise ValueError(f"Unknown mode: {config_mode}")
    
    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  Project: {par.projectName}")
    print(f"  Dataset: {par.datasetName}")
    print(f"  Start edge samples: {par.numSamplesStartEdge}")
    print(f"  End edge samples: {par.numSamplesEndEdge}")
    print(f"  Workers: {par.numWorkers}")
    print(f"  Debug visualizations: {par.generateDebugVisualizations}")
    print(f"  Weight tolerance: {par.weightVerificationTolerance}")
    print()
    
    # Clean up previous run if exists
    if os.path.exists(par.logName):
        os.remove(par.logName)
    if os.path.exists(par.projectName + '.' + 'Project'):
        os.remove(par.projectName + '.' + 'Project')
    
    # Create project and run
    startTime = time.perf_counter()
    project = Project(par)
    project.parallelExecute()
    finishTime = time.perf_counter()
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print(f"Total time: {finishTime - startTime:.2f}s")
    print(f"Jobs completed: {project.jobCounter}")
    print(f"Jobs failed: {project.errorCounter}")
    print(f"Success rate: {100*project.jobCounter/(project.jobCounter+project.errorCounter):.1f}%")
    print(f"Output file: {project.dataset.csvPath}")
    print("="*70 + "\n")


def inspect_output_csv(csv_path):
    """
    Inspect the generated CSV file to verify format.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    """
    import numpy as np
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    data = np.loadtxt(csv_path, delimiter=',')
    
    print("\n" + "="*70)
    print("CSV FILE INSPECTION")
    print("="*70)
    print(f"File: {csv_path}")
    print(f"Shape: {data.shape}")
    print(f"  Rows (samples): {data.shape[0]}")
    print(f"  Columns: {data.shape[1]}")
    print()
    print("Column structure:")
    print("  Columns 0-11:  Signed distances from cut line to 12 feature points")
    print("  Columns 12-15: Quadrature weights for basis {1, x, y, xy}")
    print()
    print("First 3 samples:")
    print(data[:3])
    print()
    print("Weight statistics (columns 12-15):")
    weights = data[:, 12:16]
    print(f"  Mean: {np.mean(weights, axis=0)}")
    print(f"  Std:  {np.std(weights, axis=0)}")
    print(f"  Min:  {np.min(weights, axis=0)}")
    print(f"  Max:  {np.max(weights, axis=0)}")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run moment fitting data generation pipeline'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='test',
        choices=['test', 'debug', 'production'],
        help='Configuration mode: test (small), debug (tiny+viz), production (large)'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect output CSV after generation'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(args.mode)
    
    # Inspect output if requested
    if args.inspect:
        # Construct CSV path based on mode
        if args.mode == 'test':
            csv_path = 'Data/MomentFit_Test/Test_Dataset.csv'
        elif args.mode == 'debug':
            csv_path = 'Data/MomentFit_Debug/Debug_Dataset.csv'
        elif args.mode == 'production':
            csv_path = 'Data/MomentFit_Production/Training_Dataset.csv'
        
        inspect_output_csv(csv_path)


if __name__ == '__main__':
    main()