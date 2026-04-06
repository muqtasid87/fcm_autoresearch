"""
Quick integration test for the moment fitting pipeline with new features.
"""

import os
import sys
import shutil
import numpy as np


from fcm_quadrature.data_generation.parameters import ProjectParameter
from fcm_quadrature.data_generation.job import Project


def run_integration_test():
    """Run a small integration test with all new features."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: New Features")
    print("="*60)

    # Clean up any existing test files
    for f in ['log.txt', 'IntegTest.Project']:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists('Data/IntegTest'):
        shutil.rmtree('Data/IntegTest')

    # Test 1: All-edges mode without polygon vertices
    print("\n--- Test 1: All-edges mode, no polygon vertices ---")
    par1 = ProjectParameter(projectName='IntegTest', datasetName='AllEdges')
    par1.numSamplesStartEdge = 3
    par1.numSamplesEndEdge = 3
    par1.useAllEdgesAsStart = True
    par1.savePolygonVertices = False
    par1.numWorkers = 2
    par1.subListLength = 4

    try:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    project1 = Project(par1)
    project1.parallelExecute()

    # Check output
    csv_path1 = 'Data/IntegTest/AllEdges.csv'
    if os.path.exists(csv_path1):
        data1 = np.loadtxt(csv_path1, delimiter=',')
        print(f"Generated {data1.shape[0]} samples")
        print(f"Columns: {data1.shape[1]} (expected 16: 12 distances + 4 weights)")
        assert data1.shape[1] == 16, f"Expected 16 columns, got {data1.shape[1]}"
        print("✓ Test 1 PASSED")
    else:
        print("✗ Test 1 FAILED - no CSV generated")
        return False

    # Clean up
    os.remove('log.txt')
    if os.path.exists('IntegTest.Project'):
        os.remove('IntegTest.Project')

    # Test 2: Standard mode with polygon vertices
    print("\n--- Test 2: Standard mode, with polygon vertices ---")
    par2 = ProjectParameter(projectName='IntegTest', datasetName='WithPolygon')
    par2.numSamplesStartEdge = 3
    par2.numSamplesEndEdge = 3
    par2.useAllEdgesAsStart = False
    par2.savePolygonVertices = True
    par2.maxPolygonVertices = 6
    par2.numWorkers = 2
    par2.subListLength = 4
    par2.appendDataset = False

    project2 = Project(par2)
    project2.parallelExecute()

    # Check output
    csv_path2 = 'Data/IntegTest/WithPolygon.csv'
    if os.path.exists(csv_path2):
        data2 = np.loadtxt(csv_path2, delimiter=',')
        print(f"Generated {data2.shape[0]} samples")
        # Expected: 12 distances + 1 num_vertices + 12 vertex coords + 4 weights = 29
        expected_cols = 12 + 1 + (par2.maxPolygonVertices * 2) + 4
        print(f"Columns: {data2.shape[1]} (expected {expected_cols})")
        assert data2.shape[1] == expected_cols, f"Expected {expected_cols} columns, got {data2.shape[1]}"

        # Check that num_vertices column makes sense
        num_verts_col = data2[:, 12]
        print(f"Num vertices range: {num_verts_col.min():.0f} to {num_verts_col.max():.0f}")
        assert all(num_verts_col >= 3), "All polygons should have at least 3 vertices"
        assert all(num_verts_col <= 6), "All polygons should have at most 6 vertices"
        print("✓ Test 2 PASSED")
    else:
        print("✗ Test 2 FAILED - no CSV generated")
        return False

    print("\n" + "="*60)
    print("ALL INTEGRATION TESTS PASSED")
    print("="*60)
    return True


if __name__ == '__main__':
    success = run_integration_test()
    sys.exit(0 if success else 1)
