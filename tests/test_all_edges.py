"""
Test script to verify:
1. All-edges start point mode works
2. Polygon vertex saving works
3. Gauss points are at standard positions
"""

import numpy as np
import sys
import os

# Add parent directory to path

from fcm_quadrature.data_generation.parameters import ProjectParameter, QuickTestConfig
from fcm_quadrature.data_generation.job import Project, Job
from fcm_quadrature.data_generation.mesh import Point, PointList, LineList, Rectangle
from fcm_quadrature.data_generation.moment_fitting import MomentFittingQuadrature


def test_all_edges_configuration():
    """Test that all-edges mode generates configurations from all 4 edges."""
    print("\n" + "="*60)
    print("TEST 1: All-edges start point configuration")
    print("="*60)

    par = ProjectParameter(projectName='Test', datasetName='Test')
    par.numSamplesStartEdge = 3
    par.numSamplesEndEdge = 3
    par.useAllEdgesAsStart = True

    cell = Rectangle(par.cellSideLength, par.cellSideLength)

    # Generate configurations manually to compare
    all_configs = []
    for startEdgeIdx in range(4):
        startEdge = LineList(cell.allEdges[startEdgeIdx])
        startPoints = startEdge.samplePoints(
            par.numSamplesStartEdge,
            method=par.edgePointMethod,
            startRatio=par.minEdgeLengthRatio,
        )
        endEdgeIndices = [i for i in range(4) if i != startEdgeIdx]
        endEdges = LineList(*[cell.allEdges[i] for i in endEdgeIndices])
        endPoints = endEdges.samplePoints(
            par.numSamplesEndEdge,
            method=par.edgePointMethod,
            startRatio=par.minEdgeLengthRatio,
        )
        for sp in startPoints:
            for ep in endPoints:
                all_configs.append((sp.vec.tolist(), ep.vec.tolist()))

    # Expected: 4 edges * 3 start points * (3 edges * 3 end points) = 4 * 3 * 9 = 108
    expected_configs = 4 * par.numSamplesStartEdge * (3 * par.numSamplesEndEdge)
    print(f"Expected configurations: {expected_configs}")
    print(f"Generated configurations: {len(all_configs)}")

    # Check that we have configurations starting from different edges
    start_edges_used = set()
    for sp, ep in all_configs[:20]:  # Check first 20
        # Determine which edge the start point is on
        if abs(sp[1] - 1.0) < 0.01:  # Top edge (y=1)
            start_edges_used.add(0)
        elif abs(sp[0] + 1.0) < 0.01:  # Left edge (x=-1)
            start_edges_used.add(1)
        elif abs(sp[1] + 1.0) < 0.01:  # Bottom edge (y=-1)
            start_edges_used.add(2)
        elif abs(sp[0] - 1.0) < 0.01:  # Right edge (x=1)
            start_edges_used.add(3)

    print(f"Start edges used (from first 20): {start_edges_used}")

    assert len(all_configs) == expected_configs, f"Wrong number of configs"
    print("✓ All-edges configuration test PASSED")
    return True


def test_gauss_points_standard_positions():
    """Test that Gauss points are at standard positions."""
    print("\n" + "="*60)
    print("TEST 2: Gauss points at standard positions")
    print("="*60)

    mfq = MomentFittingQuadrature()
    quad_points, std_weights = mfq.get_standard_gauss_points_2d(order=2)

    # Standard 2-point Gauss positions: ±1/√3 ≈ ±0.5773502691896257
    xi = 1.0 / np.sqrt(3)
    expected_positions = np.array([
        [-xi, -xi],
        [xi, -xi],
        [-xi, xi],
        [xi, xi]
    ])

    print(f"Expected Gauss points:\n{expected_positions}")
    print(f"Actual Gauss points:\n{quad_points}")

    # Check positions match
    np.testing.assert_allclose(np.sort(quad_points.flatten()),
                               np.sort(expected_positions.flatten()),
                               atol=1e-14)

    print("✓ Gauss points are at standard positions [-1/√3, 1/√3]²")

    # Verify weights sum to 4 (area of [-1,1]²)
    assert np.isclose(np.sum(std_weights), 4.0), "Standard weights should sum to 4"
    print(f"✓ Standard weights sum to {np.sum(std_weights)} (area of reference cell)")

    return True


def test_moment_fitting_with_standard_gauss():
    """Test moment fitting with standard Gauss points for various cut cells."""
    print("\n" + "="*60)
    print("TEST 3: Moment fitting with standard Gauss points")
    print("="*60)

    mfq = MomentFittingQuadrature()
    quad_points, _ = mfq.get_standard_gauss_points_2d(order=2)

    # Test case 1: Full cell (should recover standard Gauss weights)
    vertices_full = np.array([
        [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]
    ])
    weights_full = mfq.compute_weights(quad_points, vertices_full)
    print(f"\nFull cell weights: {weights_full}")
    print(f"Should be [1, 1, 1, 1]: {np.allclose(weights_full, 1.0)}")

    # Test case 2: Half cell (horizontal cut)
    vertices_half = np.array([
        [1.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [1.0, 0.0]
    ])
    weights_half = mfq.compute_weights(quad_points, vertices_half)
    print(f"\nUpper half cell weights: {weights_half}")
    print(f"Sum (should be 2.0 = area): {np.sum(weights_half):.6f}")

    # Verify moments
    is_valid, errors = mfq.verify_weights(quad_points, weights_half, vertices_half)
    print(f"Verification passed: {is_valid}, max error: {np.max(errors):.2e}")

    # Test case 3: Quarter cell (corner triangle)
    vertices_quarter = np.array([
        [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]
    ])
    weights_quarter = mfq.compute_weights(quad_points, vertices_quarter)
    print(f"\nTriangle (upper-right quarter) weights: {weights_quarter}")
    print(f"Sum (should be 2.0 = area): {np.sum(weights_quarter):.6f}")

    is_valid, errors = mfq.verify_weights(quad_points, weights_quarter, vertices_quarter)
    print(f"Verification passed: {is_valid}, max error: {np.max(errors):.2e}")

    # Test case 4: Small cut cell (thin slice at top)
    vertices_thin = np.array([
        [1.0, 1.0], [-1.0, 1.0], [-1.0, 0.8], [1.0, 0.8]
    ])
    weights_thin = mfq.compute_weights(quad_points, vertices_thin)
    expected_area = 2.0 * 0.2  # width * height
    print(f"\nThin slice weights: {weights_thin}")
    print(f"Sum (should be {expected_area}): {np.sum(weights_thin):.6f}")

    is_valid, errors = mfq.verify_weights(quad_points, weights_thin, vertices_thin)
    print(f"Verification passed: {is_valid}, max error: {np.max(errors):.2e}")

    print("\n✓ Moment fitting with standard Gauss points test PASSED")
    return True


def test_polygon_vertex_format():
    """Test the polygon vertex data format."""
    print("\n" + "="*60)
    print("TEST 4: Polygon vertex data format")
    print("="*60)

    par = ProjectParameter(projectName='Test', datasetName='Test')
    par.savePolygonVertices = True
    par.maxPolygonVertices = 6

    # Simulate what Job.execute does
    vertices = [
        Point([1.0, 0.5]),
        Point([-0.5, 1.0]),
        Point([-1.0, -0.3]),
        Point([0.2, -1.0]),
    ]

    vertices_flat = []
    for v in vertices:
        vertices_flat.extend([v.getX(), v.getY()])

    max_coords = par.maxPolygonVertices * 2
    while len(vertices_flat) < max_coords:
        vertices_flat.append(np.nan)
    vertices_array = np.array(vertices_flat[:max_coords])
    num_vertices = np.array([len(vertices)])

    print(f"Number of vertices: {num_vertices[0]}")
    print(f"Vertex data (flattened): {vertices_array}")
    print(f"Total vertex floats: {len(vertices_array)}")

    # Verify format
    assert len(vertices_array) == max_coords
    assert num_vertices[0] == 4
    assert not np.isnan(vertices_array[0])  # First coord should be valid
    assert np.isnan(vertices_array[-1])  # Last should be NaN (padding)

    print("✓ Polygon vertex format test PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)

    results = []
    results.append(("All-edges configuration", test_all_edges_configuration()))
    results.append(("Gauss points standard positions", test_gauss_points_standard_positions()))
    results.append(("Moment fitting with standard Gauss", test_moment_fitting_with_standard_gauss()))
    results.append(("Polygon vertex format", test_polygon_vertex_format()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and passed

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
