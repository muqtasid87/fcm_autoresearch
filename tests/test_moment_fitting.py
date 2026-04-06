"""
Unit tests for moment fitting quadrature weight computation.

Tests cover:
1. Green's theorem moment computations (exact for polynomials)
2. Quadrature weight computation
3. Verification that weights correctly integrate basis functions
4. Edge cases (small/large area fractions, thin slices)
"""

import numpy as np
import pytest
from fcm_quadrature.data_generation.moment_fitting import MomentFittingQuadrature, fit_quadrature_weights


class TestGreenTheoremMoments:
    """Test the Green's theorem moment computations."""

    def setup_method(self):
        """Setup for each test."""
        self.mfq = MomentFittingQuadrature()

    def test_unit_square_moments(self):
        """Test moments for unit square [-1, 1]^2."""
        # Square from -1 to 1 (area = 4)
        vertices = np.array([
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Expected moments:
        # ∫∫ 1 dA = 4 (area)
        # ∫∫ x dA = 0 (symmetric about y-axis)
        # ∫∫ y dA = 0 (symmetric about x-axis)
        # ∫∫ xy dA = 0 (product of symmetric functions)
        expected = np.array([4.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(moments, expected, atol=1e-14)

    def test_unit_square_shifted_moments(self):
        """Test moments for shifted unit square [0, 2]^2."""
        vertices = np.array([
            [2.0, 2.0],
            [0.0, 2.0],
            [0.0, 0.0],
            [2.0, 0.0]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Expected moments for [0,2]^2:
        # ∫∫ 1 dA = 4
        # ∫∫ x dA = ∫_0^2 ∫_0^2 x dy dx = 2 * ∫_0^2 x dx = 2 * 2 = 4
        # ∫∫ y dA = 4 (by symmetry)
        # ∫∫ xy dA = ∫_0^2 ∫_0^2 xy dy dx = (∫_0^2 x dx)(∫_0^2 y dy) = 2 * 2 = 4
        expected = np.array([4.0, 4.0, 4.0, 4.0])

        np.testing.assert_allclose(moments, expected, atol=1e-14)

    def test_right_triangle_moments(self):
        """Test moments for right triangle with vertices at (0,0), (1,0), (0,1)."""
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Expected moments for right triangle:
        # Area = 1/2
        # ∫∫ x dA = ∫_0^1 ∫_0^{1-x} x dy dx = ∫_0^1 x(1-x) dx = 1/2 - 1/3 = 1/6
        # ∫∫ y dA = 1/6 (by symmetry of the triangle about y=x)
        # ∫∫ xy dA = ∫_0^1 ∫_0^{1-x} xy dy dx = ∫_0^1 x * (1-x)^2/2 dx
        #          = (1/2) ∫_0^1 (x - 2x^2 + x^3) dx = (1/2)(1/2 - 2/3 + 1/4) = 1/24
        expected = np.array([0.5, 1/6, 1/6, 1/24])

        np.testing.assert_allclose(moments, expected, atol=1e-14)

    def test_rectangle_moments(self):
        """Test moments for rectangle [a,b] x [c,d]."""
        a, b, c, d = -0.5, 1.5, -0.3, 0.7
        vertices = np.array([
            [b, d],
            [a, d],
            [a, c],
            [b, c]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Expected:
        # Area = (b-a)(d-c) = 2 * 1 = 2
        # ∫∫ x dA = (d-c) * (b^2 - a^2)/2 = 1 * (2.25 - 0.25)/2 = 1
        # ∫∫ y dA = (b-a) * (d^2 - c^2)/2 = 2 * (0.49 - 0.09)/2 = 0.4
        # ∫∫ xy dA = [(b^2-a^2)/2] * [(d^2-c^2)/2] = 1 * 0.2 = 0.2
        area = (b - a) * (d - c)
        int_x = (d - c) * (b**2 - a**2) / 2
        int_y = (b - a) * (d**2 - c**2) / 2
        int_xy = ((b**2 - a**2) / 2) * ((d**2 - c**2) / 2)
        expected = np.array([area, int_x, int_y, int_xy])

        np.testing.assert_allclose(moments, expected, atol=1e-14)

    def test_counterclockwise_invariance(self):
        """Test that result is same regardless of vertex order (CCW or CW)."""
        vertices_ccw = np.array([
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0]
        ])
        vertices_cw = vertices_ccw[::-1]

        moments_ccw = self.mfq.compute_moments(vertices_ccw, method='green')
        moments_cw = self.mfq.compute_moments(vertices_cw, method='green')

        np.testing.assert_allclose(moments_ccw, moments_cw, atol=1e-14)


class TestQuadratureWeights:
    """Test quadrature weight computation."""

    def setup_method(self):
        """Setup for each test."""
        self.mfq = MomentFittingQuadrature()

    def test_unit_square_standard_gauss(self):
        """Test weights for unit square with standard Gauss points."""
        vertices = np.array([
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0]
        ])

        quad_points, std_weights = self.mfq.get_standard_gauss_points_2d(order=2)

        weights = self.mfq.compute_weights(quad_points, vertices)

        # For the standard Gauss points on their native domain,
        # the moment-fitted weights should equal standard Gauss weights
        np.testing.assert_allclose(weights, std_weights, atol=1e-12)

    def test_triangle_weights_sum_to_area(self):
        """Test that weights sum to the area (first moment)."""
        vertices = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.5]
        ])

        # Place 4 quadrature points inside the triangle
        centroid = np.mean(vertices, axis=0)
        quad_points = np.array([
            centroid + [0.1, 0.1],
            centroid + [-0.1, 0.1],
            centroid + [0.1, -0.1],
            centroid + [-0.1, -0.1]
        ])

        weights = self.mfq.compute_weights(quad_points, vertices)

        # Weights should sum to area
        expected_area = 0.5 * abs(2.0 * 1.5)  # base * height / 2
        np.testing.assert_allclose(np.sum(weights), expected_area, atol=1e-12)

    def test_weight_verification_passes(self):
        """Test that verification passes for valid weights."""
        vertices = np.array([
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0]
        ])

        quad_points, _ = self.mfq.get_standard_gauss_points_2d(order=2)
        weights = self.mfq.compute_weights(quad_points, vertices)

        is_valid, errors = self.mfq.verify_weights(quad_points, weights, vertices)

        assert is_valid
        assert np.max(errors) < 1e-12

    def test_pentagon_weights(self):
        """Test weights for a pentagon (5 vertices)."""
        # Regular pentagon centered at origin
        n = 5
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/2
        vertices = np.array([[np.cos(a), np.sin(a)] for a in angles])

        # Use points near center
        quad_points = np.array([
            [0.2, 0.2],
            [-0.2, 0.2],
            [-0.2, -0.2],
            [0.2, -0.2]
        ])

        weights, verification = fit_quadrature_weights(quad_points, vertices, verify=True)

        assert verification['is_valid']
        assert verification['max_error'] < 1e-10


class TestCutCellConfigurations:
    """Test moment fitting for realistic cut cell configurations."""

    def setup_method(self):
        """Setup for each test."""
        self.mfq = MomentFittingQuadrature()

    def test_half_cell_horizontal_cut(self):
        """Test cut cell with horizontal cut through middle."""
        # Upper half of [-1,1]^2
        vertices = np.array([
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Expected: area = 2, ∫x = 0, ∫y = 1, ∫xy = 0
        expected = np.array([2.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(moments, expected, atol=1e-14)

    def test_quarter_cell_corner_cut(self):
        """Test cut cell with corner cut (triangle)."""
        # Triangle in upper-right quadrant
        vertices = np.array([
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Triangle with vertices (1,1), (0,1), (1,0)
        # Area = 0.5
        # ∫∫ x dA = ∫_0^1 ∫_{1-x}^1 x dy dx = ∫_0^1 x*x dx = 1/3
        # ∫∫ y dA = 1/3 (by symmetry)
        # ∫∫ xy dA = ∫_0^1 ∫_{1-x}^1 xy dy dx
        expected_area = 0.5

        np.testing.assert_allclose(moments[0], expected_area, atol=1e-14)

    def test_thin_slice_stability(self):
        """Test numerical stability for thin slices (small area fraction)."""
        # Very thin horizontal slice
        epsilon = 0.01
        vertices = np.array([
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0 - epsilon],
            [1.0, 1.0 - epsilon]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Expected area = 2 * epsilon
        expected_area = 2.0 * epsilon
        np.testing.assert_allclose(moments[0], expected_area, rtol=1e-10)

        # For thin strips, we need points with DIFFERENT y-coordinates
        # to avoid singular Vandermonde matrix (basis includes y and xy)
        y_low = 1.0 - 0.75*epsilon
        y_high = 1.0 - 0.25*epsilon
        quad_points = np.array([
            [0.5, y_low],
            [-0.5, y_low],
            [0.3, y_high],
            [-0.3, y_high]
        ])

        weights = self.mfq.compute_weights(quad_points, vertices)

        # Weights should sum to area
        np.testing.assert_allclose(np.sum(weights), expected_area, rtol=1e-8)

    def test_diagonal_cut(self):
        """Test cut cell with diagonal cut from corner to corner."""
        # Lower-right triangle of [-1,1]^2
        vertices = np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Area = 2 (half of square with area 4)
        expected_area = 2.0
        np.testing.assert_allclose(moments[0], expected_area, atol=1e-14)


class TestNumericalAccuracy:
    """Test machine-precision accuracy of the implementation."""

    def setup_method(self):
        """Setup for each test."""
        self.mfq = MomentFittingQuadrature()

    def test_green_vs_numerical_comparison(self):
        """Compare Green's theorem with numerical integration."""
        vertices = np.array([
            [1.0, 0.5],
            [-0.5, 1.0],
            [-1.0, -0.5],
            [0.5, -1.0]
        ])

        moments_green = self.mfq.compute_moments(vertices, method='green')
        moments_numerical = self.mfq.compute_moments(vertices, method='adaptive')

        # Green's theorem should be more accurate for polynomials,
        # but numerical should be close. Use both rtol and atol since
        # some moments may be near zero (symmetric shapes).
        np.testing.assert_allclose(moments_green, moments_numerical,
                                   rtol=1e-6, atol=1e-12)

    def test_basis_function_integration(self):
        """Verify weights correctly integrate each basis function."""
        vertices = np.array([
            [1.5, 0.8],
            [-0.7, 1.2],
            [-1.0, -0.5],
            [0.3, -0.9]
        ])

        # Get quadrature points (centered in polygon)
        centroid = np.mean(vertices, axis=0)
        scale = 0.3
        xi = 1.0 / np.sqrt(3)
        quad_points = np.array([
            [-xi, -xi],
            [xi, -xi],
            [-xi, xi],
            [xi, xi]
        ]) * scale + centroid

        weights = self.mfq.compute_weights(quad_points, vertices)
        moments_ref = self.mfq.compute_moments(vertices, method='green')

        # Verify each basis function
        basis = [
            lambda x, y: np.ones_like(x),
            lambda x, y: x,
            lambda x, y: y,
            lambda x, y: x * y
        ]

        for j, func in enumerate(basis):
            quad_integral = np.sum(weights * func(quad_points[:, 0], quad_points[:, 1]))
            np.testing.assert_allclose(quad_integral, moments_ref[j], atol=1e-10,
                err_msg=f"Basis function {j} integration failed")

    def test_weight_conditioning(self):
        """Test that weight computation is well-conditioned."""
        # Test with various polygon shapes
        test_cases = [
            # Square
            np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]),
            # Rectangle
            np.array([[2, 0.5], [-2, 0.5], [-2, -0.5], [2, -0.5]]),
            # Triangle
            np.array([[0, 1], [-1, -1], [1, -1]]),
            # Pentagon
            np.array([[np.cos(a), np.sin(a)]
                     for a in np.linspace(0, 2*np.pi, 5, endpoint=False)])
        ]

        for vertices in test_cases:
            centroid = np.mean(vertices, axis=0)
            xi = 1.0 / np.sqrt(3)
            quad_points = np.array([
                [-xi, -xi], [xi, -xi], [-xi, xi], [xi, xi]
            ]) * 0.3 + centroid

            weights, verification = fit_quadrature_weights(
                quad_points, vertices, verify=True
            )

            assert verification['is_valid'], f"Failed for shape with {len(vertices)} vertices"
            assert verification['max_error'] < 1e-10


class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def setup_method(self):
        """Setup for each test."""
        self.mfq = MomentFittingQuadrature()

    def test_very_small_area(self):
        """Test with very small area fraction."""
        # Tiny triangle
        eps = 1e-6
        vertices = np.array([
            [0.0, 0.0],
            [eps, 0.0],
            [0.0, eps]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        expected_area = 0.5 * eps * eps
        np.testing.assert_allclose(moments[0], expected_area, rtol=1e-10)

    def test_collinear_points_detection(self):
        """Test handling when polygon has near-collinear points."""
        # Almost flat triangle
        eps = 1e-8
        vertices = np.array([
            [0.0, 0.0],
            [1.0, eps],
            [2.0, 0.0]
        ])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Area should be very small but positive
        assert moments[0] > 0
        assert moments[0] < 1e-6

    def test_many_vertices(self):
        """Test polygon with many vertices (approximating circle)."""
        n = 100
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        vertices = np.array([[np.cos(a), np.sin(a)] for a in angles])

        moments = self.mfq.compute_moments(vertices, method='green')

        # Should approximate circle: area ≈ π
        np.testing.assert_allclose(moments[0], np.pi, rtol=1e-3)
        # By symmetry, x and y moments should be ~0
        np.testing.assert_allclose(moments[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(moments[2], 0.0, atol=1e-10)


def run_all_tests():
    """Run all tests and print summary."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_all_tests()
