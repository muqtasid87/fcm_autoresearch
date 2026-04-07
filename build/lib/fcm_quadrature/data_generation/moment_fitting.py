"""
Moment Fitting for Quadrature Weight Computation

This module computes quadrature weights for a cut cell using moment fitting
with basis functions {1, x, y, xy}.

The moment fitting problem:
Given quadrature points (x_i, y_i) for i=1,...,n_quad, find weights w_i such that:
    ∑ w_i * φ_j(x_i, y_i) = ∫∫_Ω φ_j(x, y) dA
for each basis function φ_j ∈ {1, x, y, xy}

This gives us a linear system: V^T * w = m
where:
    V[i,j] = φ_j(x_i, y_i) is the Vandermonde-like matrix
    w = [w_1, ..., w_n] are the unknown weights
    m = [m_1, ..., m_4] are the moments (integrals of basis functions)
"""

import numpy as np
from scipy import integrate
from typing import List, Tuple, Callable, Optional
import warnings


class MomentFittingQuadrature:
    """
    Compute quadrature weights using moment fitting for a 2D domain.
    """
    
    def __init__(self, basis_functions: Optional[List[Callable]] = None):
        """
        Initialize the moment fitting quadrature.
        
        Parameters
        ----------
        basis_functions : list of callable, optional
            List of basis functions. Default is {1, x, y, xy}
        """
   
        # Default basis: {1, x, y, xy}
        self.basis_functions = [
            lambda x, y: np.ones_like(x),  # 1
            lambda x, y: x,                 # x
            lambda x, y: y,                 # y
            lambda x, y: x * y              # xy
        ]

        
        self.n_basis = len(self.basis_functions)
    
    def get_standard_gauss_points_2d(self, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get standard 2D Gauss quadrature points on [-1, 1]^2.
        
        Parameters
        ----------
        order : int
            Order of Gauss quadrature (number of points per dimension)
            
        Returns
        -------
        points : np.ndarray, shape (n_points, 2)
            Gauss quadrature points
        std_weights : np.ndarray, shape (n_points,)
            Standard Gauss weights (not the moment-fitted weights)
        """
        # 1D Gauss points and weights
        if order == 2:
            # 2-point Gauss quadrature
            xi_1d = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
            w_1d = np.array([1.0, 1.0])
        else:
            raise ValueError(f"Gauss quadrature order {order} not implemented")
        
        # Tensor product for 2D
        n_1d = len(xi_1d)
        n_points = n_1d ** 2
        points = np.zeros((n_points, 2))
        std_weights = np.zeros(n_points)
        
        idx = 0
        for i in range(n_1d):
            for j in range(n_1d):
                points[idx] = [xi_1d[i], xi_1d[j]]
                std_weights[idx] = w_1d[i] * w_1d[j]
                idx += 1
        
        return points, std_weights
    
    def compute_moments(self, domain_vertices: np.ndarray, 
                       method: str = 'green') -> np.ndarray:
        """
        Compute the moments (integrals) of basis functions over the domain.
        
        Parameters
        ----------
        domain_vertices : np.ndarray, shape (n_vertices, 2)
            Vertices of the polygonal domain in counter-clockwise order
        method : str, optional
            Integration method: 'adaptive', 'simpson', or 'green'
            
        Returns
        -------
        moments : np.ndarray, shape (n_basis,)
            Moments m_j = ∫∫_Ω φ_j(x, y) dA
        """
        if method == 'green':
            return self._compute_moments_green_theorem(domain_vertices)
        else:
            return self._compute_moments_numerical(domain_vertices, method)
    
    def _compute_moments_green_theorem(self, vertices: np.ndarray) -> np.ndarray:
        """
        Compute moments using Green's theorem for exact integration of polynomials.
        
        For a polygonal domain with vertices (x_i, y_i), Green's theorem gives:
        ∫∫_Ω f(x,y) dA = ∮_∂Ω F(x,y)·dr
        
        where F is chosen such that curl(F) = f (i.e., ∂F_y/∂x - ∂F_x/∂y = f)
        
        For polygon integration, we use the shoelace formula variants.
        """
        n_vertices = len(vertices)
        moments = np.zeros(self.n_basis)
        
        # Ensure vertices are in counter-clockwise order by checking signed area
        # If negative, reverse the order
        area_test = 0.0
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            area_test += vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
        
        if area_test < 0:
            vertices = vertices[::-1]
        
        # Close the polygon for easier iteration
        vertices_closed = np.vstack([vertices, vertices[0]])
        
        # Basis: {1, x, y, xy}
        # Using Green's theorem with appropriate vector fields:
        
        # Moment 0: ∫∫ 1 dA (Area)
        # Use shoelace formula: A = (1/2) * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
        moment_1 = 0.0
        for i in range(n_vertices):
            x1, y1 = vertices_closed[i]
            x2, y2 = vertices_closed[i + 1]
            moment_1 += x1 * y2 - x2 * y1
        moments[0] = 0.5 * abs(moment_1)
        
        # Moment 1: ∫∫ x dA
        # Using Green's theorem: ∫∫ x dA = ∮ (1/2 * x^2) dy
        moment_x = 0.0
        for i in range(n_vertices):
            x1, y1 = vertices_closed[i]
            x2, y2 = vertices_closed[i + 1]
            # Line integral over edge from (x1,y1) to (x2,y2)
            # ∫ x^2 dy along edge = ∫_0^1 ((1-t)x1 + tx2)^2 * (y2-y1) dt
            # = (y2-y1) * ∫_0^1 (x1^2(1-t)^2 + 2x1*x2*t(1-t) + x2^2*t^2) dt
            # = (y2-y1) * (x1^2/3 + x1*x2/3 + x2^2/3)
            moment_x += (y2 - y1) * (x1**2 + x1*x2 + x2**2) / 3.0
        moments[1] = 0.5 * moment_x
        
        # Moment 2: ∫∫ y dA
        # Using Green's theorem: ∫∫ y dA = -∮ (1/2 * y^2) dx
        moment_y = 0.0
        for i in range(n_vertices):
            x1, y1 = vertices_closed[i]
            x2, y2 = vertices_closed[i + 1]
            # Line integral over edge
            # -∫ y^2 dx along edge = -∫_0^1 ((1-t)y1 + ty2)^2 * (x2-x1) dt
            # = -(x2-x1) * (y1^2/3 + y1*y2/3 + y2^2/3)
            moment_y += -(x2 - x1) * (y1**2 + y1*y2 + y2**2) / 3.0
        moments[2] = 0.5 * moment_y
        
        # Moment 3: ∫∫ xy dA
        # Using Green's theorem with F = (0, x^2*y/2)
        # Then ∂F_y/∂x = x*y and ∂F_x/∂y = 0, so curl(F) = xy
        # ∮ F·dr = ∮ (x^2*y/2) dy
        moment_xy = 0.0
        for i in range(n_vertices):
            x1, y1 = vertices_closed[i]
            x2, y2 = vertices_closed[i + 1]
            # Line integral: ∫ x^2*y dy along edge from (x1,y1) to (x2,y2)
            # Parametrize: x(t) = x1 + t(x2-x1), y(t) = y1 + t(y2-y1), t ∈ [0,1]
            # dy = (y2-y1)dt
            # ∫_0^1 [x1 + t(x2-x1)]^2 * [y1 + t(y2-y1)] * (y2-y1) dt
            dx = x2 - x1
            dy = y2 - y1
            # Expand: (x1 + t*dx)^2 * (y1 + t*dy) * dy
            # = (x1^2 + 2*x1*dx*t + dx^2*t^2) * (y1 + dy*t) * dy
            # = dy * [x1^2*y1 + x1^2*dy*t + 2*x1*dx*y1*t + 2*x1*dx*dy*t^2 + dx^2*y1*t^2 + dx^2*dy*t^3]
            # Integrate from 0 to 1:
            # = dy * [x1^2*y1 + x1^2*dy/2 + 2*x1*dx*y1/2 + 2*x1*dx*dy/3 + dx^2*y1/3 + dx^2*dy/4]
            # Simplify by substituting back x2 = x1+dx, y2 = y1+dy
            term1 = x1**2 * y1
            term2 = x1**2 * dy / 2.0
            term3 = 2.0 * x1 * dx * y1 / 2.0  # = x1*dx*y1
            term4 = 2.0 * x1 * dx * dy / 3.0
            term5 = dx**2 * y1 / 3.0
            term6 = dx**2 * dy / 4.0
            
            moment_xy += dy * (term1 + term2 + term3 + term4 + term5 + term6)
        moments[3] = 0.5 * moment_xy
        
        return moments
    
    def _compute_moments_numerical(self, vertices: np.ndarray, 
                                  method: str = 'adaptive') -> np.ndarray:
        """
        Compute moments using numerical integration over triangulated domain.
        """
        moments = np.zeros(self.n_basis)
        
        # Triangulate the polygon (simple fan triangulation from first vertex)
        n_vertices = len(vertices)
        for i in range(1, n_vertices - 1):
            triangle = np.array([vertices[0], vertices[i], vertices[i + 1]])
            
            # Integrate over this triangle
            for j, basis_func in enumerate(self.basis_functions):
                if method == 'adaptive':
                    moment, _ = self._integrate_triangle_adaptive(triangle, basis_func)
                moments[j] += moment
        
        return moments
    
    def _integrate_triangle_adaptive(self, triangle: np.ndarray, 
                                    func: Callable) -> Tuple[float, float]:
        """
        Integrate function over a triangle using adaptive quadrature.
        """
        v0, v1, v2 = triangle
        
        def integrand(s, t):
            """Map reference triangle [0,1]x[0,1-s] to physical triangle"""
            if s + t > 1:
                return 0.0
            x = v0[0] + s * (v1[0] - v0[0]) + t * (v2[0] - v0[0])
            y = v0[1] + s * (v1[1] - v0[1]) + t * (v2[1] - v0[1])
            # Jacobian determinant
            jac = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - 
                     (v1[1] - v0[1]) * (v2[0] - v0[0]))
            return func(x, y) * jac
        
        result, error = integrate.dblquad(
            integrand, 0, 1, 0, lambda s: 1 - s,
            epsabs=1e-10, epsrel=1e-10
        )
        return result, error
    
    
    def compute_weights(self, quad_points: np.ndarray, 
                       domain_vertices: np.ndarray,
                       moment_method: str = 'green',
                       regularization: float = 1e-12) -> np.ndarray:
        """
        Compute quadrature weights using moment fitting.
        
        Parameters
        ----------
        quad_points : np.ndarray, shape (n_points, 2)
            Quadrature points (x_i, y_i)
        domain_vertices : np.ndarray, shape (n_vertices, 2)
            Vertices of the domain (cut cell)
        moment_method : str, optional
            Method to compute moments: 'green', 'adaptive', or 'simpson'
        regularization : float, optional
            Regularization parameter for least squares (if overdetermined)
            
        Returns
        -------
        weights : np.ndarray, shape (n_points,)
            Quadrature weights w_i
        """
        n_points = len(quad_points)
        
        # Build Vandermonde matrix V[i,j] = φ_j(x_i, y_i)
        V = np.zeros((n_points, self.n_basis))
        for i, (x, y) in enumerate(quad_points):
            for j, basis_func in enumerate(self.basis_functions):
                V[i, j] = basis_func(x, y)
        
        # Compute moments
        moments = self.compute_moments(domain_vertices, method=moment_method)
        
        # Solve V * w = moments for weights w
        # where V[i,j] = φ_j(x_i, y_i) and moments[j] = ∫∫ φ_j dA
        
        if n_points == self.n_basis:
            # Square system - direct solve
            # V is n_points x n_basis, w is n_points, moments is n_basis
            # We need: sum_i w_i * V[i,j] = moments[j]
            # This is V^T * w = moments
            weights = np.linalg.solve(V.T, moments)
        elif n_points > self.n_basis:
            # Overdetermined system: V^T * w = moments
            # Use least squares: minimize ||V^T * w - moments||^2
            # Solution: w = (V * V^T)^{-1} * V * moments
            VVT = V @ V.T + regularization * np.eye(n_points)
            weights = np.linalg.solve(VVT, V @ moments)
        else:
            # Underdetermined system
            # Use minimum norm solution via pseudo-inverse
            weights = np.linalg.lstsq(V.T, moments, rcond=None)[0]
        
        return weights
    
    def verify_weights(self, quad_points: np.ndarray, weights: np.ndarray,
                      domain_vertices: np.ndarray, 
                      moment_method: str = 'green') -> Tuple[bool, np.ndarray]:
        """
        Verify that the computed weights satisfy the moment equations.
        
        Parameters
        ----------
        quad_points : np.ndarray
            Quadrature points
        weights : np.ndarray
            Quadrature weights
        domain_vertices : np.ndarray
            Domain vertices
        moment_method : str
            Method used to compute reference moments
            
        Returns
        -------
        is_valid : bool
            True if weights satisfy moment equations within tolerance
        errors : np.ndarray
            Absolute errors for each moment equation
        """
        # Compute reference moments
        moments_ref = self.compute_moments(domain_vertices, method=moment_method)
        
        # Compute moments using quadrature
        moments_quad = np.zeros(self.n_basis)
        for i, (x, y) in enumerate(quad_points):
            for j, basis_func in enumerate(self.basis_functions):
                moments_quad[j] += weights[i] * basis_func(x, y)
        
        # Compute errors
        errors = np.abs(moments_ref - moments_quad)
        # relative_errors = errors / (np.abs(moments_ref) + 1e-16)
        
        # # Check if all relative errors are small
        # tolerance = 1e-8
        # is_valid = np.all(relative_errors < tolerance)

        # If the reference moment is effectively zero, use absolute error checks instead
        absolute_tolerance = 1e-10
        relative_tolerance = 1e-8

        is_valid = True
        for i in range(len(moments_ref)):
            if abs(moments_ref[i]) < 1e-12:
                # For moments near zero, check absolute error
                if errors[i] > absolute_tolerance:
                    is_valid = False
                    break
            else:
                # For non-zero moments, check relative error
                rel_err = errors[i] / abs(moments_ref[i])
                if rel_err > relative_tolerance:
                    is_valid = False
                    break
        
        return is_valid, errors


def fit_quadrature_weights(quad_points: np.ndarray, 
                          domain_vertices: np.ndarray,
                          basis_order: int = 1,
                          moment_method: str = 'green',
                          verify: bool = True) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Convenience function to compute quadrature weights using moment fitting.
    
    Parameters
    ----------
    quad_points : np.ndarray, shape (n_points, 2)
        Quadrature points
    domain_vertices : np.ndarray, shape (n_vertices, 2)
        Vertices of domain in counter-clockwise order
    basis_order : int, optional
        Order of polynomial basis (1 for {1,x,y,xy}, higher orders not yet implemented)
    moment_method : str, optional
        Method to compute moments: 'green' (recommended), 'adaptive', or 'simpson'
    verify : bool, optional
        If True, verify the computed weights
        
    Returns
    -------
    weights : np.ndarray, shape (n_points,)
        Quadrature weights
    verification : dict or None
        Verification results if verify=True, else None
    """
    if basis_order != 1:
        raise NotImplementedError("Only basis_order=1 ({1,x,y,xy}) is currently implemented")
    
    # Create moment fitting object
    mfq = MomentFittingQuadrature()
    
    # Compute weights
    weights = mfq.compute_weights(quad_points, domain_vertices, 
                                  moment_method=moment_method)
    
    # Verify if requested
    verification = None
    if verify:
        is_valid, errors = mfq.verify_weights(quad_points, weights, 
                                              domain_vertices, moment_method)
        verification = {
            'is_valid': is_valid,
            'moment_errors': errors,
            'max_error': np.max(errors)
        }
        
        if not is_valid:
            warnings.warn(f"Moment fitting verification failed. Max error: {np.max(errors)}")
    
    return weights, verification


if __name__ == "__main__":
    # Example usage
    print("Moment Fitting Quadrature - Example")
    print("=" * 50)
    
    # Define a simple square domain
    vertices = np.array([
        [1.0, -2*2/6],
        [2*2/6,-2*2/6],
        [2*2/6, -1.0],
        # [-1.0, -1.0],
        [1.0, -1.0]
    ])
    
    vertices = np.array([
        [2.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.5],
        [-1.0, -2.0]
    ]) 
    # Get standard 2x2 Gauss points
    mfq = MomentFittingQuadrature()
    quad_points, _ = mfq.get_standard_gauss_points_2d(order=2)
    
    print(f"\nDomain vertices:\n{vertices}")
    print(f"\nQuadrature points:\n{quad_points}")
    
    # Compute weights
    weights, verification = fit_quadrature_weights(quad_points, vertices, verify=True)
    
    print(f"\nComputed weights:\n{weights}")
    print(f"\nVerification: {verification['is_valid']}")
    print(f"Max moment error: {verification['max_error']:.2e}")