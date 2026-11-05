"""Analytic limits and defining equations for the Gaussian DMF solvers."""
from typing import cast

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from lowrank.meanfield import (
    GaussianGeometry, GaussianRankOne, fixed_point_residual,
    gaussian_expectation, phase_boundaries, solve_chaotic, solve_fixed_points,
)


def test_gaussian_quadrature_matches_polynomial_moments():
    assert gaussian_expectation(lambda x: x**2, 2, 3) == pytest.approx(7)
    assert gaussian_expectation(lambda x: x**4, 0, 2) == pytest.approx(12)
    with pytest.raises(ValueError, match="nonnegative"):
        gaussian_expectation(np.tanh, 0, -1)


def test_origin_has_analytic_stability_thresholds():
    p = GaussianRankOne(.5, mean_m=1.5, mean_n=1.5)
    points = solve_fixed_points(p)
    assert len(points) == 3
    origin = points[1]
    assert origin.mean == origin.variance == 0
    assert origin.bulk_radius == .5
    assert origin.outlier == pytest.approx(2.25)
    assert not origin.stable
    assert points[0].stable and points[-1].stable
    assert points[0].mean == pytest.approx(-points[-1].mean)


def test_homogeneous_disorder_free_limit():
    p = GaussianRankOne(0, mean_m=1.5, mean_n=1.5, std_m=0)
    point = solve_fixed_points(p)[-1]
    mean = cast(float, brentq(lambda x: x - 2.25 * np.tanh(x), .01, 3))
    assert point.mean == pytest.approx(mean, abs=1e-8)
    assert point.variance < 1e-8
    assert point.outlier == pytest.approx(2.25 * (1 - np.tanh(mean)**2), abs=1e-8)


def test_input_alignment_generates_structured_response():
    p = GaussianRankOne(.8, var_input=2, cov_n_input=1)
    point, = solve_fixed_points(p)
    gain = gaussian_expectation(lambda x: 1 - np.tanh(x)**2, 0, point.variance)
    assert point.kappa[0] == pytest.approx(gain)
    assert point.mean == 0
    assert point.stable
    assert np.max(np.abs(fixed_point_residual(p, point.kappa, point.variance))) < 1e-8
    perpendicular, = solve_fixed_points(GaussianRankOne(.8, var_input=2))
    assert abs(perpendicular.kappa[0]) < 1e-8


def test_gaussian_covariance_validation():
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianRankOne(.5, cov_mn=2).geometry()
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianRankOne(.5, var_input=0, cov_n_input=1).geometry()


def test_rank_one_outlier_at_zero_disorder_matches_direct_gain_integral():
    p = GaussianRankOne(0, mean_m=1.3, mean_n=1.3, cov_mn=.3)
    point = solve_fixed_points(p)[-1]
    # E[m n phi'(m kappa)] is the only nonzero eigenvalue at g=0.
    direct = gaussian_expectation(
        lambda m: m * (p.mean_n + p.cov_mn * (m - p.mean_m)) *
        (1 - np.tanh(m * point.kappa[0])**2), p.mean_m, 1)
    assert point.outlier == pytest.approx(direct, abs=1e-8)


def test_general_rank_two_homogeneous_limit():
    p = GaussianGeometry(0, np.array([1., 2.]), np.array([.4, .8]),
                         np.zeros((2, 2)), np.zeros((2, 2)))
    point = solve_fixed_points(p, seeds=[(np.array([.4, .8]), 0)])[-1]
    mean = cast(float, brentq(lambda x: x - 2 * np.tanh(x), .01, 3))
    assert point.mean == pytest.approx(mean, abs=1e-8)
    np.testing.assert_allclose(point.kappa, p.mean_n * np.tanh(mean), atol=1e-8)
    assert point.outlier == pytest.approx(2 * (1 - np.tanh(mean)**2), abs=1e-8)


def test_chaotic_solution_satisfies_independently_integrated_energy():
    state = solve_chaotic(GaussianRankOne(1.5), quadrature_order=100)
    assert state.long_lag_variance < 1e-8
    assert state.temporal_variance > 0
    def expectation(power):
        return quad(lambda z: (np.logaddexp(np.sqrt(state.variance)*z,
                                                   -np.sqrt(state.variance)*z) - np.log(2))**power *
                    np.exp(-z*z/2)/np.sqrt(2*np.pi), -10, 10, epsabs=1e-10)[0]
    assert state.variance**2 / 2 == pytest.approx(1.5**2 * (expectation(2)-expectation(1)**2), abs=1e-8)
    with pytest.raises(ValueError, match="no nonstatic"):
        solve_chaotic(GaussianRankOne(.8))


def test_phase_boundaries_order_and_onset():
    boundaries = phase_boundaries(np.array([.5, 1., 1.2, 2.]))
    np.testing.assert_allclose(boundaries.structure_onset[:2], 1)
    assert np.isnan(boundaries.static_chaos[0])
    assert boundaries.static_chaos[1] == 1
    assert np.all(boundaries.static_chaos[2:] > boundaries.structure_onset[2:])
