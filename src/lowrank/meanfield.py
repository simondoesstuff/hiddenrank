"""Gaussian dynamical mean field for tanh low-rank networks.

Conventions follow Mastrogiuseppe & Ostojic (2018), Eqs. 40, 44, 71,
77 and 98–100. Variances refer to activations x, not firing rates tanh(x).
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import roots_hermitenorm

Array = NDArray[np.float64]


@lru_cache(maxsize=16)
def gaussian_quadrature(order: int = 200) -> tuple[Array, Array]:
    """Nodes and weights for expectations over a standard normal variable."""
    if order < 4:
        raise ValueError("quadrature order must be at least four")
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights / np.sqrt(2 * np.pi)


def gaussian_expectation(function: Callable[[Array], Array], mean: float,
                         variance: float, order: int = 200) -> float:
    if variance < 0:
        raise ValueError("variance must be nonnegative")
    nodes, weights = gaussian_quadrature(order)
    return float(weights @ function(mean + np.sqrt(variance) * nodes))


def _moments(mean: float, variance: float, order: int) -> Array:
    nodes, weights = gaussian_quadrature(order)
    phi = np.tanh(mean + np.sqrt(max(variance, 0.0)) * nodes)
    first = 1 - phi**2
    second = -2 * phi * first
    third = -2 * first**2 + 4 * phi**2 * first
    return np.asarray([weights @ item for item in
                       (phi, phi**2, first, second, third, first**2, phi * first,
                        phi * second)])


@dataclass(frozen=True)
class GaussianGeometry:
    """Joint Gaussian pattern statistics, using centered covariances.

    cov_nm[j, l] = Cov(n_j, m_l); input is one scalar Gaussian pattern.
    Its strength must already be included in input moments/covariances.
    """
    g: float
    mean_m: Array
    mean_n: Array
    cov_mm: Array
    cov_nm: Array
    mean_input: float = 0.0
    var_input: float = 0.0
    cov_m_input: Array | None = None
    cov_n_input: Array | None = None

    def __post_init__(self) -> None:
        for name in ("mean_m", "mean_n", "cov_mm", "cov_nm"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        rank = self.mean_m.size
        if self.mean_m.shape != (rank,) or self.mean_n.shape != (rank,):
            raise ValueError("pattern means must be rank-vectors")
        if self.cov_mm.shape != (rank, rank) or self.cov_nm.shape != (rank, rank):
            raise ValueError("pattern covariances must be rank by rank")
        for name in ("cov_m_input", "cov_n_input"):
            value = getattr(self, name)
            value = np.zeros(rank) if value is None else np.asarray(value, dtype=float)
            if value.shape != (rank,):
                raise ValueError("input covariances must be rank-vectors")
            object.__setattr__(self, name, value)
        if self.g < 0 or self.var_input < 0:
            raise ValueError("g and input variance must be nonnegative")
        joint = np.block([[self.cov_mm, self.m_input[:, None]],
                          [self.m_input[None, :], np.array([[self.var_input]])]])
        if not np.allclose(joint, joint.T) or np.linalg.eigvalsh(joint).min() < -1e-10:
            raise ValueError("joint covariance of m and input must be positive semidefinite")

    @property
    def rank(self) -> int:
        return self.mean_m.size

    @property
    def m_input(self) -> Array:
        return np.asarray(self.cov_m_input)

    @property
    def n_input(self) -> Array:
        return np.asarray(self.cov_n_input)

    def statistics(self, kappa: Array) -> tuple[float, float, Array]:
        mean = float(self.mean_m @ kappa + self.mean_input)
        quenched = float(kappa @ self.cov_mm @ kappa +
                         2 * self.m_input @ kappa + self.var_input)
        n_cov = self.cov_nm @ kappa + self.n_input
        return mean, max(quenched, 0.0), n_cov


@dataclass(frozen=True)
class GaussianRankOne:
    g: float
    mean_m: float = 0.0
    mean_n: float = 0.0
    std_m: float = 1.0
    std_n: float = 1.0
    cov_mn: float = 0.0
    mean_input: float = 0.0
    var_input: float = 0.0
    cov_m_input: float = 0.0
    cov_n_input: float = 0.0

    def geometry(self) -> GaussianGeometry:
        if self.std_m < 0 or self.std_n < 0:
            raise ValueError("pattern standard deviations must be nonnegative")
        joint = np.array([[self.std_m**2, self.cov_mn, self.cov_m_input],
                          [self.cov_mn, self.std_n**2, self.cov_n_input],
                          [self.cov_m_input, self.cov_n_input, self.var_input]])
        if np.linalg.eigvalsh(joint).min() < -1e-10:
            raise ValueError("joint pattern covariance must be positive semidefinite")
        return GaussianGeometry(self.g, np.array([self.mean_m]), np.array([self.mean_n]),
                                np.array([[self.std_m**2]]), np.array([[self.cov_mn]]),
                                self.mean_input, self.var_input,
                                np.array([self.cov_m_input]), np.array([self.cov_n_input]))


def _geometry(parameters: GaussianGeometry | GaussianRankOne) -> GaussianGeometry:
    return parameters.geometry() if isinstance(parameters, GaussianRankOne) else parameters


@dataclass(frozen=True)
class FixedPoint:
    kappa: Array
    mean: float
    variance: float
    bulk_radius: float
    meanfield_eigenvalues: NDArray[np.complex128]
    residual: float

    @property
    def outlier(self) -> float:
        """Largest real part of Eq. 71's collective modes (includes zero)."""
        return float(np.max(self.meanfield_eigenvalues.real))

    @property
    def stable(self) -> bool:
        return self.bulk_radius < 1 and self.outlier < 1


def fixed_point_residual(parameters: GaussianGeometry | GaussianRankOne,
                         kappa: Array, variance: float, order: int = 200) -> Array:
    p = _geometry(parameters)
    kappa = np.atleast_1d(kappa).astype(float)
    mean, quenched, n_cov = p.statistics(kappa)
    moments = _moments(mean, variance, order)
    return np.r_[kappa - p.mean_n * moments[0] - n_cov * moments[2],
                 variance - p.g**2 * moments[1] - quenched]


def _fixed_point(p: GaussianGeometry, kappa: Array, variance: float,
                 order: int) -> FixedPoint:
    mean, _, n_cov = p.statistics(kappa)
    phi, _, gain, second, third, gain2, phi_gain, phi_second = _moments(mean, variance, order)
    a = (np.outer(p.mean_n, p.mean_m) + p.cov_nm) * gain + np.outer(n_cov, p.mean_m) * second
    b = (p.mean_n * second + n_cov * third) / 2
    g_mean = 2 * p.g**2 * phi_gain
    g_var = p.g**2 * (gain2 + phi_second)
    d = 2 * (p.cov_mm @ kappa + p.m_input)
    matrix = np.zeros((p.rank + 2, p.rank + 2))
    matrix[0, 2:] = p.mean_m
    matrix[1] = np.r_[g_mean, g_var, d]
    matrix[2:, 0] = b * g_mean
    matrix[2:, 1] = b * g_var
    matrix[2:, 2:] = a + np.outer(b, d)
    return FixedPoint(kappa.copy(), mean, variance, p.g * float(np.sqrt(gain2)),
                      np.asarray(np.linalg.eigvals(matrix), dtype=complex),
                      float(np.max(np.abs(fixed_point_residual(p, kappa, variance, order)))))


def solve_fixed_points(parameters: GaussianGeometry | GaussianRankOne,
                       seeds: list[tuple[float | Array, float]] | None = None,
                       quadrature_order: int = 200, tolerance: float = 1e-7) -> list[FixedPoint]:
    """Find distinct stationary branches by multistart nonlinear solution.

    Default seeds span both signs for rank one. For rank > 1 supply seeds
    covering the desired attractors; a continuum cannot be enumerated.
    """
    p = _geometry(parameters)
    if seeds is None:
        scale = max(float(np.linalg.norm(p.mean_n)), float(np.linalg.norm(p.cov_nm)), 1.0)
        seeds = [(np.full(p.rank, k * scale), v)
                 for k in (-1, -0.25, 0, 0.25, 1)
                 for v in (0.0, max(p.g**2 + p.var_input, 0.2))]
    points: list[FixedPoint] = []
    lower = np.r_[np.full(p.rank, -np.inf), 0]
    upper = np.full(p.rank + 1, np.inf)
    for initial_kappa, initial_variance in seeds:
        kappa = np.atleast_1d(initial_kappa).astype(float)
        if kappa.shape != (p.rank,) or initial_variance < 0:
            raise ValueError("seeds must contain rank-vectors and nonnegative variance")
        initial = np.r_[kappa, initial_variance]
        result = least_squares(lambda x: fixed_point_residual(p, x[:-1], x[-1], quadrature_order),
                               initial, bounds=(lower, upper), xtol=1e-12,
                               ftol=1e-12, gtol=1e-12, max_nfev=400)
        values = result.x
        if np.linalg.norm(values) < 1e-5 and p.var_input == 0 and p.mean_input == 0:
            values = np.zeros_like(values)
        point = _fixed_point(p, values[:-1], float(values[-1]), quadrature_order)
        if point.residual > tolerance:
            continue
        if not any(np.linalg.norm(np.r_[point.kappa - other.kappa,
                                        point.variance - other.variance]) < 1e-5 for other in points):
            points.append(point)
    return sorted(points, key=lambda point: tuple(point.kappa))


@dataclass(frozen=True)
class ChaoticState:
    kappa: Array
    mean: float
    variance: float
    long_lag_variance: float
    residual: float

    @property
    def temporal_variance(self) -> float:
        return self.variance - self.long_lag_variance


def _pair_expectation(function: Callable[[Array], Array], mean: float, variance: float,
                      covariance: float, order: int) -> float:
    nodes, weights = gaussian_quadrature(order)
    conditional = mean + np.sqrt(max(covariance, 0.0)) * nodes[:, None]
    conditional = conditional + np.sqrt(max(variance - covariance, 0.0)) * nodes[None, :]
    average = function(conditional) @ weights
    return float(weights @ average**2)


def _logcosh(x: Array) -> Array:
    return np.logaddexp(x, -x) - np.log(2.0)


def solve_chaotic(parameters: GaussianGeometry | GaussianRankOne,
                  initial: tuple[float | Array, float, float] | None = None,
                  quadrature_order: int = 100, tolerance: float = 2e-6) -> ChaoticState:
    """Solve full stationary chaotic DMF (Eq. 77), excluding static roots.

    Initial tuple is (kappa, zero-lag variance, long-lag variance).
    This finds one branch; continue from preceding solutions to track a branch.
    A solution does not constitute a proof of its dynamical stability.
    """
    p = _geometry(parameters)
    if initial is None:
        initial = (np.zeros(p.rank), max(p.g**2, 0.2), 0.0)
    kappa, delta0, delta_inf = initial
    if not 0 <= delta_inf < delta0:
        raise ValueError("initial variances must satisfy 0 <= long lag < zero lag")
    def residual(values: Array) -> Array:
        k = values[:p.rank]
        di, amplitude = values[-2:]
        d0 = di + amplitude
        mean, quenched, n_cov = p.statistics(k)
        moments = _moments(mean, d0, quadrature_order)
        rates = _pair_expectation(np.tanh, mean, d0, di, quadrature_order)
        f2 = gaussian_expectation(lambda x: _logcosh(x)**2, mean, d0, quadrature_order)
        ff = _pair_expectation(_logcosh, mean, d0, di, quadrature_order)
        # Subtract the long-lag equation and divide by amplitude squared,
        # avoiding the continuum of spurious zero-amplitude (static) roots.
        energy = 0.5 - p.g**2 * (f2 - ff - amplitude * rates) / amplitude**2
        return np.r_[k - p.mean_n * moments[0] - n_cov * moments[2],
                     di - p.g**2 * rates - quenched, energy]
    initial_values = np.r_[np.atleast_1d(kappa), delta_inf, delta0 - delta_inf]
    lower = np.r_[np.full(p.rank, -np.inf), 0.0, 1e-4]
    result = least_squares(residual, initial_values, bounds=(lower, np.full(p.rank + 2, np.inf)),
                           xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=400)
    error = float(np.max(np.abs(residual(result.x))))
    if error > tolerance or result.x[-1] < 1.01e-4:
        raise ValueError(f"no nonstatic chaotic solution found (residual {error:.3g})")
    k = result.x[:p.rank]
    di, amplitude = result.x[-2:]
    mean, _, _ = p.statistics(k)
    return ChaoticState(k.copy(), mean, float(di + amplitude), float(di), error)


@dataclass(frozen=True)
class PhaseBoundaries:
    g: Array
    structure_onset: Array
    static_chaos: Array


def phase_boundaries(g_values: Array, quadrature_order: int = 100) -> PhaseBoundaries:
    """Figure 1B boundaries for mean_m=mean_n=sqrt(S), std_m=std_n=1.

    structure_onset separates zero-mean from nonzero-mean activity. For g>1
    it is computed on the full chaotic solution (Eq. 88), not on an unstable
    static fixed point. static_chaos gives S above which structured states
    are static; it is NaN for g<1, where there is no chaotic phase.
    """
    gs = np.atleast_1d(g_values).astype(float)
    if np.any(gs < 0):
        raise ValueError("g must be nonnegative")
    onset = np.ones_like(gs)
    static = np.full_like(gs, np.nan)
    for index, g in enumerate(gs):
        if np.isclose(g, 1, atol=1e-10, rtol=0):
            static[index] = 1.0
        elif g > 1:
            chaos = solve_chaotic(GaussianRankOne(float(g)), quadrature_order=quadrature_order)
            gain = _moments(0, chaos.variance, quadrature_order)[2]
            onset[index] = 1 / gain
            def boundary_residual(values: Array) -> Array:
                mean, variance = values
                phi, phi2, _, _, _, gain2, _, _ = _moments(mean, variance, quadrature_order)
                return np.array([variance - g**2 * phi2 - mean * phi,
                                 g**2 * gain2 - 1])
            result = least_squares(boundary_residual, np.array([max(g, .1), max(g*g, .1)]),
                                   bounds=(np.array([1e-7, 1e-7]), np.array([np.inf, np.inf])),
                                   xtol=1e-11, gtol=1e-11, ftol=1e-11, max_nfev=300)
            if np.max(np.abs(boundary_residual(result.x))) > 1e-6:
                raise ValueError(f"static/chaos boundary did not converge for g={g}")
            mean, variance = result.x
            phi = _moments(mean, variance, quadrature_order)[0]
            static[index] = mean / phi
    return PhaseBoundaries(gs, onset, static)
