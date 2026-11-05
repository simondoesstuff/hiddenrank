"""Population-limit task geometry and the theoretical decision reference.

All moments are for iid Gaussian patterns, irrespective of finite network seed.
The upper reference is a stationary resting branch, not an empirical maximum.
"""

from functools import lru_cache

import numpy as np

from lowrank.meanfield import (
    Array, FixedPoint, GaussianGeometry, gaussian_expectation, solve_fixed_points,
)


def context_strengths(name: str, context: int) -> tuple[float, float]:
    """Paper Fig. 5/6 cues in the order (context A, context B)."""
    if name not in {"context", "integration"} or context not in (0, 1):
        raise ValueError("require context/integration and context index 0 or 1")
    strengths = (.08, -.18 if name == "integration" else -.14)
    return strengths if context == 0 else (strengths[1], strengths[0])


def task_geometry(name: str, *, stimulus: tuple[float, float] = (0., 0.),
                  context: int | None = None, background: float = 0.,
                  modulation: float = 0.) -> GaussianGeometry:
    """Exact centered covariances from Eqs. 145, 151 and 155.

    ``background`` is the standard deviation of independent quenched input.
    Temporal input noise is deliberately absent from stationary equations.
    ``context=None`` means no cue; stimulus strengths multiply IA and IB.
    """
    a, b = stimulus
    if name == "go_nogo":
        return GaussianGeometry(.8, np.zeros(1), np.zeros(1),
                                np.array([[4.]]), np.zeros((1, 1)),
                                var_input=4 * (a*a + b*b) + background**2,
                                cov_n_input=np.array([4 * a]))
    if name == "detection":
        return GaussianGeometry(.8, np.zeros(1), np.zeros(1),
                                np.array([[1.2**2 + 2**2]]), np.array([[4.]]),
                                var_input=1.2**2 * a*a + modulation**2 + background**2,
                                cov_m_input=np.array([2 * modulation]),
                                cov_n_input=np.array([1.2**2 * a + 2 * modulation]))
    if name not in {"context", "integration"}:
        raise ValueError(f"no Gaussian task geometry for {name!r}")
    cue = np.zeros(2) if context is None else np.asarray(context_strengths(name, context))
    rm, rn, bm, bn = 3., 1.38 if name == "integration" else 1.6, .6, 1.
    # The same w enters both columns: its covariance is a full ones matrix.
    return GaussianGeometry(
        .8, np.zeros(2), np.zeros(2),
        (1.2**2 + rm**2) * np.eye(2) + bm**2 * np.ones((2, 2)),
        rm * rn * np.eye(2) + bm * bn * np.ones((2, 2)),
        var_input=1.2**2 * (a*a + b*b) + float(cue @ cue) + background**2,
        cov_m_input=rm * cue,
        cov_n_input=1.2**2 * np.array([a, b]) + rn * cue,
    )


def theoretical_readout(name: str, point: FixedPoint, *, quadrature_order: int = 400) -> float:
    """Raw E[w tanh(x)] via Gaussian integration by parts (Eq. 158)."""
    scale = {"go_nogo": 4., "detection": 1.2**2,
             "context": .6, "integration": .6}
    if name not in scale:
        raise ValueError(f"no task readout for {name!r}")
    gain = gaussian_expectation(lambda x: 1 - np.tanh(x)**2,
                                point.mean, point.variance, quadrature_order)
    return float(scale[name] * point.kappa.sum() * gain)


@lru_cache(maxsize=64)
def theoretical_upper_readout(name: str, *, context: int | None = None,
                              stimulus: tuple[float, float] = (0., 0.),
                              background: float = 0.,
                              quadrature_order: int = 400) -> float:
    """Largest stable predicted raw readout at the specified operating point.

    Defaults to the upper resting state. With measured raw baseline ``z0``,
    classify the offset endpoint against ``(upper - z0) / 2``. The default
    Go/NoGo reference is its driven Go state, because its resting state is zero.
    """
    if name == "go_nogo" and stimulus == (0., 0.):
        stimulus = (1., 0.)
    geometry = task_geometry(name, stimulus=stimulus, context=context,
                             background=background)
    seeds: list[tuple[float | Array, float]] = [
        (np.full(geometry.rank, sign * amplitude), variance)
        for sign in (-1, 1) for amplitude, variance in ((.5, 3.), (2., 40.))]
    points = solve_fixed_points(geometry, seeds=seeds, quadrature_order=quadrature_order)
    stable = [point for point in points if point.stable]
    if not stable:
        raise RuntimeError(f"no stable theoretical reference found for {name}")
    return max(theoretical_readout(name, point, quadrature_order=quadrature_order)
               for point in stable)
