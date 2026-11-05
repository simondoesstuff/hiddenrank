"""Task covariance and population-limit decision reference regressions."""

import numpy as np
import pytest

from lowrank.meanfield import solve_fixed_points
from lowrank.protocol import (
    context_strengths, task_geometry, theoretical_readout, theoretical_upper_readout,
)


def test_rank_two_geometry_keeps_shared_readout_cross_term():
    kappa = np.array([.7, -.2])
    a, b = .6, .1
    cue = np.array([.08, -.18])
    geometry = task_geometry("integration", stimulus=(a, b), context=0)
    _, variance, n_cov = geometry.statistics(kappa)
    # Eq. 155 decomposed into independent yA,yB,w,IA,IB,ctxA,ctxB.
    expected = (1.2**2 * (kappa @ kappa) + .6**2 * kappa.sum()**2
                + 1.2**2 * (a*a + b*b) + np.sum((3*kappa + cue)**2))
    assert variance == pytest.approx(expected)
    np.testing.assert_allclose(n_cov,
                               3*1.38*kappa + .6*kappa.sum() + 1.2**2*np.array([a, b]) + 1.38*cue)


def test_detection_resting_has_no_stimulus_background():
    resting = task_geometry("detection")
    driven = task_geometry("detection", stimulus=(.6, 0.), background=1.)
    assert resting.var_input == 0
    assert driven.var_input == pytest.approx(1 + 1.2**2 * .6**2)
    np.testing.assert_allclose(driven.n_input, [1.2**2 * .6])


@pytest.mark.parametrize("name", ["context", "integration"])
def test_theoretical_reference_is_context_symmetric_stable_upper_branch(name):
    first = theoretical_upper_readout(name, context=0)
    second = theoretical_upper_readout(name, context=1)
    assert first == pytest.approx(second, abs=1e-9)
    assert 0 < first < .3
    geometry = task_geometry(name, context=0)
    points = solve_fixed_points(geometry,
                                seeds=[(np.array([2., 2.]), 40.),
                                       (np.array([-2., -2.]), 40.)],
                                quadrature_order=400)
    assert len(points) == 2
    assert all(point.stable for point in points)
    assert first == pytest.approx(max(theoretical_readout(name, point) for point in points))
    # Selecting a driven operating point is explicit, not trial-dependent.
    driven = theoretical_upper_readout(name, context=0, stimulus=(1., 0.))
    assert driven > 0
    assert driven != pytest.approx(first, abs=1e-5)


def test_go_nogo_reference_is_driven_and_cues_are_validated():
    assert theoretical_upper_readout("go_nogo") > 1
    with pytest.raises(ValueError):
        context_strengths("integration", 2)
    with pytest.raises(ValueError):
        task_geometry("unknown")
