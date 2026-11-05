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
    geometry = task_geometry("integration", stimulus=(a, b), context=0, architecture="paper")
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
    assert 0 < first < .4
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


def test_supplement_architecture_is_explicit_and_restores_nogo_branch():
    # Source/paper m/n coefficient reversal changes variance and input overlap,
    # while leaving the recurrent n-m overlap matrix unchanged.
    paper = task_geometry("context", stimulus=(0., 1.), context=0, architecture="paper")
    source = task_geometry("context", stimulus=(0., 1.), context=0,
                           architecture="supplement")
    np.testing.assert_array_equal(paper.cov_nm, source.cov_nm)
    source_points = solve_fixed_points(source, seeds=[(np.array([-2., -1.]), 20.)],
                                       quadrature_order=400)
    assert len(source_points) == 1 and source_points[0].stable
    assert source_points[0].kappa.sum() < 0
    paper_points = solve_fixed_points(paper,
                                      seeds=[(np.array([a, b]), 20.)
                                             for a in (-2., 0., 2.) for b in (-2., 0., 2.)],
                                      quadrature_order=400)
    assert any(point.stable for point in paper_points)
    assert all(point.kappa.sum() > 0 for point in paper_points if point.stable)
    assert theoretical_upper_readout("context", context=0, architecture="supplement") > .29
    with pytest.raises(ValueError):
        task_geometry("context", architecture="unknown")
