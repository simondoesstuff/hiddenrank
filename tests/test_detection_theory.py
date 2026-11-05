"""Focused numerical checks for the Figure 4F branch construction."""
import numpy as np
import pytest

from lowrank.detection_theory import DetectionTheory, run_detection_theory
from lowrank.meanfield import fixed_point_residual, gaussian_expectation


def test_disorder_free_critical_overlap_and_background():
    p = DetectionTheory(g=0)
    gain = gaussian_expectation(lambda x: 1 - np.tanh(x)**2, 0, 1)
    assert p.baseline_variance() == 1
    assert p.critical_rm == pytest.approx(1 / (p.rn * gain))
    assert p.resting_critical_rm == .5
    assert p.fold(.6)[0] == 0


def test_fold_satisfies_stationarity_and_is_turning_point():
    p = DetectionTheory()
    c, k, v = p.fold(2)
    np.testing.assert_allclose(fixed_point_residual(p.parameters(2, c), np.array([k]), v), 0, atol=1e-9)
    # A saddle-node has a singular stationary Jacobian. Differentiate the
    # defining equations independently of the scalar branch parameterization.
    state = np.array([k, v])
    step = 1e-5
    jac = np.column_stack([(fixed_point_residual(p.parameters(2, c),
                                                 (state + step * e)[:1], (state + step * e)[1]) -
                            fixed_point_residual(p.parameters(2, c),
                                                 (state - step * e)[:1], (state - step * e)[1])) / (2 * step)
                           for e in np.eye(2)])
    assert abs(np.linalg.det(jac)) < 3e-5
    assert .45 < c < .65  # Paper's target detection threshold is approximately .5.


def test_threshold_increases_while_resting_timescale_decreases():
    p = DetectionTheory()
    rms = [.5001, .6, 1., 1.5, 2.]
    thresholds = [p.fold(rm)[0] for rm in rms]
    timescales = [p.resting_timescale(rm)[0] for rm in rms]
    assert np.all(np.diff(thresholds) >= 0)
    assert np.all(np.diff(timescales) < 0)
    assert timescales[0] > 100
    with pytest.raises(ValueError, match="above the critical"):
        p.resting_timescale(.5)


def test_curve_artifact_records_both_background_conventions(tmp_path):
    path = run_detection_theory(tmp_path / "curve.npz", points=8)
    with np.load(path) as data:
        assert data["rm"].shape == (8,)
        assert data["critical_rm"] == .5
        assert data["stimulus_critical_rm"] > data["critical_rm"]
        np.testing.assert_allclose(data["timescale"], 1 / (1 - data["outlier"]))
        assert data["threshold"][-1] > .45
