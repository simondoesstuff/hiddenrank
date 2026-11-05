"""The task encodings must implement the overlap geometry of the paper."""

import numpy as np
import pytest

from lowrank.tasks import contextual, detection, go_nogo, patterns, ring


def overlap(a, b):
    return a.T @ b / len(a)


def test_pattern_covariance_and_seed():
    vectors = patterns(128, 7, seed=4)
    np.testing.assert_allclose(vectors.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(overlap(vectors, vectors), np.eye(7), atol=1e-14)
    np.testing.assert_array_equal(vectors, patterns(128, 7, seed=4))
    assert not np.array_equal(vectors, patterns(128, 7, seed=5))
    with pytest.raises(ValueError):
        patterns(4, 4)


def test_go_nogo_selectivity():
    task = go_nogo(128)
    np.testing.assert_array_equal(task.m[:, 0], task.readout)
    np.testing.assert_allclose(overlap(task.n, task.directions), [[4., 0.]], atol=1e-14)
    np.testing.assert_allclose(overlap(task.m, task.n), 0, atol=1e-14)


def test_detection_shared_axis_and_readout():
    task = detection(128)
    np.testing.assert_allclose(overlap(task.n, task.m), [[4.]])
    np.testing.assert_allclose(overlap(task.n, task.directions), [[1.44, 2., 0.]], atol=1e-14)
    np.testing.assert_allclose(overlap(task.m[:, 0], task.readout), 1.44)


@pytest.mark.parametrize("integration,rn", [(False, 1.6), (True, 1.38)])
def test_context_gating_and_common_readout(integration, rn):
    task = contextual(128, integration=integration, architecture="paper")
    expected = np.full((2, 2), .6)
    np.fill_diagonal(expected, 3 * rn + .6)
    np.testing.assert_allclose(overlap(task.n, task.m), expected, atol=1e-14)
    np.testing.assert_allclose(overlap(task.n, task.directions[:, :2]), 1.44*np.eye(2), atol=1e-14)
    np.testing.assert_allclose(overlap(task.n, task.directions[:, 2:]), rn*np.eye(2), atol=1e-14)
    np.testing.assert_allclose(overlap(task.m, task.readout), [.6, .6], atol=1e-14)


def test_ring_rotational_symmetry():
    task = ring(128, strength=1.8)
    np.testing.assert_allclose(overlap(task.n, task.m), 1.8*np.eye(2), atol=1e-14)
    assert task.g == 0


def test_discrimination_dynamics_respond_to_go_but_not_orthogonal_stimulus():
    import jax.numpy as jnp
    from lowrank.network import RateNetwork, simulate

    task = go_nogo(256, seed=3)
    network = RateNetwork(task.m, task.n)
    responses = []
    for stimulus in task.directions.T:
        result = simulate(network, jnp.zeros(256),
                          jnp.broadcast_to(jnp.asarray(stimulus), (200, 256)),
                          dt=.1, sample_every=200)
        responses.append(float(np.tanh(np.asarray(result.states[-1])) @ task.readout / 256))
    assert responses[0] > .5
    # Nonlinear finite-size readout leakage remains possible despite orthogonal
    # input vectors; it must be small relative to the Go response.
    assert abs(responses[1]) < .2 * responses[0]


def test_supplement_context_coefficient_order():
    task = contextual(128)
    np.testing.assert_allclose(overlap(task.n, task.directions[:, 2:]), 3*np.eye(2), atol=1e-14)
    np.testing.assert_allclose(overlap(task.m, task.directions[:, 2:]), 1.6*np.eye(2), atol=1e-14)
