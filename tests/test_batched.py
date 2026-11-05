import jax.numpy as jnp
import numpy as np
import pytest

from lowrank.batched import simulate_driven
from lowrank.network import RateNetwork, simulate


@pytest.mark.parametrize('method', ['rk4', 'euler'])
def test_batched_observations_equal_independent_integrations(method):
    rng = np.random.default_rng(8)
    m, n = rng.normal(size=(2, 12, 2))
    chi = rng.normal(size=(12, 12)) / np.sqrt(12)
    net = RateNetwork(m, n, chi, g=.8)
    initial = rng.normal(size=(3, 12))
    directions = rng.normal(size=(12, 2))
    coeff = rng.normal(size=(13, 3, 2))
    weights = rng.normal(size=12)
    projections = rng.normal(size=(12, 3))
    measured = simulate_driven(net, initial, coeff, np.zeros_like(coeff), directions,
                              weights, projections, dt=.1, sample_every=4, method=method)
    for condition in range(3):
        result = simulate(net, jnp.asarray(initial[condition]), jnp.asarray(coeff[:, condition] @ directions.T),
                          dt=.1, sample_every=4, method=method)
        x = np.asarray(result.states)
        np.testing.assert_allclose(measured.times, result.times, atol=1e-6)
        np.testing.assert_allclose(measured.readout[:, condition], np.tanh(x) @ weights/12, atol=3e-7)
        np.testing.assert_allclose(measured.projections[:, condition], x @ projections/12, atol=3e-7)
        np.testing.assert_allclose(measured.kappa[:, condition], np.tanh(x) @ n/12, atol=3e-7)


def test_additive_white_noise_uses_supplied_brownian_increments():
    net = RateNetwork(jnp.zeros((2, 0)), jnp.zeros((2, 0)))
    coefficients = np.zeros((2, 1, 1))
    increments = np.array([[[.2]], [[-.1]]])
    result = simulate_driven(net, np.zeros((1, 2)), coefficients, increments,
                             np.ones((2, 1)), np.ones(2), np.eye(2),
                             dt=.1, sample_every=1, method='euler')
    # x_{k+1}=.9*x_k+dW_k, measured as x/N along neuron coordinate axes.
    np.testing.assert_allclose(result.projections[:, 0, 0], [0, .1, .04], atol=1e-7)
