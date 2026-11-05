import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lowrank.network import RateNetwork, gaussian_network, simulate


@pytest.mark.parametrize("rank", [0, 1, 3])
@pytest.mark.parametrize("gain", [0.0, 0.8])
def test_factored_connectivity_equals_dense(rank, gain):
    network = gaussian_network(jax.random.key(4), 17, rank, g=gain)
    rates = jax.random.normal(jax.random.key(8), (5, 17))
    np.testing.assert_allclose(
        network.recurrent(rates), rates @ network.dense_connectivity().T,
        atol=3e-7, rtol=2e-6,
    )


def test_gaussian_construction_moments_and_reproducibility():
    kwargs = dict(mean_m=1.2, mean_n=-0.5,
                  std_m=0.8, std_n=1.5, correlation=0.6)
    network = gaussian_network(jax.random.key(7), 20_000, 2, **kwargs)
    repeated = gaussian_network(jax.random.key(7), 20_000, 2, **kwargs)
    np.testing.assert_array_equal(network.n, repeated.n)
    assert network.random_matrix is None
    assert abs(float(jnp.mean(network.m)) - 1.2) < 0.02
    assert abs(float(jnp.std(network.n)) - 1.5) < 0.03
    covariance = jnp.mean((network.m - 1.2) * (network.n + 0.5))
    assert abs(float(covariance) - 0.6 * 0.8 * 1.5) < 0.03


def test_random_connectivity_normalization():
    network = gaussian_network(jax.random.key(7), 400, g=0.8)
    assert network.random_matrix is not None
    assert float(jnp.var(network.random_matrix)) == pytest.approx(1 / 400, rel=0.02)


@pytest.mark.parametrize("sample_every", [1, 7, 150])
def test_rk4_recovers_driven_leak_and_preserves_last_sample(sample_every):
    network = RateNetwork(jnp.zeros((3, 0)), jnp.zeros((3, 0)), tau=2)
    initial = jnp.array([1.0, -2.0, 4.0])
    drive = jnp.array([0.5, 0.2, -1.0])
    result = simulate(network, initial, jnp.broadcast_to(drive, (100, 3)),
                      dt=0.1, sample_every=sample_every)
    expected = drive + (initial - drive) * jnp.exp(-result.times[:, None] / 2)
    np.testing.assert_allclose(result.states, expected, atol=2e-6)
    assert float(result.times[0]) == 0
    assert float(result.times[-1]) == 10
    np.testing.assert_array_equal(result.states[0], initial)


def test_euler_and_piecewise_constant_input():
    network = RateNetwork(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    result = simulate(network, jnp.array([0]), jnp.array([[1], [0], [-1]]),
                      dt=0.5, method="euler")
    np.testing.assert_allclose(result.states[:, 0], [0, 0.5, 0.25, -0.375])


def test_nonlinear_integration_matches_independent_dense_solver():
    from scipy.integrate import solve_ivp

    network = gaussian_network(jax.random.key(32), 12, 2, g=0.7)
    initial = np.linspace(-1, 1, 12)
    dense = np.asarray(network.dense_connectivity())
    drive = np.linspace(-0.2, 0.4, 12)
    result = simulate(network, jnp.asarray(initial),
                      jnp.asarray(np.broadcast_to(drive, (100, 12))), dt=0.02)
    solution = solve_ivp(
        lambda _t, x: -x + dense @ np.tanh(x) + drive,
        (0, 2), initial, t_eval=np.asarray(result.times).clip(0, 2),
        rtol=1e-10, atol=1e-11,
    )
    np.testing.assert_allclose(result.states, solution.y.T, atol=5e-7, rtol=2e-6)


def test_empty_inputs_and_batched_simulation():
    network = gaussian_network(jax.random.key(1), 4)
    initial = jnp.ones(4)
    empty = simulate(network, initial, jnp.zeros((0, 4)))
    np.testing.assert_array_equal(empty.states, initial[None])
    trials = jnp.ones((2, 10, 4))
    batch = jax.vmap(lambda drive: simulate(network, initial, drive).states)(trials)
    assert batch.shape == (2, 11, 4)
    np.testing.assert_array_equal(batch[0], batch[1])


@pytest.mark.parametrize("kwargs", [{"tau": 0}, {"g": 1}])
def test_invalid_network_parameters(kwargs):
    with pytest.raises(ValueError):
        RateNetwork(jnp.ones((3, 1)), jnp.ones((3, 1)), **kwargs)


def test_invalid_simulation_parameters():
    network = gaussian_network(jax.random.key(1), 4)
    with pytest.raises(ValueError, match="sample_every"):
        simulate(network, jnp.zeros(4), jnp.zeros((3, 4)), sample_every=0)
