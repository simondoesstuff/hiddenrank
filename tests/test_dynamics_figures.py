"""Sharded dynamics agree with the reference RK4 trajectory and ensemble reduction."""
import json

import jax.numpy as jnp
import numpy as np
import pytest

from lowrank import dynamics_figures as figures
from lowrank.network import RateNetwork, simulate
from lowrank.spontaneous import simulate_spontaneous


@pytest.mark.parametrize('steps', [10, 11])
def test_streaming_spontaneous_matches_full_rk4(steps):
    rng = np.random.default_rng(18)
    size = 16
    m, n = rng.normal(size=(2, 4, size)).astype(np.float32)
    chi = rng.normal(size=(size, size)).astype(np.float32)/np.sqrt(size)
    gains = np.array([.7, .7, 2., 2.], dtype=np.float32)
    times, traces, stats = simulate_spontaneous(m, n, chi, gains, steps=steps,
                                               dt=.1, sample_every=3)
    for i in range(4):
        net = RateNetwork(m[i, :, None], n[i, :, None], chi, g=float(gains[i]))
        result = simulate(net, jnp.asarray(m[i]), jnp.zeros((steps, size)),
                          dt=.1, sample_every=3)
        states = np.asarray(result.states)
        tail = states[len(states)//2:]
        expected = [np.mean(np.tanh(tail) @ n[i]/size),
                    np.var(tail.mean(axis=0)), np.mean(np.var(tail, axis=0))]
        np.testing.assert_allclose(times, result.times, atol=1e-7)
        np.testing.assert_allclose(traces[i], states[:, :6], atol=2e-6)
        np.testing.assert_allclose(stats[i], expected, atol=2e-6)


def test_spontaneous_shards_have_exact_sign_symmetry(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, '_phase_boundaries', lambda: (np.zeros((1, 2)), np.zeros((1, 2))))
    monkeypatch.setattr(figures, '_spontaneous_theory', lambda _: (np.zeros(1), np.zeros((1, 4))))
    paths = [figures.run_dynamics_trial('spontaneous', tmp_path/f'{i}.npz', size=16,
             trial_id=i, duration=1.) for i in range(2)]
    output = figures.aggregate_dynamics(paths[::-1], tmp_path/'aggregate.npz')
    with np.load(output) as data:
        statistics = data['statistics']
        assert statistics.shape == (4, 4, 3)
        np.testing.assert_array_equal(statistics[:, 1::2], statistics[:, ::2]*[-1., 1., 1.])
        assert json.loads(str(data['metadata']))['trial_ids'] == [0, 1]
    with pytest.raises(ValueError, match='duplicate'):
        figures.aggregate_dynamics([paths[0], paths[0]], tmp_path/'duplicate.npz')


def test_input_aggregation_averages_before_pca(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, '_input_theory', lambda: (np.zeros(1), np.zeros((3, 1, 3))))
    paths = [figures.run_dynamics_trial('inputs', tmp_path/f'{i}.npz', size=16,
             trial_id=i, duration=2.) for i in range(2)]
    with np.load(paths[0]) as first, np.load(paths[1]) as second:
        np.testing.assert_array_equal(first['basis'], second['basis'])
        assert not np.array_equal(first['states'], second['states'])
        average = (first['states'].astype(float)+second['states'])/2
        basis = first['basis']
    output = figures.aggregate_dynamics(paths, tmp_path/'aggregate.npz')
    with np.load(output) as data:
        np.testing.assert_allclose(data['traces'], average[:, :6])
        np.testing.assert_allclose(data['coordinates'], average @ np.linalg.pinv(basis).T)
        zscore = (average-average.mean(axis=0))/np.maximum(average.std(axis=0), 1e-10)
        s = np.linalg.svd(zscore, compute_uv=False)
        np.testing.assert_allclose(data['explained'], s[:6]**2/(s**2).sum())
