import jax.numpy as jnp
import numpy as np
import pytest

from lowrank.analysis import overlaps, pca, project, readout


def test_overlaps_and_readout_use_population_normalization():
    states = jnp.array([[1.0, 2.0, -1.0], [-2.0, 0.0, 1.0]])
    vectors = jnp.array([[1.0, 2.0], [-1.0, 1.0], [0.0, 3.0]])
    np.testing.assert_allclose(overlaps(states, vectors), states @ vectors / 3)
    np.testing.assert_allclose(overlaps(states, vectors[:, 0]), states @ vectors[:, 0] / 3)
    np.testing.assert_allclose(readout(states, vectors), jnp.tanh(states) @ vectors / 3)


def test_projection_recovers_nonorthogonal_coefficients():
    basis = jnp.array([[1.0, 2.0], [0.0, 1.0], [1.0, -1.0]])
    coefficients = jnp.array([[0.4, -0.8], [1.2, 0.1]])
    np.testing.assert_allclose(project(coefficients @ basis.T, basis),
                               coefficients, atol=4e-7)


def test_pca_reconstructs_and_reports_total_variance():
    rng = np.random.default_rng(4)
    states = rng.normal(size=(30, 5)) @ rng.normal(size=(5, 8)) + 2
    result = pca(jnp.asarray(states), 5)
    np.testing.assert_allclose(result.scores @ result.components + result.mean,
                               states, atol=8e-6)
    expected = np.linalg.svd(states - states.mean(0), compute_uv=False)**2 / 29
    np.testing.assert_allclose(result.explained_variance, expected[:5], rtol=2e-6)
    np.testing.assert_allclose(result.explained_variance_ratio.sum(), 1, atol=1e-6)
    partial = pca(jnp.asarray(states), 1)
    assert float(partial.explained_variance_ratio.sum()) < 1


def test_constant_pca_and_invalid_shapes():
    result = pca(jnp.ones((5, 3)), 2)
    np.testing.assert_array_equal(result.explained_variance_ratio, [0, 0])
    assert bool(jnp.all(jnp.isfinite(result.scores)))
    with pytest.raises(ValueError):
        pca(jnp.ones((1, 3)))
    with pytest.raises(ValueError):
        pca(jnp.ones((4, 3)), 4)
