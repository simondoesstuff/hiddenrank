"""Population overlaps, readouts, and activity subspaces."""

# einx's public re-exports are missing explicit typing exports.
from einx import dot as dot  # pyright: ignore[reportPrivateImportUsage]
import equinox as eqx
import jax.numpy as jnp
from jax import Array


def overlaps(activity: Array, vectors: Array) -> Array:
    """Population inner products (1/N) activity @ vectors.

    Activity can be currents or rates; pass tanh(states) for paper kappa.
    Vectors may be one vector (N,) or a matrix (N, rank).
    """
    activity, vectors = jnp.asarray(activity), jnp.asarray(vectors)
    if vectors.ndim == 1:
        return dot("... i, i -> ...", activity, vectors) / activity.shape[-1]
    return dot("... i, i r -> ... r", activity, vectors) / activity.shape[-1]


def readout(states: Array, weights: Array) -> Array:
    """Linear readout of firing rates: z = mean(w tanh(x))."""
    return overlaps(jnp.tanh(states), weights)


def project(activity: Array, vectors: Array) -> Array:
    """Least-squares coordinates in a possibly nonorthogonal column basis.

    Unlike raw overlaps, these recover coefficients of activity = c @ V.T.
    The pseudoinverse handles a redundant or rank-deficient basis.
    """
    return jnp.asarray(activity) @ jnp.linalg.pinv(jnp.asarray(vectors)).T


class PCA(eqx.Module):
    mean: Array
    components: Array
    scores: Array
    explained_variance: Array
    explained_variance_ratio: Array


def pca(activity: Array, n_components: int = 3) -> PCA:
    """Centered SVD of (observations, neurons), with deterministic signs.

    Components have shape (components, neurons). Variance ratios use total
    variance, including discarded components. Constant activity yields zeros.
    """
    activity = jnp.asarray(activity)
    if activity.ndim != 2 or activity.shape[0] < 2:
        raise ValueError("PCA needs a matrix with at least two observations")
    if not 1 <= n_components <= min(activity.shape):
        raise ValueError("n_components exceeds the available matrix dimensions")
    mean = jnp.mean(activity, axis=0)
    u, singular, vh = jnp.linalg.svd(activity - mean, full_matrices=False)
    components = vh[:n_components]
    largest = jnp.argmax(jnp.abs(components), axis=1)
    signs = jnp.sign(components[jnp.arange(n_components), largest])
    signs = jnp.where(signs == 0, 1, signs)
    components = components * signs[:, None]
    scores = u[:, :n_components] * singular[:n_components] * signs
    variance = singular**2 / (activity.shape[0] - 1)
    total = jnp.sum(variance)
    ratio = variance[:n_components] / jnp.where(total > 0, total, 1)
    return PCA(mean, components, scores, variance[:n_components], ratio)
