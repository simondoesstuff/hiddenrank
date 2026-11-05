"""Factored low-rank tanh rate networks (Mastrogiuseppe & Ostojic, 2018)."""

from typing import Literal

# einx's public re-exports are missing explicit typing exports.
from einx import dot as dot  # pyright: ignore[reportPrivateImportUsage]
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class RateNetwork(eqx.Module):
    """tau dx/dt = -x + (g chi + m n.T / N) tanh(x) + I.

    Columns of ``m`` and ``n`` are connectivity vectors. ``random_matrix``
    stores chi, with variance 1/N (the gain g is applied separately).
    """

    m: Array
    n: Array
    random_matrix: Array | None
    g: float
    tau: float

    def __init__(self, m, n, random_matrix=None, g=0.0, tau=1.0):
        self.m = jnp.asarray(m)
        self.n = jnp.asarray(n)
        if self.m.ndim != 2 or self.m.shape != self.n.shape:
            raise ValueError("m and n must have the same (neurons, rank) shape")
        if self.m.shape[0] == 0:
            raise ValueError("a network needs at least one neuron")
        self.random_matrix = (
            None if random_matrix is None else jnp.asarray(random_matrix)
        )
        if self.random_matrix is not None and self.random_matrix.shape != (
            self.m.shape[0], self.m.shape[0]
        ):
            raise ValueError("random_matrix must have shape (neurons, neurons)")
        if tau <= 0:
            raise ValueError("tau must be positive")
        if random_matrix is None and g != 0:
            raise ValueError("nonzero g requires a random_matrix")
        self.g = g
        self.tau = tau

    def recurrent(self, rates: Array) -> Array:
        """Apply connectivity without materializing the structured matrix."""
        if self.m.shape[1] == 0:
            result = jnp.zeros_like(rates)
        else:
            kappa = dot("... i, i r -> ... r", rates, self.n) / self.m.shape[0]
            result = dot("... r, i r -> ... i", kappa, self.m)
        if self.random_matrix is not None:
            result = result + self.g * (rates @ self.random_matrix.T)
        return result

    def derivative(self, state: Array, inputs: Array) -> Array:
        return (-state + self.recurrent(jnp.tanh(state)) + inputs) / self.tau

    def dense_connectivity(self) -> Array:
        result = self.m @ self.n.T / self.m.shape[0]
        if self.random_matrix is not None:
            result = result + self.g * self.random_matrix
        return result


def gaussian_network(
    key: Array,
    size: int,
    rank: int = 1,
    *,
    g: float = 0.0,
    mean_m: float = 0.0,
    mean_n: float = 0.0,
    std_m: float = 1.0,
    std_n: float = 1.0,
    correlation: float = 0.0,
    tau: float = 1.0,
) -> RateNetwork:
    """Independent Gaussian vector pairs with specified population moments.

    Moments are distributional, not forced to hold exactly at finite N.
    No dense random matrix is allocated when g=0.
    """
    if size < 1 or rank < 0:
        raise ValueError("size must be positive and rank nonnegative")
    if not -1 <= correlation <= 1 or std_m < 0 or std_n < 0:
        raise ValueError("invalid Gaussian standard deviation or correlation")
    km, kn, kc = jax.random.split(key, 3)
    a = jax.random.normal(km, (size, rank))
    b = jax.random.normal(kn, (size, rank))
    m = mean_m + std_m * a
    n = mean_n + std_n * (
        correlation * a + jnp.sqrt(1 - correlation**2) * b
    )
    chi = None if g == 0 else jax.random.normal(kc, (size, size)) / jnp.sqrt(size)
    return RateNetwork(m, n, chi, g=g, tau=tau)


class Simulation(eqx.Module):
    times: Array
    states: Array

    @property
    def rates(self) -> Array:
        return jnp.tanh(self.states)


@eqx.filter_jit
def simulate(
    network: RateNetwork,
    initial_state: Array,
    inputs: Array,
    *,
    dt: float = 0.1,
    method: Literal["rk4", "euler"] = "rk4",
    sample_every: int = 1,
) -> Simulation:
    """Integrate piecewise-constant inputs of shape (steps, neurons).

    Includes t=0 and the final state, even for incomplete sampling blocks.
    Only sampled states are retained: working trajectory memory is O(TN/s).
    The input in row k is held fixed on [k dt, (k+1) dt).
    """
    dtype = jnp.result_type(initial_state, inputs, network.m, jnp.float32)
    initial_state = jnp.asarray(initial_state, dtype=dtype)
    inputs = jnp.asarray(inputs, dtype=dtype)
    if initial_state.shape != (network.m.shape[0],):
        raise ValueError("initial_state must have shape (neurons,)")
    if inputs.ndim != 2 or inputs.shape[1] != network.m.shape[0]:
        raise ValueError("inputs must have shape (steps, neurons)")
    if dt <= 0 or sample_every < 1:
        raise ValueError("dt and sample_every must be positive")
    if method not in ("rk4", "euler"):
        raise ValueError("method must be 'rk4' or 'euler'")

    def step(state, drive):
        k1 = network.derivative(state, drive)
        if method == "euler":
            state = state + dt * k1
        else:
            k2 = network.derivative(state + dt * k1 / 2, drive)
            k3 = network.derivative(state + dt * k2 / 2, drive)
            k4 = network.derivative(state + dt * k3, drive)
            state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state, None

    def block(state, drives):
        state, _ = jax.lax.scan(step, state, drives)
        return state, state

    steps = inputs.shape[0]
    full = steps // sample_every * sample_every
    blocks = inputs[:full].reshape((-1, sample_every, inputs.shape[1]))
    state, samples = jax.lax.scan(block, initial_state, blocks)
    states = jnp.concatenate((initial_state[None], samples), axis=0)
    times = jnp.arange(full // sample_every + 1) * (dt * sample_every)
    if full != steps:
        state, _ = jax.lax.scan(step, state, inputs[full:])
        states = jnp.concatenate((states, state[None]), axis=0)
        times = jnp.concatenate((times, jnp.asarray([steps * dt])))
    return Simulation(times, states)
