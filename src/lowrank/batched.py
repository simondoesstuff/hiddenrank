"""Memory-bounded, batched rate dynamics with low-dimensional input drives."""
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .network import RateNetwork


class Measurements(eqx.Module):
    times: Array
    readout: Array
    projections: Array
    kappa: Array
    single_units: Array


@eqx.filter_jit
def simulate_driven(network: RateNetwork, initial_states: ArrayLike, coefficients: ArrayLike,
                    noise_increments: ArrayLike, directions: ArrayLike, readout: ArrayLike,
                    projection_vectors: ArrayLike, *, dt: float = .1,
                    method: Literal['rk4', 'euler'] = 'rk4',
                    sample_every: int = 2) -> Measurements:
    """Integrate independent conditions together, sharing one quenched network.

    States are (conditions,N); coefficients/noise increments are (steps,conditions,D).
    Noise increments already include sigma*sqrt(dt). Euler plus these increments
    is Euler-Maruyama for additive white input. Only observables are persisted;
    storage is independent of N apart from the current batch state and matrix.
    """
    dtype = jnp.result_type(network.m, jnp.float32)
    states = jnp.asarray(initial_states, dtype=dtype)
    coefficients = jnp.asarray(coefficients, dtype=dtype)
    noise_increments = jnp.asarray(noise_increments, dtype=dtype)
    directions = jnp.asarray(directions, dtype=dtype)
    readout = jnp.asarray(readout, dtype=dtype)
    projection_vectors = jnp.asarray(projection_vectors, dtype=dtype)
    if states.ndim != 2 or states.shape[1] != network.m.shape[0]:
        raise ValueError('initial_states must have shape (conditions,neurons)')
    if coefficients.ndim != 3 or coefficients.shape != noise_increments.shape:
        raise ValueError('coefficient and increment arrays must share (steps,conditions,directions) shape')
    if coefficients.shape[1] != states.shape[0] or directions.shape != (states.shape[1], coefficients.shape[2]):
        raise ValueError('drive dimensions do not match states and directions')
    if dt <= 0 or sample_every < 1 or method not in ('rk4', 'euler'):
        raise ValueError('invalid integration settings')

    def observe(x):
        rates = jnp.tanh(x)
        return (rates @ readout / x.shape[-1], x @ projection_vectors / x.shape[-1],
                rates @ network.n / x.shape[-1], x[:, :min(4, x.shape[-1])])

    def step(x, drives):
        coefficient, increment = drives
        inputs = coefficient @ directions.T
        k1 = network.derivative(x, inputs)
        if method == 'euler':
            x = x + dt * k1
        else:
            k2 = network.derivative(x + dt*k1/2, inputs)
            k3 = network.derivative(x + dt*k2/2, inputs)
            k4 = network.derivative(x + dt*k3, inputs)
            x = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        # Input units in the model are currents; tau scales the additive drive.
        return x + increment @ directions.T / network.tau, None

    def block(x, drives):
        x, _ = jax.lax.scan(step, x, drives)
        return x, observe(x)

    steps = coefficients.shape[0]
    full = steps // sample_every * sample_every
    shape = (-1, sample_every, *coefficients.shape[1:])
    states_final, measurements = jax.lax.scan(block, states,
        (coefficients[:full].reshape(shape), noise_increments[:full].reshape(shape)))
    initial_measurements = observe(states)
    measurements = tuple(jnp.concatenate((start[None], sampled), axis=0)
                         for start, sampled in zip(initial_measurements, measurements))
    times = jnp.arange(full//sample_every+1)*dt*sample_every
    if full != steps:
        states_final, _ = jax.lax.scan(step, states_final,
                                      (coefficients[full:], noise_increments[full:]))
        measurements = tuple(jnp.concatenate((sampled, last[None]), axis=0)
                             for sampled, last in zip(measurements, observe(states_final)))
        times = jnp.concatenate((times, jnp.asarray([steps*dt])))
    return Measurements(times, *measurements)
