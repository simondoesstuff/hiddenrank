"""Memory-bounded RK4 integration of spontaneous rank-one networks."""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=('steps', 'dt', 'sample_every'))
def simulate_spontaneous(m, n, chi, gains, *, steps: int, dt: float, sample_every: int):
    """Share one bulk matrix across conditions; retain traces and tail moments.

    The second half of the sampled trajectory uses Welford moments. No full
    population trajectory or zero-input array is materialized.
    """
    m, n, chi, gains = (jnp.asarray(a, dtype=jnp.float32) for a in (m, n, chi, gains))
    size = m.shape[1]
    full, remainder = divmod(steps, sample_every)
    samples = full + 1 + int(remainder != 0)
    tail_start = samples // 2

    def derivative(x):
        rates = jnp.tanh(x)
        overlap = jnp.mean(rates * n, axis=1)
        return -x + gains[:, None] * (rates @ chi.T) + m * overlap[:, None]

    def step(_, x):
        k1 = derivative(x)
        k2 = derivative(x + dt * k1 / 2)
        k3 = derivative(x + dt * k2 / 2)
        k4 = derivative(x + dt * k3)
        return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def record(carry, sample_index, count_steps):
        x, count, mean, m2, overlap_sum = carry
        x = jax.lax.fori_loop(0, count_steps, step, x)
        active = sample_index >= tail_start
        next_count = count + active.astype(jnp.int32)
        delta = x - mean
        next_mean = mean + jnp.where(active, delta / jnp.maximum(next_count, 1), 0.)
        next_m2 = m2 + jnp.where(active, delta * (x - next_mean), 0.)
        overlap_sum += jnp.where(active, jnp.mean(jnp.tanh(x)*n, axis=1), 0.)
        return (x, next_count, next_mean, next_m2, overlap_sum), x[:, :6]

    carry = (m, jnp.asarray(0), jnp.zeros_like(m), jnp.zeros_like(m), jnp.zeros(m.shape[0]))
    carry, first = record(carry, jnp.asarray(0), 0)
    carry, traces = jax.lax.scan(lambda c, i: record(c, i, sample_every), carry,
                                 jnp.arange(1, full+1))
    traces = jnp.concatenate((first[None], traces), axis=0)
    times = jnp.arange(full+1) * dt * sample_every
    if remainder:
        carry, last = record(carry, jnp.asarray(samples-1), remainder)
        traces = jnp.concatenate((traces, last[None]), axis=0)
        times = jnp.concatenate((times, jnp.asarray([steps*dt])))
    _, count, mean, m2, overlap_sum = carry
    statistics = jnp.stack((overlap_sum/count, jnp.var(mean, axis=1),
                            jnp.mean(m2/count, axis=1)), axis=1)
    return times, jnp.swapaxes(traces, 0, 1), statistics
