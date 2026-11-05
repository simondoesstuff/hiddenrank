"""Seeded finite-network recreations of task panels in Figures 3–6."""

from pathlib import Path
import json

import numpy as np

from . import tasks


def _conditions(name: str):
    """Return stimulus pairs, contexts, noise, labels and curve abscissae."""
    if name == "go_nogo":
        alpha = np.linspace(0, 1, 9)
        # First two entries are the original A/B discrimination task; remaining
        # entries use a distinct network with n=A+B (Eq. 149).
        return np.array([[1., 0.], [0., 1.]] + [[np.sqrt(a), np.sqrt(1-a)] for a in alpha]), np.zeros(11, int), 0., ["Go A", "Nogo B"] + [f"mixture {a:.2f}" for a in alpha], np.r_[0., 1., alpha]
    if name == "detection":
        values = np.linspace(0, 1, 11)
        return np.stack([values, np.zeros_like(values)], axis=1), np.zeros(11, int), .4, [f"c={v:.1f}" for v in values], values
    if name in ("context", "integration"):
        if name == "context":
            stimuli = np.array([[1., 0.], [0., 1.]])
            labels = ["A", "B"]
        else:
            values = np.linspace(0., 1., 7)
            stimuli = np.array([(a, b) for a in values for b in values])
            labels = [f"A={a:.2f}, B={b:.2f}" for a, b in stimuli]
        return np.tile(stimuli, (2, 1)), np.repeat([0, 1], len(stimuli)), .3 if name == "integration" else 0., [f"context {c}: {label}" for c in "AB" for label in labels], np.arange(2 * len(stimuli))
    raise ValueError(f"Unknown experiment: {name}")


def run_experiment(name: str, output: str | Path, *, size: int = 256,
                   trials: int = 4, seed: int = 0, dt: float = .1,
                   duration: float = 20., orthogonal: bool = False) -> Path:
    """Save compact NPZ measurements; sizes/repetitions are workflow-controlled.

    All trials are saved (including errors). Input noise is sampled independently
    at each integration step, with the stated discrete-time standard deviation.
    """
    import jax.numpy as jnp
    from .network import RateNetwork, simulate

    if size < 9 or trials < 1 or dt <= 0 or duration <= 0:
        raise ValueError("Require size>=9, trials>=1, dt>0, duration>0")
    constructors = {"go_nogo": tasks.go_nogo, "detection": tasks.detection,
                    "context": tasks.contextual,
                    "integration": lambda n, s, **kw: tasks.contextual(n, s, integration=True, **kw)}
    if name == "ring":
        return _run_ring(output, size=size, trials=trials, seed=seed, dt=dt,
                         duration=duration, orthogonal=orthogonal)
    if name not in constructors:
        raise ValueError(f"Unknown experiment: {name}")
    task = constructors[name](size, seed, orthogonal=orthogonal)
    stimulus, contexts, sigma, labels, axis = _conditions(name)
    baseline_steps = max(1, round(15 / dt))
    stimulus_steps = max(1, round(duration / dt))
    recovery_steps = max(1, round(10 / dt))
    steps = baseline_steps + stimulus_steps + recovery_steps
    stride = max(1, round(.2 / dt))
    rng = np.random.default_rng(seed + 1000)
    traces, projections, endpoints, baselines, upper_states = [], [], [], [], []
    # The shape stays constant across conditions, allowing one JIT compilation.
    for trial in range(trials):
        chi = rng.normal(size=(size, size)).astype(np.float32)
        chi *= np.float32(1 / np.sqrt(size))
        trial_traces, trial_proj, trial_end, trial_baseline, trial_upper = [], [], [], [], []
        for condition, (ab, context) in enumerate(zip(stimulus, contexts, strict=True)):
            n = task.n.copy()
            if name == "go_nogo" and condition >= 2:
                n = task.directions.sum(axis=1)[:, None]
            network = RateNetwork(jnp.asarray(task.m, dtype=jnp.float32),
                                  jnp.asarray(n, dtype=jnp.float32),
                                  random_matrix=jnp.asarray(chi), g=task.g)
            coefficients = np.zeros((steps, len(task.labels)), np.float32)
            if name in ("context", "integration"):
                gating = [.08, -.18 if name == "integration" else -.14]
                if context == 1:
                    gating = gating[::-1]
                coefficients[:, 2:] = gating
            interval = slice(baseline_steps, baseline_steps + stimulus_steps)
            channels = 1 if name == "detection" else 2
            coefficients[interval, :channels] = ab[:channels]
            if sigma:
                coefficients[interval, :channels] += rng.normal(0, sigma, (stimulus_steps, channels))
            if name == "detection":
                coefficients[:, 2] = 1.  # paper's quenched orthogonal input
            inputs = jnp.asarray(coefficients @ task.directions.T)
            initial = np.zeros(size) if name == "go_nogo" else -task.m.sum(axis=1)
            result = simulate(network, jnp.asarray(initial, dtype=jnp.float32), inputs,
                              dt=dt, sample_every=stride)
            states = np.asarray(result.states)
            times = np.asarray(result.times)
            readout = np.tanh(states) @ task.readout / size
            baseline_index = int(np.argmin(abs(times - baseline_steps * dt)))
            endpoint_index = int(np.argmin(abs(times - (baseline_steps + stimulus_steps) * dt)))
            offset = readout[baseline_index]
            trial_baseline.append(offset)
            trial_traces.append(readout - offset)
            if name == "go_nogo":
                projection_vectors = np.stack([task.directions[:, 0], task.readout], axis=1)
            elif name == "detection":
                projection_vectors = np.stack([task.directions[:, 0], task.m[:, 0]], axis=1)
            else:
                projection_vectors = np.column_stack([task.directions[:, :2], task.readout])
            trial_proj.append(states @ projection_vectors / size)
            trial_end.append(readout[endpoint_index] - offset)
            # Independently measure positive resting attractor at the same context.
            rest_inputs = jnp.asarray(coefficients[:baseline_steps] @ task.directions.T)
            if name == "go_nogo":
                # This architecture has no positive spontaneous attractor.
                # Calibrate its decision against a driven Go reference instead.
                rest_inputs = jnp.broadcast_to(jnp.asarray(task.directions[:, 0]), (baseline_steps, size))
            upper = simulate(network, jnp.asarray(task.m.sum(axis=1), dtype=jnp.float32),
                             rest_inputs, dt=dt, sample_every=baseline_steps)
            trial_upper.append(float(np.tanh(np.asarray(upper.states[-1])) @ task.readout / size - offset))
        traces.append(trial_traces)
        projections.append(trial_proj)
        endpoints.append(trial_end)
        baselines.append(trial_baseline)
        upper_states.append(trial_upper)
    endpoints = np.asarray(endpoints)
    upper_states = np.asarray(upper_states)
    # A measured upper attractor is used instead of an unimplemented theory fit.
    # This is explicitly recorded in metadata and figures.
    threshold = upper_states / 2
    decisions = endpoints > threshold
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(name=name, size=size, trials=trials, seed=seed, dt=dt,
                    duration=duration, sigma=sigma, g=task.g,
                    orthogonal_patterns=orthogonal,
                    decision_threshold=("half measured driven-Go-minus-baseline readout" if name == "go_nogo"
                                        else "half measured upper-minus-baseline resting readout"),
                    baseline_duration=baseline_steps * dt,
                    stimulus_end=(baseline_steps + stimulus_steps) * dt)
    np.savez_compressed(path, experiment=name, times=times, traces=np.asarray(traces),
                        projections=np.asarray(projections), endpoints=endpoints,
                        baselines=np.asarray(baselines), upper_states=upper_states,
                        decisions=decisions, stimulus=stimulus, contexts=contexts,
                        axis=axis, labels=np.asarray(labels), metadata=json.dumps(metadata))
    return path


def _run_ring(output, *, size, trials, seed, dt, duration, orthogonal):
    import jax.numpy as jnp
    from .network import RateNetwork, simulate

    task = tasks.ring(size, seed, orthogonal=orthogonal)
    network = RateNetwork(jnp.asarray(task.m, dtype=jnp.float32), jnp.asarray(task.n, dtype=jnp.float32))
    angles = np.linspace(0, 2 * np.pi, max(8, trials), endpoint=False)
    trajectories = []
    for angle in angles:
        initial = task.m @ np.array([np.cos(angle), np.sin(angle)])
        result = simulate(network, jnp.asarray(initial, dtype=jnp.float32),
                          jnp.zeros((round(duration / dt), size)), dt=dt,
                          sample_every=max(1, round(.2 / dt)))
        trajectories.append(np.tanh(np.asarray(result.states)) @ task.n / size)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, experiment="ring", times=np.asarray(result.times), trajectories=np.asarray(trajectories),
                        angles=angles, metadata=json.dumps(dict(name="ring", size=size,
                        trials=len(angles), seed=seed, dt=dt, duration=duration,
                        orthogonal_patterns=orthogonal, g=0., strength=1.8)))
    return path
