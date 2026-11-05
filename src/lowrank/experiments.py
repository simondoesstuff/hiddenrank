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
                   duration: float = 20., orthogonal: bool = False,
                   trial_start: int = 0, condition_set: str = "full", architecture: str = "supplement",
                   baseline_duration: float | None = None,
                   transient_duration: float = 10., recovery_duration: float = 0.) -> Path:
    """Save seeded observables, batching conditions against one random network.

    The pattern seed stays fixed across realizations. Each realization has its
    own SeedSequence([seed, trial_id, 1000]), so sharding/resuming preserves draws.
    No failed trials are filtered. White input noise uses Euler-Maruyama.
    """
    import jax.numpy as jnp
    from .network import RateNetwork
    from .batched import simulate_driven
    from .protocol import context_strengths, theoretical_upper_readout

    if size < 9 or trials < 1 or dt <= 0 or duration <= 0 or trial_start < 0:
        raise ValueError("Require size>=9, trials>=1, dt>0, duration>0, trial_start>=0")
    if transient_duration < 0 or recovery_duration < 0:
        raise ValueError("phase durations must be nonnegative")
    constructors = {"go_nogo": tasks.go_nogo, "detection": tasks.detection,
                    "context": tasks.contextual,
                    "integration": lambda n, s, **kw: tasks.contextual(n, s, integration=True, **kw)}
    if name == "ring":
        return _run_ring(output, size=size, trials=trials, seed=seed, dt=dt,
                         duration=duration, orthogonal=orthogonal)
    if name not in constructors:
        raise ValueError(f"Unknown experiment: {name}")
    if condition_set not in {"full", "representative"}:
        raise ValueError("condition_set must be full or representative")
    constructor_kwargs = {"architecture": architecture} if name in ("context", "integration") else {}
    task = constructors[name](size, seed, orthogonal=orthogonal, **constructor_kwargs)
    stimulus, contexts, sigma, labels, axis = _conditions(name)
    if name == "integration" and condition_set == "representative":
        stimulus = np.tile([[.6, .1], [.1, .6]], (2, 1))
        contexts = np.repeat([0, 1], 2)
        labels = [f"context {'AB'[c]}: A={a:.1f}, B={b:.1f}"
                  for (a, b), c in zip(stimulus, contexts, strict=True)]
        axis = np.arange(4)
    count = len(stimulus)
    if baseline_duration is None:
        baseline_duration = 7. if name == "detection" else 15.
    if baseline_duration <= 0:
        raise ValueError("baseline_duration must be positive")
    transient_steps = round(transient_duration / dt)
    baseline_steps = max(1, round(baseline_duration / dt))
    stimulus_steps = max(1, round(duration / dt))
    recovery_steps = round(recovery_duration / dt)
    onset = transient_steps + baseline_steps
    end = onset + stimulus_steps
    steps = end + recovery_steps
    stride = max(1, round(.2 / dt))
    coefficients = np.zeros((steps, count, len(task.labels)), np.float32)
    if name in ("context", "integration"):
        coefficients[transient_steps:, :, 2:] = np.array([context_strengths(name, int(c)) for c in contexts])
    channels = 1 if name == "detection" else 2
    coefficients[onset:end, :, :channels] = stimulus[None, :, :channels]
    if name == "detection":
        coefficients[onset:end, :, 2] = 1.  # Quenched h exists only during stimulation.
    if name == "go_nogo":
        projection_vectors = np.stack([task.directions[:, 0], task.readout], axis=1)
    elif name == "detection":
        projection_vectors = np.stack([task.directions[:, 0], task.m[:, 0]], axis=1)
    else:
        projection_vectors = np.column_stack([task.directions[:, :2], task.readout])
    base_projection_count = projection_vectors.shape[1]
    if name in ("context", "integration"):
        scaffold = task.m - .6*task.readout[:, None]
        scaffold = scaffold * (size / np.sum(scaffold**2, axis=0))
        projection_vectors = np.column_stack([projection_vectors, scaffold])
    # Predict the upper raw resting readout independently of each finite network.
    reference = np.array([theoretical_upper_readout(name, context=int(c) if name in
                          ("context", "integration") else None, architecture=architecture) for c in contexts])
    traces, projections, kappas, units, endpoints, baselines = [], [], [], [], [], []
    scaffold_coordinates = []
    for trial_id in range(trial_start, trial_start + trials):
        rng = np.random.default_rng(np.random.SeedSequence([seed, trial_id, 1000]))
        chi = rng.standard_normal((size, size), dtype=np.float32)
        chi *= np.float32(1 / np.sqrt(size))
        increments = np.zeros_like(coefficients)
        if sigma:
            increments[onset:end, :, :channels] = rng.normal(
                0, sigma*np.sqrt(dt), (stimulus_steps, count, channels))
        groups = [np.arange(2), np.arange(2, count)] if name == "go_nogo" else [np.arange(count)]
        observables = []
        for group in groups:
            n = task.n if name != "go_nogo" or group[0] == 0 else task.directions.sum(axis=1)[:, None]
            network = RateNetwork(jnp.asarray(task.m, dtype=jnp.float32),
                                  jnp.asarray(n, dtype=jnp.float32), jnp.asarray(chi), g=task.g)
            initial = np.zeros(size) if name == "go_nogo" else -(task.m + n).sum(axis=1)
            measured = simulate_driven(network, np.broadcast_to(initial, (len(group), size)),
                coefficients[:, group], increments[:, group], task.directions, task.readout,
                projection_vectors, dt=dt, method="euler" if sigma else "rk4", sample_every=stride)
            observables.append(measured)
        raw_times = np.asarray(observables[0].times)
        keep = raw_times >= transient_steps*dt - 1e-5
        times = raw_times[keep] - transient_steps*dt
        def collect(attribute):
            return np.concatenate([np.asarray(getattr(item, attribute)) for item in observables], axis=1)[keep].swapaxes(0, 1)
        raw_readout = collect("readout")
        baseline_index = int(np.argmin(abs(times - baseline_steps*dt)))
        endpoint_index = int(np.argmin(abs(times - (baseline_steps+stimulus_steps)*dt)))
        offset = raw_readout[:, baseline_index]
        baselines.append(offset)
        traces.append(raw_readout - offset[:, None])
        endpoints.append(raw_readout[:, endpoint_index] - offset)
        coordinates = collect("projections")
        projections.append(coordinates[..., :base_projection_count])
        if name in ("context", "integration"):
            scaffold_coordinates.append(coordinates[..., base_projection_count:])
        kappas.append(collect("kappa"))
        units.append(collect("single_units"))
    endpoints = np.asarray(endpoints)
    baselines = np.asarray(baselines)
    upper_states = reference[None, :] - baselines
    threshold = upper_states / 2
    decisions = endpoints > threshold
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(name=name, size=size, trials=trials, seed=seed, dt=dt,
                    duration=stimulus_steps*dt, sigma=sigma, g=task.g,
                    orthogonal_patterns=orthogonal, architecture=architecture, protocol_version=2,
                    trial_start=trial_start, condition_set=condition_set,
                    decision_threshold=("half theoretical driven-Go-minus-baseline readout" if name == "go_nogo"
                                        else "half theoretical upper-resting-minus-baseline readout"),
                    integrator="Euler-Maruyama" if sigma else "RK4",
                    noise_convention="sigma sqrt(dt) additive increments",
                    transient_duration=transient_steps*dt, recovery_duration=recovery_steps*dt,
                    baseline_duration=baseline_steps*dt,
                    stimulus_end=(baseline_steps+stimulus_steps)*dt,
                    timing_source="Fig4/5 public example; Fig6 reconstructed from the displayed 0-100 axis",
                    scaffold_coordinates="x projections onto m_j-bm*w, normalized by empirical squared norm",
                    original_commit="35211877f641cd16ec239dc07330478d5b118175")
    np.savez_compressed(path, experiment=name, times=times, traces=np.asarray(traces),
                        projections=np.asarray(projections), scaffold_coordinates=np.asarray(scaffold_coordinates), kappa=np.asarray(kappas),
                        single_units=np.asarray(units), endpoints=endpoints, baselines=baselines,
                        upper_states=upper_states, theoretical_upper_raw=reference,
                        decisions=decisions, stimulus=stimulus, contexts=contexts,
                        axis=axis, labels=np.asarray(labels), metadata=json.dumps(metadata),
                        trial_ids=np.arange(trial_start, trial_start+trials))
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


def aggregate_experiments(inputs: list[str | Path], output: str | Path) -> Path:
    """Assemble independent realization checkpoints without changing their draws."""
    if not inputs:
        raise ValueError('at least one realization is required')
    datasets = []
    for source in inputs:
        with np.load(source, allow_pickle=False) as archive:
            datasets.append({key: archive[key] for key in archive.files})
    trial_keys = {'traces', 'projections', 'kappa', 'single_units', 'endpoints',
                  'baselines', 'upper_states', 'decisions', 'trial_ids'}
    if datasets[0]['scaffold_coordinates'].size:
        trial_keys.add('scaffold_coordinates')
    output_data = datasets[0].copy()
    metadata = json.loads(str(output_data['metadata']))
    protocol = {key: value for key, value in metadata.items() if key not in {'trials', 'trial_start'}}
    for dataset in datasets[1:]:
        other = json.loads(str(dataset['metadata']))
        if {key: value for key, value in other.items() if key not in {'trials', 'trial_start'}} != protocol:
            raise ValueError('realization protocols differ')
        for key in output_data.keys() - trial_keys - {'metadata'}:
            if not np.array_equal(output_data[key], dataset[key]):
                raise ValueError(f'realization coordinate {key} differs')
    for key in trial_keys:
        output_data[key] = np.concatenate([dataset[key] for dataset in datasets], axis=0)
    ids = output_data['trial_ids']
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate realization IDs')
    order = np.argsort(ids)
    for key in trial_keys:
        output_data[key] = output_data[key][order]
    metadata['trials'] = len(ids)
    metadata['trial_start'] = int(ids.min())
    output_data['metadata'] = json.dumps(metadata)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output_data)
    return path
