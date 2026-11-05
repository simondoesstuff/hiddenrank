# Task recreations

| Run | Recipe | Recreated panels |
| --- | --- | --- |
| `go_nogo` | Eq. 145, S=2 | Readout, population trajectory, mixture generalization |
| `detection` | Eq. 151, rm=rn=2 | Noisy detection, psychometric response |
| `context` | Released Fig. 5 recipe, m-context=1.6, n-context=3 | Contextual output and trajectories |
| `integration` | Same convention, m-context=1.38, n-context=3 | Outputs, population/scaffold trajectories, response maps |
| `ring` | Eq. 156, overlap=1.8 | Relaxation toward a rank-two ring |
| `detection-theory` | Stationary fold and resting collective eigenvalue | Fig. 4F threshold–timescale curve |

## Protocol

- Contextual tasks default to `architecture=supplement`. `architecture=paper` retains the literal printed coefficients, which produce different stimulus gating. See [source discrepancy](protocol.md).
- White stimulus noise uses **sigma √dt** increments: Euler-Maruyama, sigma=.4 for detection and .3 for integration. Deterministic tasks use RK4.
- A 10-unit free transient is discarded. Detection has 7 baseline + 22 stimulus units; contextual tasks have 15 baseline + 85 stimulus units. Fig. 6 timing is reconstructed from the plotted 0–100 axis, not recovered source code.
- Detection's independent quenched input starts with the stimulus. Context begins after the free transient and persists through the trial.
- Decisions compare offset endpoints with half the **theoretical** upper resting readout, offset by the same baseline. Go/Nogo instead uses the theoretical driven Go state. The exact reference operating point is not specified in the released classifier; ours is explicit.
- iid patterns are available at published N. Preview orthogonalization enforces limiting covariances in small networks; metadata records it.
- No failed trials are filtered. Bands show realization SEM; psychometric errors show binomial standard errors.

## Saved observables

NPZ archives contain readouts, input/readout projections, recurrent overlaps, scaffold coordinates, four single-unit traces, endpoints, decisions, theoretical references, and protocol metadata. Dense matrices and full population trajectories are not stored.

Each random network gets a reproducible realization ID. Patterns stay fixed while connectivity and stimulus noise vary. Trial checkpoints can be aggregated or extended without repeating completed simulations.

PCA/connectivity-correlation panels and theoretical transient overlays remain outside the current recreations. The scaffold circle is a population-limit reference, not a fitted attractor for the full contextual task.
