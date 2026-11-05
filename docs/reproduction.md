# Reproduction guide

- `just figures`: small local run; outputs `data/preview/index.html`, PNG/SVG and NPZ.
- `just paper`: iid patterns at published N and realization counts. Untested at full scale; not a guarantee of published error bars.
- Edit `config/preview.yaml` or supply `--configfile` to Snakemake. Separate `profile` values keep results apart.
- `just figures 2`: two concurrent jobs. Start with one on memory-constrained machines.

## Protocol

| Setting | Preview | Paper-scale configuration |
| --- | --- | --- |
| Fig. 1 N / trials / duration | 512 / 2 / 120 | 5000 / 15 / 800 |
| Fig. 2 N / trials | 512 / 2 | 3500 / 20 |
| Fig. 3 N / trials | 1024 / 2 | 2500 / 20 |
| Fig. 4 N / trials | 1024 / 2 | 2500 / 100 |
| Fig. 5–6 N / trials | 1024 / 2 | 7500 / 50 |
| Task pattern sampling | Centered, orthogonalized Gaussian draws | iid Gaussian |
| Integration | RK4, dt=0.1 | RK4, dt=0.1 |

- Published Figure 4 uses 4 realizations for trajectories and 100 for performance; Figures 5–6 use smaller populations for trajectories. Here each figure reuses its performance ensemble for all panels.
- Task stimuli last 20 time constants after a 15-unit baseline, followed by 10 recovery units. These choices are explicit, not claimed as recovered original source-code settings.
- Fig. 1 theory uses population-limit Gaussian DMF, independent of finite-network traces. Fig. 2 averages network realizations before z-scored PCA; plotted coordinates use least-squares coefficients in the input/m basis.
- Fig. 2 branches are stationary mean-field roots. Solid branches are stable; dashed branches are unstable. Finite-size sweeps are not fitted to them.
- Fig. 1 spontaneous statistics discard the first half as transient. At preview size/duration, chaotic-looking traces can retain slow finite-size drift.
- Task noise has fixed standard deviation per integration step. Preserve dt when comparing protocols.

## Storage and runtime

- Only the random connectivity is dense: O(N²) float32. Structured recurrence stays factored, O(Nr).
- Measurements are compressed; full dense connectivity and task state trajectories are not persisted.
- One environment serves all worktrees. No duplicated environments, model weights, or compiled Rust artifacts.
- Snakemake tracks simulation source, parameters, plotting source, and gallery dependencies independently.
- `data/` is ignored by Git. Recreate artifacts from committed code/configuration; no push is required.

If a Nix shell injects a different Python's packages via `PYTHONPATH`, the just recipes clear it before calling uv. Snakemake comes from the uv environment to keep its binary dependencies consistent.
