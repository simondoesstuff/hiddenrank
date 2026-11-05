# Reproduction guide

| Command | Output | Scope |
| --- | --- | --- |
| `just figures` | `data/preview/index.html` | Small networks; six figures + Fig. 4F |
| `just pilot` | `data/pilot/index.html` | N=7500, one iid contextual realization |
| `just replication` | `data/replication/index.html` | N=7500, fifty realizations; full contextual stimulus grids |
| `just paper` | `data/paper/index.html` | Published N and ensemble counts for Figures 1–4 |

Each gallery links PNG, SVG and NPZ. `just dry-run` inspects the preview DAG. Pass a config file and explicit target to Snakemake to run one figure.

## Scale

| Figure | Preview N / realizations | Paper N / realizations |
| --- | --- | --- |
| 1 | 512 / 2 | 5000 / 15; duration 800 |
| 2 | 512 / 2 | 3500 / 20 |
| 3 | 1024 / 2 | 2500 / 20 |
| 4 | 1024 / 2 | 2500 / 100 |
| 5–6 | 1024 / 2 | 7500 / 50 |

- Paper Fig. 5/6 trajectories used smaller populations than performance. Our panels reuse the performance population and retain all errors.
- The `replication` profile uses fifty contextual connectivity realizations, matching the published performance ensemble size. Protocol differences remain explicit.
- Timing, input noise, coefficient-order discrepancies and classifier convention: [protocol](protocol.md).
- Fig. 1 theory is independent of finite-network traces. Fig. 2 averages realizations before z-scored PCA; solid/dashed mean-field branches are stable/unstable.

## Runtime and resuming

- Snakemake stores each task realization in `data/<profile>/replicates/<figure>/<id>.npz`, then aggregates it.
- Increasing `trials` reuses completed IDs. Override `trials` to extend the contextual ensemble further. Seeds are independent of execution order, shard boundaries and ensemble size.
- Conditions share the same quenched matrix and run as a batch. Only compact observables are retained. Memory: O(N² + conditions × N).
- Gaussian patterns stay fixed across connectivity realizations. No successful-seed selection is performed.
- Use `just replication 2` for two concurrent jobs. The CPU implementation uses JAX; no Rust build is needed.
- `data/` is ignored by Git. Code and configuration are committed; artifacts are reproduced locally.

The just recipes clear inherited `PYTHONPATH` to prevent Nix's Python packages from shadowing the uv environment.
