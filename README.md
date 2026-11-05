# Low-rank recurrent networks

Recreate key figures from **Mastrogiuseppe & Ostojic (2018)**, *Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks*. [Paper](https://doi.org/10.1016/j.neuron.2018.07.003) · [Local PDF](docs/ostojic.pdf)

```sh
uv sync
just figures                 # Snakemake; six figures, PNG + SVG + gallery
just test                    # basedpyright + pytest
just replication             # N=7500 iid contextual figures, 10 realizations
```

Open the [preview gallery](data/preview/index.html) or [paper-size contextual gallery](data/replication/index.html) after the corresponding run. Compressed simulation measurements live in `data/preview/simulations/`; plots are in `data/preview/figures/`. Runs are seeded and Snakemake resumes completed stages.

| Figure | Recreated content |
| --- | --- |
| [1](data/preview/figures/figure1_spontaneous.png) | Spontaneous regimes, overlap/variance curves, stationary and chaotic traces |
| [2](data/preview/figures/figure2_inputs.png) | Step responses, low-dimensional trajectories, PCA, input-response branches |
| [3](data/preview/figures/figure3_go_nogo.png) | Go/Nogo readout, population geometry, mixture generalization |
| [4](data/preview/figures/figure4_detection.png) | Noisy detection, bistable switching, psychometric response |
| [5](data/preview/figures/figure5_context.png) | Context-dependent discrimination and state trajectories |
| [6](data/preview/figures/figure6_integration.png) | Context-dependent integration, population/scaffold trajectories, response maps |
| [4F](data/preview/figures/figure4_threshold_timescale.png) | Detection threshold and effective timescale near the bifurcation |

The **preview** uses N=512 for Figs. 1–2 and N=1024 with orthogonalized Gaussian patterns for Figs. 3–6, two realizations each. Orthogonalization enforces the paper's limiting pattern covariances in smaller networks. The **replication** configuration uses N=7500, iid Gaussian patterns and ten realizations for Figures 5–6. The **paper** configuration expands to the published population/ensemble sizes for every figure.

```sh
just paper                  # original N / realization counts; see config/paper.yaml
# Extend the existing contextual checkpoints from 10 to 50 realizations:
uv run snakemake -s workflow/Snakefile --configfile config/replication.yaml --config trials=50 --cores 2
just dry-run
# One figure, or custom settings:
uv run snakemake -s workflow/Snakefile --cores 1 data/preview/figures/figure3_go_nogo.png
uv run python -m lowrank run ring data/ring.npz --size 1024 --orthogonal-patterns
uv run python -m lowrank plot data/ring.npz data/ring.png
```

These are numerical recreations of selected panels, not exact copies. The contextual recipe follows the released supplementary code: it reverses two coefficients relative to the printed paper, restoring the intended gating. The printed recipe remains available with `--architecture paper`. Noise uses Euler-Maruyama increments and decision thresholds use theoretical stationary references. All errors are retained. Omitted panels and protocol differences: [experiments](docs/experiments.md) · [reproduction](docs/reproduction.md).

Implementation: [network model](docs/model.md) · [mean-field theory](docs/meanfield.md). Equinox/JAX handles recurrent dynamics; NumPy/SciPy handles Gaussian mean field; Snakemake manages artifacts. No training, downloaded models, or Rust build is needed.
