# Task recreations

| Run | Paper architecture | Recreated panels |
| --- | --- | --- |
| `go_nogo` | Eq. 145, S=2 | Go/Nogo traces, state projections, Eq. 149 mixture generalization |
| `detection` | Eq. 151, S=1.2, rm=rn=2 | Noisy traces, trajectories, psychometric curve |
| `context` | Eq. 155, rm=3, rn=1.6, bm=.6, bn=1 | Context-dependent traces, trajectories, response bars |
| `integration` | Eq. 155, rn=1.38 | Noisy traces, context-dependent response heatmaps |
| `ring` | Eq. 156, overlap=1.8, g=0 | Relaxation toward a rank-two ring |

- Dynamics use the full finite recurrent network, including quenched Gaussian connectivity (g=.8 except ring).
- Seeds, sizes, time step and repetitions are stored with compressed NPZ measurements. Plots never filter incorrect trials.
- Default experiment patterns use the paper's iid Gaussian sampling. Small networks can have substantial finite-size errors. Optional `orthogonal=True` in `run_experiment` centers and orthogonalizes Gaussian draws, enforcing limiting covariance for small demonstrations; metadata records this choice.
- Noise is independent per integration step: sigma=.4 for detection and .3 for integration. Changing dt changes its integrated power; these are discrete noisy-input recreations, not a continuous-time diffusion limit.
- Negative initial states relax for 15 time constants before stimulation. Stimuli last 20 by default; recovery lasts 10. Context is present throughout. Detection includes unit-variance quenched orthogonal input.
- Go decisions use half the **measured** positive-minus-negative resting readout, rather than the paper's theoretical upper-state prediction. Both attractors are simulated independently. Go/Nogo has no positive spontaneous attractor, so its decision reference is the separately simulated driven Go state. Raw endpoints and reference states are saved for reanalysis.
- Bands/error bars on readout plots show SEM over realizations; detection probabilities show binomial standard errors. Small trial counts give coarse probability estimates.
- Paper population sizes: Fig. 3 N=2500; Fig. 4 N=2500; Fig. 5 trajectories N=3000; Fig. 6 trajectories N=4000. Fig. 5/6 performance uses N=7500 and 50 realizations. Reduced workflow profiles target qualitative panel recreation, not published error bars.

The recreated panels omit PCA/connectivity-correlation analyses, theoretical trajectory overlays and the Fig. 4F bifurcation/timescale curve. Ring symmetry is exact in the population limit; finite networks can select preferred angles.
