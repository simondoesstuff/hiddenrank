# Figures 1–2: paper-size dynamics

- Snakemake schedules one checkpoint per bulk-disorder realization. Completed trials survive interruption; theory curves are solved only during aggregation.
- Figure 1: N=5000, 15 realizations, T=800, RK4 dt=0.1. Four conditions share one Gaussian bulk matrix per realization. Patterns are shared across the ensemble.
- Figure 2: N=3500, 20 realizations. Population currents are averaged across disorder **before** z-scoring and PCA.

## Memory and symmetry

Figure 1 retains six unit traces plus online, per-neuron tail moments; no full population trajectory is stored. Tail statistics use the second half of samples, including the final sample. Welford updates avoid cancellation near stationary states.

The autonomous tanh dynamics are odd: trajectories initialized at −m are exactly the negatives of those initialized at +m. Each positive trajectory therefore supplies both signs of overlap and identical spatial/temporal variances. Four conditions are integrated together by the same RK4 updates as the reference simulator.

Figure 2 retains each population trajectory because ensemble averaging precedes PCA (about 1.4 MB uncompressed per default T=20 trial).

## Random draws

Centered patterns use `default_rng(seed)`; independent disorder uses `SeedSequence([seed, trial_id, 1818])`. Patterns and Gaussian sampling laws match the previous implementation; bulk draws change to make trial checkpoints independent of scheduling and ensemble size.
