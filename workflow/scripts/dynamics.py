"""Snakemake stages for resumable Figure 1–2 realizations."""
from lowrank.dynamics_figures import run_dynamics_trial, aggregate_dynamics

if snakemake.params.operation == "trial":
    run_dynamics_trial(snakemake.params.experiment, snakemake.output[0],
                       size=snakemake.params.size, seed=snakemake.params.seed,
                       trial_id=snakemake.params.trial, dt=snakemake.params.dt,
                       duration=snakemake.params.duration)
else:
    aggregate_dynamics(list(snakemake.input.shards), snakemake.output[0])
