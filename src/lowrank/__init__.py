"""Low-rank recurrent network figure reproductions."""


def main() -> None:
    """Run simulation or plotting stages used by Snakemake."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Generate a seeded figure dataset")
    run.add_argument("experiment", choices=["spontaneous", "inputs", "go_nogo", "detection", "context", "integration", "ring"])
    run.add_argument("output", type=Path)
    run.add_argument("--size", type=int, default=512)
    run.add_argument("--trials", type=int, default=4)
    run.add_argument("--seed", type=int, default=2018)
    run.add_argument("--dt", type=float, default=0.1)
    run.add_argument("--duration", type=float, default=40.)
    run.add_argument("--trial-start", type=int, default=0)
    run.add_argument("--condition-set", choices=["full", "representative"], default="full")
    run.add_argument("--architecture", choices=["supplement", "paper"], default="supplement")
    run.add_argument("--baseline-duration", type=float, default=None)
    run.add_argument("--transient-duration", type=float, default=10.)
    run.add_argument("--recovery-duration", type=float, default=0.)
    run.add_argument("--orthogonal-patterns", action="store_true", help="Enforce limiting pattern covariance in small task demonstrations")
    plot = sub.add_parser("plot", help="Render a saved dataset")
    plot.add_argument("input", type=Path)
    plot.add_argument("output", type=Path)
    aggregate = sub.add_parser("aggregate", help="Combine realization checkpoints")
    aggregate.add_argument("output", type=Path)
    aggregate.add_argument("inputs", type=Path, nargs="+")
    theory = sub.add_parser("detection-theory", help="Figure 4F threshold and timescale")
    theory.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "aggregate":
        from lowrank.experiments import aggregate_experiments
        aggregate_experiments(args.inputs, args.output)
        return
    if args.command == "detection-theory":
        from lowrank.detection_theory import run_detection_theory
        run_detection_theory(args.output)
        return
    if args.command == "run":
        if args.experiment in {"spontaneous", "inputs"}:
            from lowrank.dynamics_figures import run_experiment
        else:
            from lowrank.experiments import run_experiment
        kwargs = {} if args.experiment in {"spontaneous", "inputs"} else {"orthogonal": args.orthogonal_patterns,
            "trial_start": args.trial_start, "condition_set": args.condition_set,
            "architecture": args.architecture, "baseline_duration": args.baseline_duration,
            "transient_duration": args.transient_duration, "recovery_duration": args.recovery_duration}
        run_experiment(args.experiment, args.output, size=args.size, trials=args.trials,
                       seed=args.seed, dt=args.dt, duration=args.duration, **kwargs)
    else:
        import numpy as np
        with np.load(args.input) as data:
            experiment = str(data["experiment"]) if "experiment" in data else "detection_theory" if "fold_kappa" in data else ""
        if experiment in {"spontaneous", "inputs"}:
            from lowrank.dynamics_figures import plot_experiment
        elif experiment == "detection_theory":
            from lowrank.detection_theory import plot_detection_theory as plot_experiment
        else:
            from lowrank.plotting import plot_experiment
        plot_experiment(args.input, args.output)
