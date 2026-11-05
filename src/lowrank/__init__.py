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
    plot = sub.add_parser("plot", help="Render a saved dataset")
    plot.add_argument("input", type=Path)
    plot.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if args.experiment in {"spontaneous", "inputs"}:
            from lowrank.dynamics_figures import run_experiment
        else:
            from lowrank.experiments import run_experiment
        run_experiment(args.experiment, args.output, size=args.size, trials=args.trials,
                       seed=args.seed, dt=args.dt, duration=args.duration)
    else:
        import numpy as np
        with np.load(args.input) as data:
            experiment = str(data["experiment"]) if "experiment" in data else ""
        if experiment in {"spontaneous", "inputs"}:
            from lowrank.dynamics_figures import plot_experiment
        else:
            from lowrank.plotting import plot_experiment
        plot_experiment(args.input, args.output)
