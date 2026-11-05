"""Standalone, publication-style task panels from saved simulation results."""

import json
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#2675a3", "#c65452", "#319276", "#a478b8"]


def _trace(ax, times, values, label, color):
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
    ax.plot(times, mean, label=label, color=color, lw=1.8)
    ax.fill_between(times, mean-sem, mean+sem, color=color, alpha=.16, linewidth=0)


def _integration_selection(data):
    """Nearest available (0.6, 0.1) pair separately in each context."""
    selected = []
    for context in np.unique(data["contexts"]):
        indices = np.flatnonzero(data["contexts"] == context)
        distance = np.sum((data["stimulus"][indices] - [.6, .1])**2, axis=1)
        selected.append(int(indices[np.argmin(distance)]))
    return selected



def _integration_groups(data, key):
    """Per-realization paths, with equal weight for each irrelevant strength.

    Average duplicate samples within a stimulus pair first, so neither archive
    order nor repeated conditions change the weight of a nuisance strength.
    Sparse representative archives use the nuisance strengths they contain.
    """
    groups = []
    for context in np.unique(data["contexts"]):
        relevant = int(context)
        indices = np.flatnonzero(data["contexts"] == context)
        for strength in np.unique(data["stimulus"][indices, relevant]):
            selected = indices[data["stimulus"][indices, relevant] == strength]
            nuisance = data["stimulus"][selected, 1 - relevant]
            paths = [data[key][:, selected[nuisance == value]].mean(axis=1)
                     for value in np.unique(nuisance)]
            groups.append((int(context), float(strength), np.stack(paths).mean(axis=0)))
    return groups


def _response_grid(stimulus, responses):
    """A complete Cartesian grid, or None for a representative sparse subset."""
    a, b = (np.unique(stimulus[:, i]) for i in range(2))
    if len(stimulus) != len(a) * len(b) or len(np.unique(stimulus, axis=0)) != len(stimulus):
        return None
    values = np.empty((len(b), len(a)))
    for pair, response in zip(stimulus, responses, strict=True):
        values[np.searchsorted(b, pair[1]), np.searchsorted(a, pair[0])] = response
    return a, b, values


def _integration_ring(g=.8, architecture="paper"):
    """Eq. 156 Gaussian scaffold, before adding readout/context/stimulus terms."""
    from scipy.optimize import brentq
    from .meanfield import gaussian_expectation

    if architecture not in ("paper", "supplement"):
        raise ValueError(f"Unknown contextual architecture: {architecture}")
    rm, rn = (1.38, 3.) if architecture == "supplement" else (3., 1.38)
    overlap = rm * rn
    variance = cast(float, brentq(lambda v: overlap * gaussian_expectation(
        lambda z: 1 - np.tanh(z)**2, 0., v) - 1, 0., 100.))
    rate_variance = gaussian_expectation(lambda z: np.tanh(z)**2, 0., variance)
    radius = np.sqrt(max(0., variance - g*g*rate_variance) / (1.2**2 + rm**2))
    angle = np.linspace(0., 2*np.pi, 201)
    return radius * np.column_stack([np.cos(angle), np.sin(angle)])


def _integration_figure(data, meta):
    """Fig. 6 C–F: outputs, state projections, decisions, recurrent overlaps."""
    fig = plt.figure(figsize=(14, 8.5))
    grid = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[1, 2])]
    fig.subplots_adjust(left=.07, right=.945, bottom=.09, top=.85, hspace=.55, wspace=.45)
    coordinate_key = "scaffold_coordinates" if "scaffold_coordinates" in data else "kappa"
    selected = _integration_selection(data)
    times = data["times"]
    interval = (times >= meta["baseline_duration"]) & (times <= meta["stimulus_end"])
    axes[0].axvspan(meta["baseline_duration"], meta["stimulus_end"], color=".8", alpha=.25)
    for i, condition in enumerate(selected):
        color = COLORS[i % len(COLORS)]
        context = int(data["contexts"][condition])
        label = f"Context {'AB'[context]}: " + ", ".join(
            f"{channel}={value:.2g}" for channel, value in zip("AB", data["stimulus"][condition], strict=True))
        _trace(axes[0], times, data["traces"][:, condition], label, color)
    strengths = np.unique(data["stimulus"])
    colors = {float(value): plt.colormaps["viridis"](position)
              for value, position in zip(strengths, np.linspace(.1, .9, len(strengths)), strict=True)}
    for context, strength, paths in _integration_groups(data, "projections"):
        ax = axes[context + 1]
        color = colors[strength]
        label = f"{'AB'[context]}={strength:.2g}"
        # Match the paper's relevant-input/readout plane after marginalizing
        # over the irrelevant channel; failed Go trials remain in the mean.
        for path in paths:
            trajectory = path[interval][:, [context, 2]]
            ax.plot(*trajectory.T, color=color, alpha=.2, lw=.7)
        trajectory = paths.mean(axis=0)[interval][:, [context, 2]]
        ax.plot(*trajectory.T, color=color, lw=2, label=label)
        ax.scatter(*trajectory[0], color=color, marker="o", s=16)
        ax.scatter(*trajectory[-1], color=color, marker="^", s=24)
        ax.set(xlabel=rf"$I_{'AB'[context]}$ projection", ylabel=r"$w$ projection",
               title=f"Context {'AB'[context]}: population state")
        ax.legend(frameon=False, fontsize=7)
    if coordinate_key in data:
        groups = _integration_groups(data, coordinate_key)
        displayed = set()
        for context in np.unique(data["contexts"]):
            available = np.array([strength for group_context, strength, _ in groups
                                  if group_context == context])
            displayed.update((int(context), float(available[np.argmin(abs(available - target))]))
                             for target in [.1, .4, .6, .9])
        for context, strength, paths in groups:
            if (context, strength) not in displayed:
                continue
            color = colors[strength]
            style = "-" if context == 0 else "--"
            for path in paths:
                axes[5].plot(*path[interval].T, color=color, ls=style, alpha=.15, lw=.7)
            trajectory = paths.mean(axis=0)[interval]
            axes[5].plot(*trajectory.T, color=color, ls=style, lw=2,
                         label=f"{'AB'[context]}={strength:.2g}")
            axes[5].scatter(*trajectory[0], color=color, s=16)
            axes[5].scatter(*trajectory[-1], color=color, marker="^", s=24)
    axes[0].set(xlabel=r"Time / $\tau$", ylabel="Readout − baseline", title="Population output (all trials)")
    axes[0].legend(frameon=False, fontsize=8)
    probability = data["decisions"].mean(axis=0)
    for context, ax in enumerate(axes[3:5]):
        indices = np.flatnonzero(data["contexts"] == context)
        stimulus = data["stimulus"][indices]
        result = _response_grid(stimulus, probability[indices])
        if result is not None and min(len(result[0]), len(result[1])) > 1:
            a, b, values = result
            mesh = ax.pcolormesh(a, b, values, shading="nearest", cmap="viridis", vmin=0, vmax=1)
            ax.set(xlabel="Mean strength A", ylabel="Mean strength B")
            fig.colorbar(mesh, ax=ax, shrink=.75, label="P(Go)")
        else:
            ax.bar(np.arange(len(indices)), probability[indices], color=COLORS[context])
            ax.set_xticks(np.arange(len(indices)), [f"{a:g}, {b:g}" for a, b in stimulus], rotation=30)
            ax.set(xlabel="Sampled (A, B) strengths", ylabel="P(Go)", ylim=(0, 1.05))
        ax.set_title(f"Context {'AB'[context]}: response probability")
    if coordinate_key in data:
        ring = data["ring_background"] if "ring_background" in data else _integration_ring(meta.get("g", .8), meta.get("architecture", "paper"))
        axes[5].plot(*ring.T, color=".75", lw=2, label="Eq. 156 scaffold (mean field)", zorder=-1)
        limits = axes[5].get_xlim()
        axes[5].plot(limits, limits, ls="--", color=".75", lw=.8, zorder=-2)
        axes[5].set(xlabel=r"$\kappa_1=\langle n^{(1)}\tanh x\rangle$",
                    ylabel=r"$\kappa_2=\langle n^{(2)}\tanh x\rangle$", title="Recurrent population overlaps")
        if coordinate_key == "scaffold_coordinates":
            axes[5].set(xlabel=r"$m^{(1)}$ scaffold coordinate",
                        ylabel=r"$m^{(2)}$ scaffold coordinate", title="Population scaffold trajectories")
        axes[5].set_aspect("equal", adjustable="datalim")
        axes[5].legend(frameon=False, fontsize=6, ncol=2)
    else:
        axes[5].axis("off")
        axes[5].text(.5, .5, "Recurrent overlaps unavailable\nRegenerate simulation data", ha="center", va="center")
    return fig


def plot_experiment(source: str | Path, output: str | Path) -> Path:
    """Render PNG or SVG with all parameters taken from the input archive."""
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    meta = json.loads(str(data["metadata"]))
    name = meta["name"]
    with plt.rc_context({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.dpi": 140}):
        if name == "ring":
            fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
            for trajectory in data["trajectories"]:
                ax.plot(*trajectory.T, alpha=.8)
                ax.scatter(*trajectory[-1], s=25)
            ax.set(xlabel=r"$\kappa_1$", ylabel=r"$\kappa_2$", title="Rank-two ring: relaxation from different angles")
            ax.set_aspect("equal", adjustable="datalim")
        elif name == "integration":
            fig = _integration_figure(data, meta)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
            fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.76, wspace=.44)
            times = data["times"]
            start, end = meta["baseline_duration"], meta["stimulus_end"]
            traces = data["traces"]
            if name == "go_nogo":
                selected = [0, 1]
            elif name == "detection":
                selected = [3, 5, 7]
            elif name == "context":
                selected = [0, 1, 2, 3]
            for i, condition in enumerate(selected):
                _trace(axes[0], times, traces[:, condition], str(data["labels"][condition]), COLORS[i])
            axes[0].axvspan(start, end, color=".8", alpha=.25, zorder=-1)
            axes[0].set(xlabel=r"Time / $\tau$", ylabel="Readout − baseline", title="Population output")
            axes[0].legend(frameon=False, fontsize=8)
            if name == "context":
                axes[1].remove()
                axes[1] = fig.add_subplot(1, 3, 2, projection="3d")
            for i, condition in enumerate(selected):
                trajectory = data["projections"][:, condition].mean(axis=0)
                axes[1].plot(*trajectory.T, color=COLORS[i], lw=1.8)
                axes[1].scatter(*trajectory[-1], color=COLORS[i], s=18)
            axes[1].set(xlabel="Projection on input A" if name != "detection" else "Projection on stimulus",
                        ylabel="Projection on input B" if name == "context" else "Projection on recurrent m" if name == "detection" else "Projection on readout w",
                        title="Population state trajectory")
            if name == "context":
                axes[1].set_xlabel(r"$I_A$ projection", labelpad=2)
                axes[1].set_ylabel(r"$I_B$ projection", labelpad=2)
                axes[1].set_zlabel(r"$w$ projection", labelpad=4)
                axes[1].view_init(elev=24, azim=-54)
                axes[1].tick_params(labelsize=8)
            if name == "go_nogo":
                values = data["endpoints"][:, 2:]
                mean = values.mean(axis=0)
                sem = values.std(axis=0, ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.zeros_like(mean)
                axes[2].errorbar(data["axis"][2:], mean, yerr=sem, marker="o", color=COLORS[0], capsize=2)
                axes[2].set(xlabel=r"Mixture $\alpha$", ylabel="End-stimulus readout", title=r"Generalization: $n=I_A+I_B$")
            elif name == "detection":
                probability = data["decisions"].mean(axis=0)
                error = np.sqrt(probability*(1-probability)/meta["trials"])
                axes[2].errorbar(data["axis"], probability, yerr=error, color=COLORS[0], marker="o", capsize=2)
                axes[2].set(xlabel="Mean stimulus strength c", ylabel="P(Go)", ylim=(-.05, 1.05), title="Noisy detection")
            else:
                probability = data["decisions"].mean(axis=0)
                error = np.sqrt(probability*(1-probability)/meta["trials"])
                x = np.arange(4)
                axes[2].bar(x, probability, yerr=error, color=COLORS, width=.65, capsize=2)
                axes[2].set_xticks(x, ["A / ctx A", "B / ctx A", "A / ctx B", "B / ctx B"], rotation=20)
                axes[2].set(ylabel="P(Go)", ylim=(-.05, 1.05), title="Context-dependent discrimination")
        titles = {"go_nogo": "Figure 3 · Go / Nogo and generalization (Eq. 145)",
                  "detection": "Figure 4 · Detection of a noisy continuous stimulus (Eq. 151)",
                  "context": "Figure 5 · Context-dependent discrimination (Eq. 155)",
                  "integration": "Figure 6 · Context-dependent evidence integration (Eq. 155)",
                  "ring": "Rank-two continuous attractor (Eq. 156)"}
        sampling = "orthogonal patterns" if meta.get("orthogonal_patterns") else "iid Gaussian patterns"
        repetitions = "realization" if meta["trials"] == 1 else "realizations"
        if name == "context":
            # Matplotlib's constrained layout cannot reserve 3D axis labels.
            fig.set_layout_engine(None)
            fig.subplots_adjust(left=.065, right=.985, bottom=.23, top=.73, wspace=.44)
        architecture = f" · {meta['architecture']} architecture" if "architecture" in meta else ""
        fig.suptitle(f"{titles[name]}\nFinite-network recreation · N={meta['size']} · {meta['trials']} {repetitions} · {sampling} · seed={meta['seed']}{architecture}", fontsize=12)
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return path
