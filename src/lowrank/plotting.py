"""Standalone, publication-style task panels from saved simulation results."""

import json
from pathlib import Path

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
            else:
                # Same stimulus pair in the two contexts (cA=2/3,cB=1/6).
                selected = [29, 78]
            for i, condition in enumerate(selected):
                _trace(axes[0], times, traces[:, condition], str(data["labels"][condition]), COLORS[i])
            axes[0].axvspan(start, end, color=".8", alpha=.25, zorder=-1)
            axes[0].set(xlabel=r"Time / $\tau$", ylabel="Readout − baseline", title="Population output")
            axes[0].legend(frameon=False, fontsize=8)
            if name == "integration":
                for context, ax in enumerate(axes[1:]):
                    values = data["decisions"].mean(axis=0)[context*49:(context+1)*49].reshape(7, 7)
                    image = ax.imshow(values.T, origin="lower", extent=(-1/12, 13/12, -1/12, 13/12),
                                      cmap="viridis", vmin=0, vmax=1, aspect="equal")
                    ax.set(title=f"Context {'AB'[context]}: P(Go)", xlabel="Mean strength A", ylabel="Mean strength B")
                    ax.set_xticks([0, .5, 1])
                    ax.set_yticks([0, .5, 1])
                fig.colorbar(image, ax=list(axes[1:]), shrink=.8, label="Response probability")
            else:
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
                    values = data["endpoints"]
                    mean = values.mean(axis=0)
                    sem = values.std(axis=0, ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.zeros_like(mean)
                    x = np.arange(4)
                    axes[2].bar(x, mean, yerr=sem, color=COLORS, width=.65, capsize=2)
                    axes[2].set_xticks(x, ["A / ctx A", "B / ctx A", "A / ctx B", "B / ctx B"], rotation=20)
                    axes[2].set(ylabel="End-stimulus readout", title="Context-dependent discrimination")
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
        fig.suptitle(f"{titles[name]}\nFinite-network recreation · N={meta['size']} · {meta['trials']} {repetitions} · {sampling} · seed={meta['seed']}", fontsize=12)
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return path
