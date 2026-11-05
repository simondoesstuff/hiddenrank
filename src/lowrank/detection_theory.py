"""Figure 4F: stationary detection folds and collective relaxation times."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult, brentq, minimize_scalar

from .meanfield import GaussianRankOne, gaussian_quadrature, solve_fixed_points


@dataclass(frozen=True)
class DetectionTheory:
    """Eq. 151 with independent Gaussian stimulus, readout, overlap and noise."""

    g: float = .8
    rn: float = 2.
    stimulus_std: float = 1.2
    readout_std: float = 1.2
    background_variance: float = 1.
    quadrature_order: int = 200

    def __post_init__(self) -> None:
        if not 0 <= self.g < 1 or self.rn <= 0 or self.stimulus_std <= 0 or self.readout_std < 0 or self.background_variance < 0:
            raise ValueError("invalid detection parameters")

    def moments(self, variance: float) -> tuple[float, float]:
        nodes, weights = gaussian_quadrature(self.quadrature_order)
        phi = np.tanh(np.sqrt(variance) * nodes)
        return float(weights @ (1 - phi**2)), float(weights @ phi**2)

    def parameters(self, rm: float, strength: float = 0.) -> GaussianRankOne:
        if rm < 0:
            raise ValueError("rm must be nonnegative")
        return GaussianRankOne(self.g, std_m=np.hypot(self.readout_std, rm),
                               std_n=np.hypot(self.stimulus_std, self.rn),
                               cov_mn=rm * self.rn,
                               var_input=self.background_variance + (strength * self.stimulus_std)**2,
                               cov_n_input=strength * self.stimulus_std**2)

    def baseline_variance(self) -> float:
        """Variance of the trivial resting state, including quenched noise."""
        if self.g == 0:
            return self.background_variance
        return cast(float, brentq(lambda v: v - self.g**2 * self.moments(v)[1] - self.background_variance,
                            self.background_variance, self.background_variance + self.g**2))

    @property
    def critical_rm(self) -> float:
        return 1 / (self.rn * self.moments(self.baseline_variance())[0])

    @property
    def resting_critical_rm(self) -> float:
        """The zero-background resting bifurcation has structure strength one."""
        return 1 / self.rn

    def fold(self, rm: float) -> tuple[float, float, float]:
        """Return (input threshold, negative overlap, activation variance).

        Parameterize the stationary branch by its variance and maximize c
        between its trivial and nonzero c=0 endpoints. This bounded scalar
        search follows the same fold all the way to the pitchfork.
        """
        v0 = self.baseline_variance()
        overlap = rm * self.rn
        if rm <= self.critical_rm:
            return 0., 0., v0
        upper = max(2 * v0, 1.)
        while self.moments(upper)[0] * overlap > 1:
            upper *= 2
        vr = cast(float, brentq(lambda v: overlap * self.moments(v)[0] - 1, v0, upper))
        def branch(fraction: float) -> tuple[float, float, float]:
            v = v0 + fraction * (vr - v0)
            gain, phi2 = self.moments(v)
            a = 1 / gain - overlap
            kappa = -np.sqrt(max(0., v - self.g**2 * phi2 - self.background_variance) /
                             (self.readout_std**2 + rm**2 + a**2 / self.stimulus_std**2))
            return kappa * a / self.stimulus_std**2, float(kappa), v
        optimum = cast(OptimizeResult, minimize_scalar(lambda fraction: -branch(float(fraction))[0],
                                  bounds=(0., 1.), method="bounded", options={"xatol": 1e-12}))
        return branch(float(optimum.x))

    def resting_timescale(self, rm: float) -> tuple[float, float]:
        """Return (tau, lambda): tau = 1/(1-lambda), in neuron time units."""
        rest = replace(self, background_variance=0.)
        if rm <= self.resting_critical_rm:
            raise ValueError("resting Go branch requires rm above the critical value")
        p = rest.parameters(rm)
        v0 = rest.baseline_variance()
        upper = max(2 * v0, 1.)
        while rest.moments(upper)[0] * rm * self.rn > 1:
            upper *= 2
        variance = cast(float, brentq(lambda v: rest.moments(v)[0] * rm * self.rn - 1, v0, upper))
        kappa = np.sqrt((variance - self.g**2 * rest.moments(variance)[1]) /
                        p.std_m**2)
        points = solve_fixed_points(p, seeds=[(float(kappa), variance)], quadrature_order=self.quadrature_order)
        if len(points) != 1 or not points[0].stable:
            raise ValueError("no stable resting branch found")
        eigenvalue = points[0].outlier
        return 1 / (1 - eigenvalue), eigenvalue


def run_detection_theory(output: str | Path, *, points: int = 120,
                         parameters: DetectionTheory | None = None) -> Path:
    """Save the Fig. 4F curve; include a logarithmic approach to criticality."""
    if points < 4:
        raise ValueError("at least four curve points are required")
    p = DetectionTheory() if parameters is None else parameters
    if p.resting_critical_rm >= 2:
        raise ValueError("critical rm must be below the paper's upper endpoint rm=2")
    rm = p.resting_critical_rm + np.geomspace(1e-4, 2 - p.resting_critical_rm, points)
    folds = np.asarray([p.fold(float(value)) for value in rm])
    relaxation = np.asarray([p.resting_timescale(float(value)) for value in rm])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, rm=rm, structure_strength=rm * p.rn,
                        threshold=folds[:, 0], fold_kappa=folds[:, 1], fold_variance=folds[:, 2],
                        timescale=relaxation[:, 0], outlier=relaxation[:, 1],
                        critical_rm=p.resting_critical_rm, stimulus_critical_rm=p.critical_rm, g=p.g, rn=p.rn, stimulus_std=p.stimulus_std,
                        readout_std=p.readout_std, background_variance=p.background_variance)
    return output


def plot_detection_theory(source: str | Path, output: str | Path) -> None:
    """Draw Fig. 4F's threshold and effective timescale against structure strength."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(source) as data:
        fig, ax = plt.subplots(figsize=(5., 3.4), constrained_layout=True)
        other = ax.twinx()
        ax.plot(data["structure_strength"], data["threshold"], color="#d34b40", lw=2)
        other.plot(data["structure_strength"], data["timescale"], color="#454545", lw=2)
        ax.set(xlabel=r"Structure strength $\rho_m\rho_n$", ylabel="Detection threshold", ylim=(0, None))
        other.set(ylabel=r"Effective timescale $\tau/\tau_0$", ylim=(0, 50))
        ax.tick_params(axis="y", colors="#d34b40")
        ax.yaxis.label.set_color("#d34b40")
        ax.set_title("Figure 4F · threshold and timescale")
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
        plt.close(fig)
