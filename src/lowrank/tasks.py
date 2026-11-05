"""Task architectures from Mastrogiuseppe & Ostojic (2018), Eqs. 145, 151, 155."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Task:
    """Factorized connectivity and externally addressable input directions."""

    m: NDArray[np.float64]
    n: NDArray[np.float64]
    directions: NDArray[np.float64]
    readout: NDArray[np.float64]
    labels: tuple[str, ...]
    g: float = 0.8


def patterns(size: int, count: int, seed: int = 0, *, orthogonal: bool = True):
    """Independent Gaussian draws; optionally enforce their limiting covariance.

    Orthogonalized draws have zero empirical means and unit empirical variances.
    This reduces covariance sampling errors in small demonstration networks.
    """
    if size <= count:
        raise ValueError("size must exceed the number of patterns")
    values = np.random.default_rng(seed).normal(size=(size, count))
    if orthogonal:
        values -= values.mean(axis=0)
        values, r = np.linalg.qr(values)
        values *= np.sign(np.diag(r))[None, :] * np.sqrt(size)
    return values


def go_nogo(size: int, seed: int = 0, *, orthogonal: bool = True) -> Task:
    """Eq. 145: input A overlaps n; the readout coincides with m (S=2)."""
    a, b, w = (2 * patterns(size, 3, seed, orthogonal=orthogonal)).T
    return Task(w[:, None], a[:, None], np.stack([a, b], axis=1), w, ("A", "B"))


def detection(size: int, seed: int = 0, *, rm: float = 2., rn: float = 2.,
              orthogonal: bool = True) -> Task:
    """Eq. 151: shared y generates bistability; y also modulates threshold."""
    stimulus, w, y, background = patterns(size, 4, seed, orthogonal=orthogonal).T
    stimulus, w = 1.2 * stimulus, 1.2 * w
    return Task((w + rm * y)[:, None], (stimulus + rn * y)[:, None],
                np.stack([stimulus, y, background], axis=1), w,
                ("stimulus", "modulation", "background"))


def contextual(size: int, seed: int = 0, *, integration: bool = False,
               orthogonal: bool = True) -> Task:
    """Eq. 155, including the distinct Fig. 5 and Fig. 6 values of rn."""
    a, b, ya, yb, ca, cb, w = patterns(size, 7, seed, orthogonal=orthogonal).T
    a, b, ya, yb = (1.2 * v for v in (a, b, ya, yb))
    rm, rn, bm, bn = 3., 1.38 if integration else 1.6, .6, 1.
    m = np.stack([ya + rm * ca + bm * w, yb + rm * cb + bm * w], axis=1)
    n = np.stack([a + rn * ca + bn * w, b + rn * cb + bn * w], axis=1)
    return Task(m, n, np.stack([a, b, ca, cb], axis=1), w,
                ("A", "B", "context A", "context B"))


def ring(size: int, seed: int = 0, *, strength: float = 1.8,
         orthogonal: bool = True) -> Task:
    """Rotationally symmetric rank-two connectivity (Eq. 156, g=0)."""
    ya, yb, ca, cb = patterns(size, 4, seed, orthogonal=orthogonal).T
    m = np.stack([ya + np.sqrt(strength) * ca, yb + np.sqrt(strength) * cb], axis=1)
    n = np.sqrt(strength) * np.stack([ca, cb], axis=1)
    return Task(m, n, np.stack([ca, cb], axis=1), ca, ("A", "B"), g=0.)
