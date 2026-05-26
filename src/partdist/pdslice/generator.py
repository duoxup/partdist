"""Parametrised distribution generators for SliceDistribution.

Produces zero-mean, uncorrelated geometric shapes from a small set of
primitives. Twiss matching, centroid shifts, and inter-axis correlations
are out of scope — apply manipulator routines after generation to set
those.

Shapes:
    - Gaussian:       1D normal with optional truncation (ASTRA-style `cut`)
    - Uniform:        1D hard-edged uniform on [mean - L/2, mean + L/2]
    - Plateau:        1D flat top with Fermi-Dirac soft edges of width `r`
    - RadialUniform:  joint 2D uniform on a disk of radius R (KV-like)
    - Isotropic:      joint (px, py, pz) on half-sphere |p|=p_mag, pz >= 0

Units:
    - positions:  metres (SI)
    - momenta:    eV/c  (matches SliceDistribution's stored unit)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class Gaussian:
    """1D Gaussian shape.

    Parameters
    ----------
    sig : float
        rms (= σ) of the input Gaussian (before truncation, if any).
    mean : float
        Centroid. Default 0.0; non-zero only physically meaningful on the
        pz axis (acts as the longitudinal momentum anchor).
    cut : float | None
        Truncation in σ units (ASTRA's ``C_sig_*``). If set, sampling
        draws from a Gaussian restricted to ``[mean - cut*sig, mean + cut*sig]``;
        the *resulting* rms is smaller than ``sig``.
    """
    sig: float
    mean: float = 0.0
    cut: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sig <= 0:
            raise ValueError(f"Gaussian.sig must be > 0, got {self.sig}")
        if self.cut is not None and self.cut <= 0:
            raise ValueError(f"Gaussian.cut must be > 0 if given, got {self.cut}")

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.cut is None:
            return rng.normal(loc=self.mean, scale=self.sig, size=n)
        from scipy.stats import truncnorm
        a, b = -self.cut, self.cut
        return truncnorm.rvs(
            a, b, loc=self.mean, scale=self.sig, size=n, random_state=rng,
        )


@dataclass(frozen=True)
class Uniform:
    """1D uniform shape on ``[mean - L/2, mean + L/2]``.

    L is FWHM = full support width (hard edges).
    """
    L: float
    mean: float = 0.0

    def __post_init__(self) -> None:
        if self.L <= 0:
            raise ValueError(f"Uniform.L must be > 0, got {self.L}")

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        half = self.L / 2.0
        return rng.uniform(low=self.mean - half, high=self.mean + half, size=n)


@dataclass(frozen=True)
class Plateau:
    """1D flat-top shape with Fermi-Dirac soft edges.

    PDF (symmetric):
        f(x) ∝ 1 / (1 + exp(2·(2|x - mean| - L) / r))

    L is FWHM (half-max width); the actual support extends ~5·r beyond.
    """
    L: float
    r: float
    mean: float = 0.0

    def __post_init__(self) -> None:
        if self.L <= 0:
            raise ValueError(f"Plateau.L must be > 0, got {self.L}")
        if self.r <= 0:
            raise ValueError(f"Plateau.r must be > 0, got {self.r}")

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Rejection-sample from the symmetric Fermi-Dirac plateau.

        f(x) = 1 / (1 + exp(2·(2|x| - L) / r))   (unnormalised, peak = 1)
        """
        half_L = self.L / 2.0
        r = self.r
        window_half = half_L + 5.0 * r
        out = np.empty(n, dtype=float)
        filled = 0
        oversample = max(1.0, 2.5 * window_half / self.L)
        while filled < n:
            trial_n = int((n - filled) * oversample) + 16
            x_try = rng.uniform(-window_half, window_half, size=trial_n)
            u = rng.uniform(0.0, 1.0, size=trial_n)
            f_x = 1.0 / (1.0 + np.exp(2.0 * (2.0 * np.abs(x_try) - self.L) / r))
            accepted = x_try[u <= f_x]
            take = min(n - filled, len(accepted))
            out[filled:filled + take] = accepted[:take]
            filled += take
        return out + self.mean


@dataclass(frozen=True)
class RadialUniform:
    """Joint 2D uniform shape on a disk of radius R centred at origin.

    Used on (x, y) via the ``transverse=`` parameter of ``make_slice``,
    or on (px, py) via ``transverse_momentum=``.
    """
    R: float

    def __post_init__(self) -> None:
        if self.R <= 0:
            raise ValueError(f"RadialUniform.R must be > 0, got {self.R}")

    def _sample2d(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Uniform on disk: r = R·√U, θ = 2πU₂ gives uniform area density."""
        u1 = rng.uniform(0.0, 1.0, size=n)
        u2 = rng.uniform(0.0, 1.0, size=n)
        radius = self.R * np.sqrt(u1)
        theta = 2.0 * np.pi * u2
        a = radius * np.cos(theta)
        b = radius * np.sin(theta)
        return a, b


@dataclass(frozen=True)
class Isotropic:
    """Joint (px, py, pz) on a half-sphere of radius p_mag with pz >= 0.

    Physically: isotropic emission into a half-sphere (e.g. cathode emission).
    Implies σ_px = σ_py = p_mag/√3, σ_pz = p_mag/(2√3), <pz> = p_mag/2.

    p_mag is in eV/c.
    """
    p_mag: float

    def __post_init__(self) -> None:
        if self.p_mag <= 0:
            raise ValueError(f"Isotropic.p_mag must be > 0, got {self.p_mag}")


# Type alias for independent-axis shapes (1D)
_Axis1D = Union[Gaussian, Uniform, Plateau]
